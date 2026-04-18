"""
A* pathfinding visualizer → cinematic MP4.
Wilmette → SW Chicago, rendered with multi-radius bloom, age-faded trails,
white-hot frontier, and a final glowing path reveal.
"""
from __future__ import annotations

import argparse
import hashlib
import math
import os
import pickle
import subprocess
import sys
import time
from dataclasses import dataclass
from heapq import heappush, heappop
from pathlib import Path

import cairo
import cv2
import numpy as np
import osmnx as ox
import networkx as nx

# ───────────────────────────── config ─────────────────────────────
ORIGIN_ADDR = "10 Post Office Square, Boston, MA 02109"
DEST_ADDR   = "2 Leighton Street, Cambridge, MA 02141"

# ── render profiles ──────────────────────────────────────────────────────────────────
# Two presets: SD (canonical 1920-bounded, 30 fps) and HQ (2560-bounded, 120 fps).
# Resolution-dependent values (bloom sigmas, line widths) scale with MAX_DIM so the HQ
# version looks identical, just crisper. Per-frame decay rates are expressed as
# per-second and converted to per-frame at runtime so motion feels the same at any fps.
@dataclass
class RenderProfile:
    name: str
    max_dim: int              # longest side of the output canvas in px
    fps: int
    out_path: Path
    crf: int                  # libx264 quality (lower = better)
    preset: str               # libx264 encoder preset
    # per-second rates (converted to per-frame at runtime)
    hot_cool_per_sec: float   # fraction of frontier brightness retained each second

SD_PROFILE = RenderProfile(
    name="sd",
    max_dim=1920,
    fps=30,
    out_path=Path("boston_astar_wide.mp4"),
    crf=16,
    preset="medium",
    hot_cool_per_sec=0.88 ** 30,   # preserves the existing look at 30 fps
)

HQ_PROFILE = RenderProfile(
    name="hq",
    max_dim=2560,                  # 1440p on the long side
    fps=120,
    out_path=Path("boston_astar_wide_hq.mp4"),
    crf=15,
    preset="slow",                 # worth the extra encode time given 4× the frames
    hot_cool_per_sec=0.88 ** 30,   # identical wall-clock cooling
)

# Selected at CLI-parse time. Render functions read from these module-level constants.
PROFILE: RenderProfile = SD_PROFILE

# All resolution-scaled values are derived relative to a 1920-px reference canvas so
# both profiles produce visually equivalent glow thickness, line weight, etc.
RES_REF = 1920

# These are (re)assigned from PROFILE in `apply_profile()`:
MAX_DIM = SD_PROFILE.max_dim
FPS = SD_PROFILE.fps
OUT_PATH = SD_PROFILE.out_path
MARGIN = 40

# WIDTH/HEIGHT are populated at runtime once the projected bbox is known.
WIDTH: int = 0
HEIGHT: int = 0

SIMPLIFY_GRAPH = False   # False → keep every OSM node; slower render but catches disconnected ramps

# Graph cache is keyed on the origin/dest pair so switching routes doesn't silently reuse a
# stale bbox. Hash kept short — collisions don't matter, just uniqueness within one machine.
def _cache_path_for(origin: str, dest: str) -> Path:
    key = hashlib.sha1(f"{origin}→{dest}|simplify={SIMPLIFY_GRAPH}".encode()).hexdigest()[:12]
    return Path(f".graph_cache_{key}.pkl")

CACHE_PATH = _cache_path_for(ORIGIN_ADDR, DEST_ADDR)

# bbox auto-framing: padding is proportional to the route span so a cross-town 2-mile hop and
# a 30-mile suburb-to-city trip both get a sensible amount of surrounding context. Values are
# in degrees; the orthogonal floors keep near-axis-aligned routes from rendering as a sliver.
PAD_MIN       = 0.012   # ~1.3 km floor — short urban routes still have breathing room
PAD_FRAC      = 0.35    # fraction of route span added to each side of the dominant axis
PAD_ORTHO_FRAC = 0.18   # minimum breathing room on the short axis, tied to the long axis span

# pacing: total video length in seconds for each phase
SPREAD_SECONDS = 32      # exploration animation
PATH_SECONDS   = 6       # final path reveal
HOLD_SECONDS   = 4       # hold on completed frame

# visuals
BG_COLOR = np.array([0.010, 0.004, 0.012], dtype=np.float32)   # near-black plum
ROAD_COLOR = np.array([0.38, 0.08, 0.04], dtype=np.float32) * 0.55  # dim crimson

# exploration color (white-hot frontier → amber trail via cooling)
HOT_COLOR   = np.array([1.00, 0.85, 0.55], dtype=np.float32) * 1.9
TRAIL_COLOR = np.array([1.00, 0.42, 0.08], dtype=np.float32) * 0.55

PATH_COLOR = np.array([1.00, 0.95, 0.75], dtype=np.float32) * 2.6
ENDPOINT_COLOR = np.array([1.00, 0.98, 0.90], dtype=np.float32) * 3.5

BLOOM_WEIGHTS = [0.55, 0.65, 0.55]

VIGNETTE_STRENGTH = 0.45
TONEMAP_EXPOSURE  = 1.15

# These scale with resolution and are populated by apply_profile():
ROAD_WIDTH: float = 0.9
PATH_WIDTH: float = 2.4
BLOOM_SIGMAS: list[float] = [3.0, 12.0, 40.0]
HOT_COOL_RATE: float = 0.88   # per-frame multiplier, derived from profile's per-second rate


def apply_profile(profile: RenderProfile) -> None:
    """Wire the selected profile into module-level constants. Scales all resolution-
    dependent visual parameters so the HQ render looks identical, just sharper."""
    global PROFILE, MAX_DIM, FPS, OUT_PATH
    global ROAD_WIDTH, PATH_WIDTH, BLOOM_SIGMAS, HOT_COOL_RATE

    PROFILE = profile
    MAX_DIM  = profile.max_dim
    FPS      = profile.fps
    OUT_PATH = profile.out_path

    # line widths and bloom kernel sizes scale linearly with the canvas long-side so the
    # visual appearance stays constant across resolutions
    k = profile.max_dim / RES_REF
    ROAD_WIDTH   = 0.9 * k
    PATH_WIDTH   = 2.4 * k
    BLOOM_SIGMAS = [3.0 * k, 12.0 * k, 40.0 * k]

    # convert per-second cooling to per-frame so wall-clock cooling matches at any fps
    HOT_COOL_RATE = profile.hot_cool_per_sec ** (1.0 / profile.fps)


# ───────────────────────────── graph + A* ─────────────────────────────
def load_graph():
    if CACHE_PATH.exists():
        print(f"[graph] loading cache {CACHE_PATH}")
        with open(CACHE_PATH, "rb") as f:
            return pickle.load(f)

    print("[geo] geocoding addresses…")
    o_lat, o_lon = ox.geocode(ORIGIN_ADDR)
    d_lat, d_lon = ox.geocode(DEST_ADDR)
    print(f"  origin = ({o_lat:.5f}, {o_lon:.5f})")
    print(f"  dest   = ({d_lat:.5f}, {d_lon:.5f})")

    span_lat = abs(o_lat - d_lat)
    span_lon = abs(o_lon - d_lon)
    pad_lat = max(PAD_MIN, span_lat * PAD_FRAC, span_lon * PAD_ORTHO_FRAC)
    pad_lon = max(PAD_MIN, span_lon * PAD_FRAC, span_lat * PAD_ORTHO_FRAC)
    print(f"[geo] route span  Δlat={span_lat:.4f}°  Δlon={span_lon:.4f}°  → pad=({pad_lat:.4f}°, {pad_lon:.4f}°)")

    north = max(o_lat, d_lat) + pad_lat
    south = min(o_lat, d_lat) - pad_lat
    east  = max(o_lon, d_lon) + pad_lon
    west  = min(o_lon, d_lon) - pad_lon

    print(f"[osm] fetching drive network for bbox N{north:.3f} S{south:.3f} E{east:.3f} W{west:.3f}…")
    t0 = time.time()
    G = ox.graph_from_bbox(
        bbox=(west, south, east, north),
        network_type="drive",
        simplify=SIMPLIFY_GRAPH,
        retain_all=False,
    )
    print(f"  fetched {len(G.nodes)} nodes, {len(G.edges)} edges in {time.time()-t0:.1f}s")

    print("[osm] projecting to UTM…")
    G = ox.project_graph(G)

    print("[osm] annotating edges with speed + travel_time…")
    G = ox.add_edge_speeds(G)

    # Manual correction: OSM has Chicago motorways tagged conservatively (many at 55 mph).
    # Real-world Kennedy/Edens/Dan Ryan/Stevenson driving is ~75 mph. motorway_link covers both
    # slow cloverleaf ramps and fast through-ramps (e.g. the Circle Interchange Kennedy↔Stevenson
    # flyover, which flows at highway speed). OSM can't distinguish them, so a blended 65 mph
    # value reflects real drive time and prevents A* from wrongly detouring onto surface streets
    # to skip the interchange.
    MOTORWAY_KPH      = 75.0 * 1.60934   # ≈ 120.7
    MOTORWAY_LINK_KPH = 65.0 * 1.60934   # ≈ 104.6
    n_mw = n_link = 0
    for _, _, data in G.edges(data=True):
        hwy = data.get("highway")
        if isinstance(hwy, list):
            hwy = hwy[0]
        if hwy == "motorway":
            data["speed_kph"] = MOTORWAY_KPH
            n_mw += 1
        elif hwy == "motorway_link":
            data["speed_kph"] = MOTORWAY_LINK_KPH
            n_link += 1
    print(f"  set {n_mw} motorway edges → 75 mph, {n_link} motorway_link edges → 55 mph")

    G = ox.add_edge_travel_times(G)

    # project query points into the same CRS so nearest_nodes works on projected graph
    import geopandas as gpd
    from shapely.geometry import Point
    pts = gpd.GeoDataFrame(
        geometry=[Point(o_lon, o_lat), Point(d_lon, d_lat)],
        crs="EPSG:4326",
    ).to_crs(G.graph["crs"])
    o_x, o_y = pts.geometry.iloc[0].x, pts.geometry.iloc[0].y
    d_x, d_y = pts.geometry.iloc[1].x, pts.geometry.iloc[1].y

    origin_node = ox.distance.nearest_nodes(G, X=o_x, Y=o_y)
    dest_node   = ox.distance.nearest_nodes(G, X=d_x, Y=d_y)
    print(f"  origin node {origin_node}  dest node {dest_node}")

    payload = (G, origin_node, dest_node)
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(payload, f)
    return payload


def astar_trace(G: nx.MultiDiGraph, start: int, goal: int):
    """A* minimizing travel time. Heuristic = straight-line distance / top speed, in seconds."""
    gx, gy = G.nodes[goal]["x"], G.nodes[goal]["y"]

    # fastest edge in the network, in m/s; bounds the admissible heuristic.
    max_mps = max(d.get("speed_kph", 0.0) for _, _, d in G.edges(data=True)) * 1000.0 / 3600.0
    print(f"  heuristic top speed: {max_mps * 3.6:.1f} kph")

    def h(n):
        nx_, ny_ = G.nodes[n]["x"], G.nodes[n]["y"]
        return math.hypot(nx_ - gx, ny_ - gy) / max_mps

    g_score = {start: 0.0}
    came_from: dict[int, int] = {}
    open_heap = [(h(start), 0, start)]
    counter = 1
    explored: list[tuple[int, int]] = []  # (u, v) in relaxation order
    closed: set[int] = set()

    while open_heap:
        f, _, u = heappop(open_heap)
        if u in closed:
            continue
        closed.add(u)
        if u == goal:
            break
        for _, v, data in G.out_edges(u, data=True):
            w = data.get("travel_time") or (data["length"] / max(data.get("speed_kph", 40.0), 5.0) * 3.6)
            tentative = g_score[u] + w
            if tentative < g_score.get(v, float("inf")):
                g_score[v] = tentative
                came_from[v] = u
                explored.append((u, v))
                heappush(open_heap, (tentative + h(v), counter, v))
                counter += 1

    # reconstruct path
    path = [goal]
    while path[-1] in came_from:
        path.append(came_from[path[-1]])
    path.reverse()
    if path[0] != start:
        raise RuntimeError("no path found")
    return explored, path


# ───────────────────────────── projection ─────────────────────────────
@dataclass
class Projector:
    minx: float
    miny: float
    dy: float
    scale: float
    offx: float
    offy: float

    def __call__(self, x: float, y: float) -> tuple[float, float]:
        px = self.offx + (x - self.minx) * self.scale
        py = self.offy + (self.dy - (y - self.miny)) * self.scale
        return px, py


def build_projector(G) -> tuple[int, int, Projector]:
    """Size the canvas to the projected graph's extent: long side = MAX_DIM, short side
    scaled to preserve geographic aspect ratio. Returns (width, height, projector).
    Both dims are rounded to even numbers so libx264/yuv420p is happy."""
    xs = np.fromiter((d["x"] for _, d in G.nodes(data=True)), dtype=np.float64)
    ys = np.fromiter((d["y"] for _, d in G.nodes(data=True)), dtype=np.float64)
    dx, dy = xs.max() - xs.min(), ys.max() - ys.min()

    aspect = dx / dy
    if aspect >= 1.0:  # wider than tall
        width = MAX_DIM
        height = int(round(MAX_DIM / aspect))
    else:              # taller than wide
        height = MAX_DIM
        width = int(round(MAX_DIM * aspect))

    # force even dims for H.264
    width  &= ~1
    height &= ~1

    scale = min((width - 2 * MARGIN) / dx, (height - 2 * MARGIN) / dy)
    projector = Projector(
        minx=xs.min(), miny=ys.min(), dy=dy, scale=scale,
        offx=(width  - dx * scale) / 2,
        offy=(height - dy * scale) / 2,
    )
    return width, height, projector


# ───────────────────────────── cairo helpers ─────────────────────────────
def new_surface(alpha: bool = True) -> cairo.ImageSurface:
    return cairo.ImageSurface(cairo.FORMAT_ARGB32, WIDTH, HEIGHT)


def surface_to_rgb_f32(surf: cairo.ImageSurface) -> np.ndarray:
    """Cairo ARGB32 is BGRA little-endian. Full 3-channel readback, used only for the one-time base layer."""
    buf = np.frombuffer(surf.get_data(), dtype=np.uint8).reshape(HEIGHT, WIDTH, 4)
    return cv2.cvtColor(buf, cv2.COLOR_BGRA2RGB).astype(np.float32) * (1.0 / 255.0)


def surface_alpha_f32(surf: cairo.ImageSurface) -> np.ndarray:
    """Grab just the alpha channel as [H,W] float32. Strokes use a single solid color, so alpha
    is sufficient — the color tint is applied at use-site via broadcasting."""
    buf = np.frombuffer(surf.get_data(), dtype=np.uint8).reshape(HEIGHT, WIDTH, 4)
    return buf[..., 3].astype(np.float32) * (1.0 / 255.0)


# Pyramidal separable-gaussian approximation. Identical to scipy.ndimage.gaussian_filter within
# 1/255 for the sigmas we use, but up to ~350× faster at σ=40 because the big blur runs on a
# downsampled image (gaussian and decimation commute for frequencies below Nyquist).
def fast_blur_rgb(img: np.ndarray, sigma: float) -> np.ndarray:
    if sigma < 8.0:
        return cv2.GaussianBlur(img, (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REPLICATE)
    ds = 4 if sigma < 25.0 else 8
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(w // ds, 1), max(h // ds, 1)), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), sigmaX=sigma / ds, sigmaY=sigma / ds, borderType=cv2.BORDER_REPLICATE)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def draw_edge_geom(ctx: cairo.Context, edge_data, u_xy, v_xy, proj: Projector):
    """Draw an edge using its geometry linestring if available, else straight line."""
    geom = edge_data.get("geometry")
    if geom is not None:
        xs, ys = geom.xy
        px, py = proj(xs[0], ys[0])
        ctx.move_to(px, py)
        for xx, yy in zip(xs[1:], ys[1:]):
            px, py = proj(xx, yy)
            ctx.line_to(px, py)
    else:
        ux, uy = proj(*u_xy)
        vx, vy = proj(*v_xy)
        ctx.move_to(ux, uy)
        ctx.line_to(vx, vy)


# ───────────────────────────── render ─────────────────────────────
def render_video():
    global WIDTH, HEIGHT
    G, origin_node, dest_node = load_graph()
    WIDTH, HEIGHT, proj = build_projector(G)
    print(f"[render] canvas: {WIDTH}×{HEIGHT}")

    print("[astar] running A*…")
    t0 = time.time()
    explored, path = astar_trace(G, origin_node, dest_node)
    print(f"  explored {len(explored)} edges, path has {len(path)} nodes, dt={time.time()-t0:.2f}s")

    # ── base layer: dim road network, rendered once ─────────────────────
    print("[render] drawing base road layer…")
    base_surf = new_surface()
    ctx = cairo.Context(base_surf)
    ctx.set_source_rgb(*BG_COLOR)
    ctx.paint()
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    ctx.set_line_join(cairo.LINE_JOIN_ROUND)
    ctx.set_line_width(ROAD_WIDTH)
    ctx.set_source_rgba(*ROAD_COLOR, 1.0)
    for u, v, data in G.edges(data=True):
        ux, uy = G.nodes[u]["x"], G.nodes[u]["y"]
        vx, vy = G.nodes[v]["x"], G.nodes[v]["y"]
        draw_edge_geom(ctx, data, (ux, uy), (vx, vy), proj)
    ctx.stroke()
    base_rgb = surface_to_rgb_f32(base_surf).copy()

    # ── precompute an edge geometry cache (projected pixel polylines) ───
    print("[render] caching edge polylines in pixel space…")
    edge_polylines: dict[tuple[int, int], list[tuple[float, float]]] = {}
    for u, v, data in G.edges(data=True):
        ux, uy = G.nodes[u]["x"], G.nodes[u]["y"]
        vx, vy = G.nodes[v]["x"], G.nodes[v]["y"]
        geom = data.get("geometry")
        if geom is not None:
            xs, ys = geom.xy
            pts = [proj(x, y) for x, y in zip(xs, ys)]
        else:
            pts = [proj(ux, uy), proj(vx, vy)]
        edge_polylines[(u, v)] = pts

    # ── pacing schedule (ease in-out cubic) ─────────────────────────────
    spread_frames = int(SPREAD_SECONDS * FPS)
    path_frames   = int(PATH_SECONDS * FPS)
    hold_frames   = int(HOLD_SECONDS * FPS)
    total_frames  = spread_frames + path_frames + hold_frames

    total_edges = len(explored)
    def ease(t):  # cubic in-out
        return 3 * t * t - 2 * t * t * t
    edge_cumulative = [int(ease((i + 1) / spread_frames) * total_edges) for i in range(spread_frames)]
    edge_cumulative[-1] = total_edges

    # ── ffmpeg pipe ─────────────────────────────────────────────────────
    if OUT_PATH.exists():
        OUT_PATH.unlink()
    ff_cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{WIDTH}x{HEIGHT}", "-r", str(FPS),
        "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-crf", str(PROFILE.crf), "-preset", PROFILE.preset,
        "-movflags", "+faststart",
        str(OUT_PATH),
    ]
    ff = subprocess.Popen(ff_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    assert ff.stdin is not None

    # ── vignette mask ───────────────────────────────────────────────────
    yy, xx = np.mgrid[0:HEIGHT, 0:WIDTH].astype(np.float32)
    cx, cy = WIDTH / 2, HEIGHT / 2
    r = np.sqrt(((xx - cx) / cx) ** 2 + ((yy - cy) / cy) ** 2)
    vignette = np.clip(1.0 - VIGNETTE_STRENGTH * (r ** 2), 0.0, 1.0)[..., None]

    # ── buffers ─────────────────────────────────────────────────────────
    trail_buffer = np.zeros((HEIGHT, WIDTH, 3), dtype=np.float32)  # persistent amber
    hot_buffer   = np.zeros((HEIGHT, WIDTH, 3), dtype=np.float32)  # fading white-hot frontier
    path_buffer  = np.zeros((HEIGHT, WIDTH, 3), dtype=np.float32)  # final path glow

    # reusable scratch surface for per-frame stroke
    scratch_surf = new_surface()
    scratch_ctx  = cairo.Context(scratch_surf)
    scratch_ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    scratch_ctx.set_line_join(cairo.LINE_JOIN_ROUND)

    def _clear_rect(x0, y0, x1, y1):
        # clip + snap out to integer bounds, small margin so round caps fully erase
        x0 = max(0.0, math.floor(x0) - 2.0)
        y0 = max(0.0, math.floor(y0) - 2.0)
        x1 = min(float(WIDTH),  math.ceil(x1) + 2.0)
        y1 = min(float(HEIGHT), math.ceil(y1) + 2.0)
        if x1 <= x0 or y1 <= y0:
            return
        scratch_ctx.save()
        scratch_ctx.rectangle(x0, y0, x1 - x0, y1 - y0)
        scratch_ctx.clip()
        scratch_ctx.set_operator(cairo.OPERATOR_CLEAR)
        scratch_ctx.paint()
        scratch_ctx.set_operator(cairo.OPERATOR_OVER)
        scratch_ctx.restore()

    # track the dirty region we last wrote to so we only need to clear that much
    last_dirty: list[float] | None = None  # [x0,y0,x1,y1] or None → clear nothing

    def stroke_edges_alpha(edges, width) -> tuple[np.ndarray, list[float] | None]:
        """Stroke edges as a solid opaque line; return (alpha_mask[H,W] float32, bbox)."""
        nonlocal last_dirty
        if last_dirty is None:
            scratch_ctx.set_operator(cairo.OPERATOR_CLEAR); scratch_ctx.paint()
            scratch_ctx.set_operator(cairo.OPERATOR_OVER)
        else:
            _clear_rect(*last_dirty)

        pad = width * 0.6 + 1.0
        x0 = y0 = math.inf
        x1 = y1 = -math.inf
        scratch_ctx.set_line_width(width)
        scratch_ctx.set_source_rgba(1.0, 1.0, 1.0, 1.0)
        any_drawn = False
        for key in edges:
            pts = edge_polylines.get(key)
            if pts is None:
                continue
            any_drawn = True
            scratch_ctx.move_to(*pts[0])
            for p in pts[1:]:
                scratch_ctx.line_to(*p)
            for px, py in pts:
                if px < x0: x0 = px
                if py < y0: y0 = py
                if px > x1: x1 = px
                if py > y1: y1 = py
        scratch_ctx.stroke()
        scratch_surf.flush()
        if not any_drawn:
            last_dirty = None
            return np.zeros((HEIGHT, WIDTH), dtype=np.float32), None
        bbox = [x0 - pad, y0 - pad, x1 + pad, y1 + pad]
        last_dirty = bbox
        return surface_alpha_f32(scratch_surf), bbox

    # scratch buffers for the composite — allocate once, reuse every frame
    composite_bright = np.empty((HEIGHT, WIDTH, 3), dtype=np.float32)
    composite_img    = np.empty((HEIGHT, WIDTH, 3), dtype=np.float32)

    def composite_and_pipe():
        bright = composite_bright
        img    = composite_img

        np.copyto(bright, trail_buffer)
        bright += hot_buffer
        bright += path_buffer

        blur0 = fast_blur_rgb(bright, BLOOM_SIGMAS[0])
        blur1 = fast_blur_rgb(bright, BLOOM_SIGMAS[1])
        blur2 = fast_blur_rgb(bright, BLOOM_SIGMAS[2])

        np.copyto(img, base_rgb)
        img += bright
        img += blur0 * BLOOM_WEIGHTS[0]
        img += blur1 * BLOOM_WEIGHTS[1]
        img += blur2 * BLOOM_WEIGHTS[2]
        img *= vignette

        # tonemap in-place: 1 - exp(-img * exposure)
        np.multiply(img, -TONEMAP_EXPOSURE, out=img)
        np.exp(img, out=img)
        np.subtract(1.0, img, out=img)

        out = cv2.convertScaleAbs(img, alpha=255.0)
        ff.stdin.write(out.tobytes())

    # endpoint markers (always-on) — rendered once into a persistent layer
    origin_xy = proj(G.nodes[origin_node]["x"], G.nodes[origin_node]["y"])
    dest_xy   = proj(G.nodes[dest_node]["x"],   G.nodes[dest_node]["y"])

    def stamp_endpoints() -> np.ndarray:
        nonlocal last_dirty
        scratch_ctx.set_operator(cairo.OPERATOR_CLEAR); scratch_ctx.paint()
        scratch_ctx.set_operator(cairo.OPERATOR_OVER)
        scratch_ctx.set_source_rgba(1.0, 1.0, 1.0, 1.0)
        for (px, py) in (origin_xy, dest_xy):
            scratch_ctx.arc(px, py, 4.5, 0, 2 * math.pi)
            scratch_ctx.fill()
        scratch_surf.flush()
        mask = surface_alpha_f32(scratch_surf)
        last_dirty = None  # full clear on next stroke
        # bake directly into path_buffer as RGB: mask × color
        return mask[..., None] * ENDPOINT_COLOR

    endpoint_layer = stamp_endpoints()

    # ── phase 1: exploration ────────────────────────────────────────────
    # Stroke the new edges ONCE at hot-width per frame; reuse the same alpha mask for the
    # cooler trail tint. Hot and trail had different widths before, but at 1.6 vs 1.2 the
    # visual difference is imperceptible against the bloom halo — one stroke saves ~12 ms/frame.
    print(f"[render] phase 1: exploration ({spread_frames} frames)")
    path_buffer = np.maximum(path_buffer, endpoint_layer)
    t0 = time.time()
    idx = 0
    for i in range(spread_frames):
        target = edge_cumulative[i]
        new_edges = explored[idx:target]
        idx = target

        if new_edges:
            mask, _ = stroke_edges_alpha(new_edges, 1.6)
            # hot: additive (frontier stacks), trail: max (saturating trail)
            mask_rgb = mask[..., None]
            hot_buffer += mask_rgb * HOT_COLOR
            np.maximum(trail_buffer, mask_rgb * TRAIL_COLOR, out=trail_buffer)

        hot_buffer *= HOT_COOL_RATE
        composite_and_pipe()
        if i % 60 == 0 or i == spread_frames - 1:
            elapsed = time.time() - t0
            print(f"  frame {i+1}/{spread_frames}  edges {idx}/{total_edges}  {elapsed:.1f}s", flush=True)

    # ── phase 2: path reveal ────────────────────────────────────────────
    # The path stroke grows over time; rather than restroking from scratch each frame, we track
    # how many edges are newly visible and only stroke those, then max-merge into path_buffer.
    print(f"[render] phase 2: path reveal ({path_frames} frames)", flush=True)
    path_edges = list(zip(path[:-1], path[1:]))
    last_revealed = 0
    for i in range(path_frames):
        frac = (i + 1) / path_frames
        frac_eased = 1 - (1 - frac) ** 2
        n_reveal = max(1, int(frac_eased * len(path_edges)))
        new_path_edges = path_edges[last_revealed:n_reveal]
        last_revealed = n_reveal

        if new_path_edges:
            mask, _ = stroke_edges_alpha(new_path_edges, PATH_WIDTH)
            np.maximum(path_buffer, mask[..., None] * PATH_COLOR, out=path_buffer)

        np.maximum(path_buffer, endpoint_layer, out=path_buffer)
        hot_buffer *= HOT_COOL_RATE
        composite_and_pipe()

    # ── phase 3: hold ───────────────────────────────────────────────────
    print(f"[render] phase 3: hold ({hold_frames} frames)", flush=True)
    for _ in range(hold_frames):
        hot_buffer *= 0.995
        composite_and_pipe()

    ff.stdin.close()
    err = ff.stderr.read().decode(errors="ignore")
    ret = ff.wait()
    if ret != 0:
        print(err, file=sys.stderr)
        raise RuntimeError(f"ffmpeg exited {ret}")
    print(f"[done] wrote {OUT_PATH} ({total_frames} frames @ {FPS}fps)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A* pathfinding visualizer → MP4")
    parser.add_argument(
        "--hq", action="store_true",
        help="render the 1440p-bounded / 120 fps HQ version to chicago_astar_wide_hq.mp4",
    )
    args = parser.parse_args()
    apply_profile(HQ_PROFILE if args.hq else SD_PROFILE)
    print(f"[profile] {PROFILE.name}: long side {MAX_DIM}px @ {FPS} fps → {OUT_PATH}")
    render_video()
