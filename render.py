"""
A* pathfinding visualizer → cinematic MP4.

Geocodes any origin/destination pair, auto-frames the bounding box around the route,
and renders multi-radius bloom, age-faded trails, white-hot frontier, and a final
glowing path reveal.

Usage:
    python render.py "10 Post Office Square, Boston, MA" "2 Leighton St, Cambridge, MA"
    python render.py "<origin>" "<dest>" --hq           # 1440p / 120 fps
    python render.py "<origin>" "<dest>" --out my.mp4   # override output path
"""
from __future__ import annotations

import argparse
import hashlib
import math
import os
import pickle
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from heapq import heappush, heappop
from pathlib import Path
from typing import NamedTuple

import cairo
import cv2
import numpy as np
import osmnx as ox
import networkx as nx

# ───────────────────────────── config ─────────────────────────────
# Edit these for the no-args path (`python render.py`). CLI positional args override.
# Set both to "" to require explicit CLI args every run.
DEFAULT_ORIGIN = "10 Post Office Square, Boston, MA 02109"
DEFAULT_DEST   = "2 Leighton Street, Cambridge, MA 02141"

# OSM Overpass API politeness. As of 2026-04 overpass-api.de bans requests using generic
# User-Agent strings (osmnx's default trips this) — symptom is HTTP 406 Not Acceptable.
# Set a unique identifier with contact info per OSM's usage policy:
#   https://operations.osmfoundation.org/policies/api/
# Edit if you want your name/email here so the OSM admins can reach you if needed.
OVERPASS_USER_AGENT = "astar-cinematic-render/1.0 (personal viz; contact via github)"
OVERPASS_REFERER    = "https://github.com/"

# Resolved at startup from defaults or CLI args.
ORIGIN_ADDR: str = ""
DEST_ADDR:   str = ""

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
    out_path=Path(""),             # filled in at runtime from origin/dest slugs
    crf=16,
    preset="medium",
    hot_cool_per_sec=0.88 ** 30,   # preserves the existing look at 30 fps
)

HQ_PROFILE = RenderProfile(
    name="hq",
    max_dim=2560,                  # 1440p on the long side
    fps=120,
    out_path=Path(""),
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
    # `v3` bump: now SCC-pruned at fetch time; older caches contain the un-pruned graph
    # which breaks A* on motorway-only continental routes (origin snaps into a 5-node
    # dead-end ramp pocket).
    key = hashlib.sha1(f"{origin}→{dest}|simplify={SIMPLIFY_GRAPH}|v3".encode()).hexdigest()[:12]
    return Path(f".graph_cache_{key}.pkl")

CACHE_PATH: Path = Path()  # set in main()


def _slug(addr: str, max_len: int = 20) -> str:
    """Filename-friendly slug from an address. Tries street+city first; if that's too long,
    falls back to just the city (last comma-delimited segment we kept)."""
    parts = [p.strip() for p in addr.split(",")][:2]  # street, city — drop state/zip/country
    cleaned = " ".join(parts).lower()
    cleaned = re.sub(r"\b(suite|ste|apt|unit|fl|floor|rm|room)\s*\w+", "", cleaned)
    cleaned = re.sub(r"#\s*\w+", "", cleaned)
    cleaned = re.sub(r"[^a-z0-9]+", "_", cleaned).strip("_")
    if cleaned and len(cleaned) <= max_len:
        return cleaned
    city = parts[-1] if len(parts) > 1 else parts[0]
    city = re.sub(r"[^a-z0-9]+", "_", city.lower()).strip("_")
    return (city[:max_len].rstrip("_") or "loc")


def _default_out_path(origin: str, dest: str, hq: bool) -> Path:
    """Build a readable output filename from the two addresses, e.g. boston_to_cambridge.mp4."""
    suffix = "_hq" if hq else ""
    return Path(f"{_slug(origin)}_to_{_slug(dest)}_astar{suffix}.mp4")


def geocode_or_die(addr: str, label: str) -> tuple[float, float]:
    """Geocode `addr` or exit immediately with a useful message. Run upfront on every
    invocation so a typo'd address fails in ~1 second instead of after graph load."""
    try:
        lat, lon = ox.geocode(addr)
    except Exception as e:
        print(f"\n[geocode error] could not resolve {label} address:", file=sys.stderr)
        print(f"  {addr!r}", file=sys.stderr)
        print(f"  → {type(e).__name__}: {e}", file=sys.stderr)
        print(
            "\nTry dropping suite/unit numbers or country, or simplify to "
            "'street, city, state'. Nominatim is picky.\n",
            file=sys.stderr,
        )
        sys.exit(1)
    print(f"[geocode] {label}: {addr!r} → ({lat:.5f}, {lon:.5f})")
    return lat, lon

# bbox auto-framing.
#
# Calibrated against the original Chicago wide HQ render. That route had span 0.29° lat × 0.07°
# lon and was hand-padded to (+0.04°, +0.16°) — i.e. ~14% of the long axis on each side, with the
# short axis blown out to bring the framed aspect to ~1.3:1. The same two parameters reproduce
# Chicago exactly and give a consistent feel across any route size:
#   - PAD_FRAC: percent of route span padded on each side of the dominant axis
#   - TARGET_MAX_ASPECT: cap on the framed bbox aspect ratio (long/short); short axis expands
#     to fill the rest, which is what gives narrow N-S routes their generous metro context.
# This is also how percent-padding stays meaningful at any scale: NYC→LA gets a continental
# letterbox at the same aspect, not 600 km of empty Mexico/Canada padded as a fixed degree count.
PAD_FRAC          = 0.14
TARGET_MAX_ASPECT = 1.3
PAD_MIN_KM        = 1.5    # absolute floor on each side — keeps cross-the-street trips legible

M_PER_DEG_LAT = 110_540.0   # WGS84 mean — good enough for visualization framing


# Adaptive road detail.
#
# At 1920px the screen has a fixed pixel budget regardless of bbox size, so a continental
# render that fetches every alley would cost millions of edges and look like static. Tiers
# drop progressively-less-important OSM road classes as bbox area grows, keeping edge density
# per pixel roughly constant — and rendering time bounded — across 7 orders of magnitude.
#
# Width scales up as we trim, so motorway-only continental renders aren't a hairline. The
# tier_seam below the threshold uses tier N's filter; one km² past it uses tier N+1's filter
# (and slightly fatter strokes). Visually the seam is invisible because no single render
# spans two tiers — only the population of renders does.
#
# overpass_chunk_km² controls how aggressively osmnx slices the bbox into sub-queries against
# the public Overpass API. Default is 2500 km² which is correct for "drive (full)" (dense
# data → conservative chunks), but devastating for sparse tiers — a continental motorway-only
# fetch at the default cap would be ~7,700 sequential HTTP calls instead of ~10. Cap scales
# inversely with the tier's expected result density, so each chunk returns a similar amount
# of data regardless of how zoomed-out we are.
def _filter_for(*levels: str) -> str:
    """Build an Overpass `["highway"~"a|b|..."]` filter, auto-including `_link` variants
    for the major hierarchy levels (so e.g. dropping primary also drops primary_link)."""
    parts: list[str] = []
    for lvl in levels:
        parts.append(lvl)
        if lvl in ("motorway", "trunk", "primary", "secondary", "tertiary"):
            parts.append(f"{lvl}_link")
    return f'["highway"~"{"|".join(parts)}"]'


class RoadTier(NamedTuple):
    max_area_km2: float          # tier selected if framed bbox area ≤ this
    osmnx_kwargs: dict           # passed to ox.graph_from_bbox; either network_type or custom_filter
    label: str                   # human-readable tier name (printed at startup)
    width_scale: float           # multiplier on ROAD_WIDTH/PATH_WIDTH so coarser maps stay legible
    overpass_chunk_km2: float    # max km² per Overpass sub-query (osmnx max_query_area_size)
    overpass_timeout_s: int      # per-request HTTP timeout (osmnx requests_timeout)


ROAD_TIERS: list[RoadTier] = [
    RoadTier(    2_500, {"network_type": "drive"},                                                                "drive (full)", 1.00,     2_500, 180),
    RoadTier(   10_000, {"custom_filter": _filter_for("motorway", "trunk", "primary", "secondary", "tertiary",
                                                      "unclassified", "residential")},                           "no service",   1.05,     5_000, 240),
    RoadTier(   40_000, {"custom_filter": _filter_for("motorway", "trunk", "primary", "secondary", "tertiary")}, "tertiary+",    1.20,    15_000, 240),
    RoadTier(  150_000, {"custom_filter": _filter_for("motorway", "trunk", "primary", "secondary")},             "secondary+",   1.40,    50_000, 300),
    RoadTier(  600_000, {"custom_filter": _filter_for("motorway", "trunk", "primary")},                          "primary+",     1.70,   200_000, 300),
    RoadTier(3_000_000, {"custom_filter": _filter_for("motorway", "trunk")},                                     "trunk+",       2.20,   600_000, 300),
    RoadTier(math.inf,  {"custom_filter": _filter_for("motorway")},                                              "interstates",  2.90, 2_000_000, 300),
]


def select_tier(area_km2: float) -> RoadTier:
    for tier in ROAD_TIERS:
        if area_km2 <= tier.max_area_km2:
            return tier
    raise RuntimeError("unreachable")


def _patch_osmnx_observability(expected_chunks: int) -> None:
    """Wrap multiple osmnx internals so the silent post-fetch phase prints status.

    A bare osmnx fetch on a continental route looks like:
        - 12-15 chunks download in seconds (silent for fast queries / cache hits)
        - parses 5M+ JSON elements (1-2 min, silent)
        - builds NetworkX graph (1-2 min, silent)
        - truncates to bbox polygon (~30s, silent)
        - finds largest connected component (1-3 min, silent — the worst offender)
    Total can be 5-10+ minutes of dead air after the chunk lines stop. This wraps each
    phase so it prints a status line. The chunk estimate is rough — osmnx subdivides on
    a projected-meters grid, and continental UTM distortion makes the actual count run
    a few past our km-based estimate (hence the leading ~)."""
    from osmnx import _overpass
    from osmnx import graph as _g
    from osmnx import truncate as _t

    # 1. per-chunk Overpass downloads
    original_req = _overpass._overpass_request
    state = {"i": 0, "t0": time.time()}

    def wrapped_req(data):
        state["i"] += 1
        i = state["i"]
        chunk_t0 = time.time()
        print(f"[osm]   chunk {i:>2}/~{expected_chunks}: POST…", flush=True)
        try:
            result = original_req(data)
        except Exception as exc:
            print(f"[osm]   chunk {i:>2}/~{expected_chunks}: failed after {time.time()-chunk_t0:.0f}s ({type(exc).__name__})", flush=True)
            raise
        dt      = time.time() - chunk_t0
        elapsed = time.time() - state["t0"]
        n_elem  = len(result.get("elements", [])) if isinstance(result, dict) else 0
        print(f"[osm]   chunk {i:>2}/~{expected_chunks}: ok in {dt:.0f}s, {n_elem:,} elements  (total {elapsed:.0f}s)", flush=True)
        return result

    _overpass._overpass_request = wrapped_req

    # 2. graph build (parse JSON → NetworkX nodes/edges). Slowest single step for
    #    continental routes — millions of element-dict lookups.
    original_build = _g._create_graph

    # _create_graph internally consumes the chunk generator (which triggers each POST), so
    # any "starting" message would print before the chunks — only print after it returns.
    def wrapped_build(*args, **kwargs):
        t0 = time.time()
        result = original_build(*args, **kwargs)
        print(f"[osm] graph built: {len(result.nodes):,} nodes, {len(result.edges):,} edges in {time.time()-t0:.1f}s", flush=True)
        return result

    _g._create_graph = wrapped_build

    # 3. polygon truncation + largest-component filter. `largest_component` is the
    #    worst offender on continental graphs — does a weakly-connected-components scan
    #    over hundreds of thousands of nodes.
    original_trunc = _t.truncate_graph_polygon

    def wrapped_trunc(G, polygon, *args, **kwargs):
        before = len(G.nodes)
        t0 = time.time()
        result = original_trunc(G, polygon, *args, **kwargs)
        print(f"[osm] truncated to bbox: {len(result.nodes):,} of {before:,} nodes in {time.time()-t0:.1f}s", flush=True)
        return result

    _t.truncate_graph_polygon = wrapped_trunc

    original_largest = _t.largest_component

    def wrapped_largest(G, *args, **kwargs):
        before = len(G.nodes)
        t0 = time.time()
        result = original_largest(G, *args, **kwargs)
        kind = "strongly" if kwargs.get("strongly") else "weakly"
        print(f"[osm] largest {kind}-connected component: kept {len(result.nodes):,} of {before:,} nodes in {time.time()-t0:.1f}s", flush=True)
        return result

    _t.largest_component = wrapped_largest


def compute_bbox(o_lat: float, o_lon: float, d_lat: float, d_lon: float):
    """Frame the route in a Chicago-wide-HQ-equivalent bbox. Returns
    (north, south, east, west, framed_w_km, framed_h_km)."""
    mid_lat = (o_lat + d_lat) / 2
    m_per_deg_lon = M_PER_DEG_LAT * math.cos(math.radians(mid_lat))

    span_x_km = abs(o_lon - d_lon) * m_per_deg_lon / 1000.0
    span_y_km = abs(o_lat - d_lat) * M_PER_DEG_LAT / 1000.0

    W = max(span_x_km * (1 + 2 * PAD_FRAC), span_x_km + 2 * PAD_MIN_KM)
    H = max(span_y_km * (1 + 2 * PAD_FRAC), span_y_km + 2 * PAD_MIN_KM)

    if W > H * TARGET_MAX_ASPECT:
        H = W / TARGET_MAX_ASPECT
    elif H > W * TARGET_MAX_ASPECT:
        W = H / TARGET_MAX_ASPECT

    pad_x_km = (W - span_x_km) / 2
    pad_y_km = (H - span_y_km) / 2
    pad_lon  = pad_x_km * 1000.0 / m_per_deg_lon
    pad_lat  = pad_y_km * 1000.0 / M_PER_DEG_LAT

    north = max(o_lat, d_lat) + pad_lat
    south = min(o_lat, d_lat) - pad_lat
    east  = max(o_lon, d_lon) + pad_lon
    west  = min(o_lon, d_lon) - pad_lon
    return north, south, east, west, W, H

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


def apply_tier_scale(scale: float) -> None:
    """Inflate stroke widths to match the road tier we're rendering at — a continental
    motorway-only graph at 1920px needs ~3× the stroke a downtown-grid render does, or it
    disappears against the bloom. Called after load_graph (which picks the tier)."""
    global ROAD_WIDTH, PATH_WIDTH
    ROAD_WIDTH *= scale
    PATH_WIDTH *= scale


# ───────────────────────────── graph + A* ─────────────────────────────
def load_graph(o_lat: float, o_lon: float, d_lat: float, d_lon: float):
    if CACHE_PATH.exists():
        print(f"[graph] loading cache {CACHE_PATH}")
        with open(CACHE_PATH, "rb") as f:
            return pickle.load(f)

    north, south, east, west, frame_w_km, frame_h_km = compute_bbox(o_lat, o_lon, d_lat, d_lon)
    area_km2 = frame_w_km * frame_h_km
    tier = select_tier(area_km2)
    aspect = max(frame_w_km, frame_h_km) / min(frame_w_km, frame_h_km)
    print(f"[frame] {frame_w_km:.1f} × {frame_h_km:.1f} km  (aspect {aspect:.2f}, area {area_km2:,.0f} km²)")
    print(f"[frame] bbox  N{north:.4f}  S{south:.4f}  E{east:.4f}  W{west:.4f}")
    print(f"[tier]  {tier.label}  →  road/path width × {tier.width_scale:.2f}")

    # Tell osmnx to use bigger Overpass chunks for sparse tiers — see ROAD_TIERS comment.
    # Without this a continental motorway-only fetch chunks into ~7,700 sequential requests;
    # with it, ~12. We also bump the per-request timeout because bigger chunks mean Overpass
    # spends more CPU per query (still well under its server-side limits for sparse data).
    ox.settings.max_query_area_size = tier.overpass_chunk_km2 * 1e6
    ox.settings.requests_timeout    = tier.overpass_timeout_s
    # osmnx subdivides on a square grid with side = sqrt(chunk cap), not by area, so the real
    # chunk count is ceil(W/side) * ceil(H/side) — slightly more than naive area÷cap.
    quad_km = math.sqrt(tier.overpass_chunk_km2)
    n_chunks = max(1, math.ceil(frame_w_km / quad_km)) * max(1, math.ceil(frame_h_km / quad_km))
    print(f"[osm]   chunk cap {tier.overpass_chunk_km2:,} km²  →  ~{n_chunks} Overpass sub-quer{'y' if n_chunks == 1 else 'ies'}  (timeout {tier.overpass_timeout_s}s)")

    _patch_osmnx_observability(n_chunks)
    t0 = time.time()
    G = ox.graph_from_bbox(
        bbox=(west, south, east, north),
        simplify=SIMPLIFY_GRAPH,
        retain_all=False,
        **tier.osmnx_kwargs,
    )
    print(f"[osm] graph_from_bbox total: {time.time()-t0:.1f}s  →  {len(G.nodes):,} nodes, {len(G.edges):,} edges")

    # osmnx's built-in largest_component is WEAKLY connected only — it treats the graph as
    # undirected. Fine for dense `drive` networks (residential streets form return paths),
    # but disastrous for motorway-only continental fetches: dual-carriageway interstates
    # have eastbound and westbound as separate ways/nodes, and the only links between them
    # are interchange ramps. Many one-way ramps and truncated boundary segments end up as
    # tiny isolated SCCs (~1.6M of them on LA→NYC). nearest_nodes happily snaps origin into
    # a 5-node dead-end pocket and A* fails with "no path found". Pruning to the largest
    # STRONGLY-connected component guarantees a directed route exists between any pair of
    # remaining nodes. Cheap on small graphs (≈no-op for `drive`), ~25s on continental
    # motorway. We drop nodes that weren't useful for through-routing anyway.
    G = ox.truncate.largest_component(G, strongly=True)

    # UTM is accurate inside one ~600 km zone but distorts badly across continental spans;
    # switch to Web Mercator once the bbox is wider than that. Web Mercator stretches the
    # poles but for sub-Arctic visualization it just looks like the familiar US/world map.
    lon_span_deg = east - west
    to_crs = "EPSG:3857" if lon_span_deg > 4.0 else None
    t0 = time.time()
    print(f"[osm] projecting to {'Web Mercator' if to_crs else 'UTM'}…", flush=True)
    G = ox.project_graph(G, to_crs=to_crs)
    print(f"[osm] projected in {time.time()-t0:.1f}s", flush=True)

    t0 = time.time()
    print("[osm] annotating edges with speed + travel_time…", flush=True)
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

    G = ox.add_edge_travel_times(G)
    print(f"[osm] annotated in {time.time()-t0:.1f}s ({n_mw:,} motorway, {n_link:,} motorway_link overrides)", flush=True)

    t0 = time.time()
    print("[osm] snapping origin/dest to nearest graph nodes…", flush=True)
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
    print(f"[osm] origin → node {origin_node}, dest → node {dest_node}  ({time.time()-t0:.1f}s)", flush=True)

    payload = (G, origin_node, dest_node, tier.width_scale)
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
def render_video(o_lat: float, o_lon: float, d_lat: float, d_lon: float):
    global WIDTH, HEIGHT
    G, origin_node, dest_node, width_scale = load_graph(o_lat, o_lon, d_lat, d_lon)
    apply_tier_scale(width_scale)
    WIDTH, HEIGHT, proj = build_projector(G)
    print(f"[render] canvas: {WIDTH}×{HEIGHT}  road/path widths: {ROAD_WIDTH:.2f}/{PATH_WIDTH:.2f}px")

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


def main():
    global ORIGIN_ADDR, DEST_ADDR, CACHE_PATH, OUT_PATH

    parser = argparse.ArgumentParser(
        description="A* pathfinding visualizer → cinematic MP4",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python render.py '10 Post Office Square, Boston, MA' '2 Leighton St, Cambridge, MA'\n"
            "  python render.py 'Logan Airport, Boston, MA' 'Fenway Park, Boston, MA' --hq\n"
        ),
    )
    parser.add_argument("origin", nargs="?", default=DEFAULT_ORIGIN,
                        help=f"origin address (default from DEFAULT_ORIGIN: {DEFAULT_ORIGIN!r})")
    parser.add_argument("dest", nargs="?", default=DEFAULT_DEST,
                        help=f"destination address (default from DEFAULT_DEST: {DEFAULT_DEST!r})")
    parser.add_argument("--hq", action="store_true",
                        help="render the 1440p-bounded / 120 fps HQ version (slower, much sharper)")
    parser.add_argument("--out", type=Path, default=None,
                        help="override output filename (default: <origin>_to_<dest>_astar[_hq].mp4)")
    args = parser.parse_args()

    if not args.origin or not args.dest:
        parser.error(
            "missing addresses — pass them on the CLI, or set DEFAULT_ORIGIN/DEFAULT_DEST at the top of render.py"
        )

    # Identify ourselves to OSM/Overpass before any HTTP traffic — see comment on the
    # OVERPASS_USER_AGENT constant. Without this, overpass-api.de's abuse rules return
    # HTTP 406 for sparse-tier (continental) queries.
    ox.settings.http_user_agent = OVERPASS_USER_AGENT
    ox.settings.http_referer    = OVERPASS_REFERER

    ORIGIN_ADDR = args.origin
    DEST_ADDR   = args.dest

    # validate both addresses upfront — surface geocoding errors before any expensive work
    print("[geocode] validating addresses…", flush=True)
    o_lat, o_lon = geocode_or_die(args.origin, "origin")
    d_lat, d_lon = geocode_or_die(args.dest,   "dest")

    apply_profile(HQ_PROFILE if args.hq else SD_PROFILE)
    OUT_PATH = args.out if args.out is not None else _default_out_path(args.origin, args.dest, hq=args.hq)
    CACHE_PATH = _cache_path_for(args.origin, args.dest)

    print(f"[profile] {PROFILE.name}: long side {MAX_DIM}px @ {FPS} fps → {OUT_PATH}")
    print(f"[cache]   {CACHE_PATH}")

    render_video(o_lat, o_lon, d_lat, d_lon)


if __name__ == "__main__":
    main()
