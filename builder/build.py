"""MapKoon v3 proof builder: Overture (OpenStreetMap + Microsoft buildings) -> ready-made 3D tiles.
Everything heavy happens here, once: roads are merged into clean shapes (junctions included), lifted on bridges with
smooth ramps, painted with lane lines and crosswalks from the data; buildings are extruded with roofs and colours.
The browser only loads the triangles and draws them.

Tile file (little endian):  'MK3T' u32 headerLen  header(JSON)  then the buffers listed in the header, back to back.
Each layer: positions f32 xyz (metres, local), colours u8 rgb, extra f32 (per vertex, layer dependent), indices u32."""
import json, math, os, sys, struct, hashlib, gzip, glob, argparse
import numpy as np
from shapely import wkb
from shapely.geometry import LineString, Polygon, MultiPolygon, box, Point
from shapely.ops import unary_union, substring
import mapbox_earcut as earcut

ap = argparse.ArgumentParser()
ap.add_argument('data'); ap.add_argument('out')
ap.add_argument('--tiles', help='tx0,tz0,tx1,tz1: the tiles this chunk owns (end exclusive)')
ap.add_argument('--bbox', help='W,S,E,N (old style, owns every tile)')
ap.add_argument('--pad', type=float, default=300, help='metres of extra data read around the owned tiles')
ap.add_argument('--heights', default='', help='folder of heights_*.json (City of Montréal laser heights)')
ap.add_argument('--raw', action='store_true', help='also write the old uncompressed tiles (MK3T) for the preview')
A = ap.parse_args()
D = A.data; OUT = A.out
LON0, LAT0 = -73.5625, 45.4575
KX = 111320 * math.cos(math.radians(LAT0)); KY = 110540
T = 500.0   # tile size, metres


def to_local(g):
    from shapely import transform
    return transform(g, lambda a: np.column_stack([(a[:, 0] - LON0) * KX, -(a[:, 1] - LAT0) * KY]))


AREA = None


def load(name):
    """features in local metres, cut to the area (a river polygon covers hundreds of km; only our piece is kept)"""
    R = json.load(open(os.path.join(D, name + '.json')))
    out = []
    for r in R:
        g = to_local(wkb.loads(bytes.fromhex(r['geometry'])))
        if AREA is not None and g.geom_type in ('Polygon', 'MultiPolygon') and not AREA.contains(g):
            g = g.intersection(AREA)
            if g.is_empty: continue
        r['g'] = g; out.append(r)
    return out


def jl(s):
    if s is None or s == 'None': return None
    if isinstance(s, (list, dict)): return s
    return json.loads(s.replace("'", '"').replace('None', 'null').replace('True', 'true').replace('False', 'false'))


def hexrgb(h):
    h = h.lstrip('#'); return [int(h[i:i + 2], 16) for i in (0, 2, 4)]


class Layer:
    def __init__(s): s.p = []; s.c = []; s.e = []; s.i = []; s.n = 0

    def add(s, pts, cols, tris, extra=None):
        s.i.extend([t + s.n for t in tris]); s.p.extend(pts); s.c.extend(cols)
        s.e.extend(extra if extra is not None else [0.0] * (len(pts) // 3)); s.n += len(pts) // 3


tiles = {}


def tile_of(x, z): return (math.floor(x / T), math.floor(z / T))


def layer(tk, name):
    t = tiles.setdefault(tk, {}); return t.setdefault(name, Layer())


def flat_poly(poly, y, col, lname, extra=0.0):
    """a flat polygon (with holes) at height y, split into tiles"""
    if poly.is_empty: return
    polys = list(poly.geoms) if hasattr(poly, 'geoms') else [poly]
    for P in polys:
        if P.geom_type != 'Polygon' or P.area < 0.5: continue
        x0, z0, x1, z1 = P.bounds
        for tx in range(math.floor(x0 / T), math.floor(x1 / T) + 1):
            for tz in range(math.floor(z0 / T), math.floor(z1 / T) + 1):
                Q = P.intersection(box(tx * T, tz * T, (tx + 1) * T, (tz + 1) * T))
                for q in (Q.geoms if hasattr(Q, 'geoms') else [Q]):
                    if q.geom_type != 'Polygon' or q.area < 0.3: continue
                    rings = [np.array(q.exterior.coords)[:-1]] + [np.array(r.coords)[:-1] for r in q.interiors]
                    v = np.vstack(rings).astype(np.float32); ends = np.cumsum([len(r) for r in rings]).astype(np.uint32)
                    try: tri = earcut.triangulate_float32(v, ends)
                    except Exception: continue
                    pts = []
                    for a in v: pts += [float(a[0]), y, float(a[1])]
                    layer((tx, tz), lname).add(pts, col * len(v), [int(t) for t in tri], [extra] * len(v))


# ---------------------------------------------------------------- ground
if A.tiles:
    TX0, TZ0, TX1, TZ1 = map(int, A.tiles.split(','))
    AREA = box(TX0 * T - A.pad, TZ0 * T - A.pad, TX1 * T + A.pad, TZ1 * T + A.pad)
    def owned(tk): return TX0 <= tk[0] < TX1 and TZ0 <= tk[1] < TZ1
else:
    BB = [float(v) for v in (A.bbox or '-73.59,45.44,-73.535,45.475').split(',')]
    AREA = box((BB[0] - LON0) * KX - 50, -(BB[3] - LAT0) * KY - 50, (BB[2] - LON0) * KX + 50, -(BB[1] - LAT0) * KY + 50)
    def owned(tk): return True
lu = load('land_use'); wa = load('water'); lc = load('land_cover'); inf = load('infrastructure')
GREEN = {'grass', 'park', 'pitch', 'playground', 'recreation_ground', 'garden', 'flowerbed', 'meadow', 'fairway', 'green',
         'tee', 'village_green', 'cemetery', 'golf_course', 'dog_park', 'allotments'}
for r in lu:
    g = r['g']
    if g.geom_type not in ('Polygon', 'MultiPolygon'): continue
    c = r.get('class')
    if c in GREEN: flat_poly(g, 0.02, [88, 170, 70] if c != 'pitch' else [70, 160, 75], 'ground')
    elif c == 'bunker': flat_poly(g, 0.03, [222, 205, 150], 'ground')
for r in lc:
    if r['subtype'] in ('forest', 'shrub'): flat_poly(r['g'], 0.015, [60, 130, 60], 'ground')
for r in wa:
    g = r['g']
    if g.geom_type in ('Polygon', 'MultiPolygon') and r['class'] != 'swimming_pool': flat_poly(g, 0.01, [40, 120, 190], 'water')
    elif r['class'] == 'swimming_pool' and g.geom_type == 'Polygon': flat_poly(g, 0.05, [90, 190, 230], 'water')
for r in inf:
    if r['class'] == 'parking' and r['g'].geom_type in ('Polygon', 'MultiPolygon'):
        flat_poly(r['g'], 0.04, [96, 98, 104], 'paved')

print('ground done', flush=True)
# ---------------------------------------------------------------- roads
seg = load('segment')
WID = {'motorway': 11.5, 'trunk': 10.5, 'primary': 14, 'secondary': 12, 'tertiary': 10, 'residential': 8.5,
       'unclassified': 7.5, 'living_street': 6, 'service': 5.5, 'unknown': 6}
SUBW = {'link': 5.5, 'driveway': 3.6, 'parking_aisle': 6.5, 'alley': 4.2}
roads = []
for r in seg:
    if r['subtype'] != 'road' or r['class'] not in WID or r['g'].geom_type != 'LineString': continue
    sub = r.get('subclass'); w = SUBW.get(sub, WID[r['class']])
    if sub == 'link' and r['class'] == 'motorway': w = 6
    acc = jl(r.get('access_restrictions')) or []
    ow = any(a.get('access_type') == 'denied' and (a.get('when') or {}).get('heading') == 'backward' and not (a.get('when') or {}).get('mode') for a in acc)
    lv = jl(r.get('level_rules')) or []; fl = jl(r.get('road_flags')) or []
    roads.append(dict(g=r['g'], cls=r['class'], sub=sub, w=w, ow=ow, lv=lv, fl=fl, id=r['id']))
print('roads', len(roads))

STEP = 4.0; RAMP = 90.0; LH = 6.5   # sampling (m), ramp length (m), height per level (m)


def profile(rd):
    """points every few metres along the road and the deck height at each (levels from the data, smoothed into ramps)"""
    g = rd['g']; L = g.length; n = max(2, int(L / STEP) + 1)
    ss = np.linspace(0, L, n); pts = np.array([g.interpolate(s).coords[0] for s in ss])
    lvl = np.zeros(n)
    for rule in rd['lv']:
        v = rule.get('value') or 0; bt = rule.get('between') or [0, 1]
        m = (ss >= bt[0] * L - 0.01) & (ss <= bt[1] * L + 0.01); lvl[m] = v
    return ss, pts, lvl


# heights at the road ends (connectors) are shared, so ramps on both sides of a junction meet: two passes
prof = [profile(rd) for rd in roads]
end_h = {}
for rd, (ss, pts, lvl) in zip(roads, prof):
    for k in (0, -1):
        key = (round(pts[k][0], 1), round(pts[k][1], 1)); end_h[key] = max(end_h.get(key, 0), lvl[k])
H = []
for rd, (ss, pts, lvl) in zip(roads, prof):
    L = ss[-1]; tgt = lvl * LH
    a = end_h.get((round(pts[0][0], 1), round(pts[0][1], 1)), 0) * LH; b = end_h.get((round(pts[-1][0], 1), round(pts[-1][1], 1)), 0) * LH
    # each sample: the highest level within reach, minus the slope down to it (a smooth ramp, never steeper than LH/RAMP)
    h = np.zeros(len(ss))
    src = list(zip(ss, tgt)) + [(0.0, a), (L, b)]
    for i, s in enumerate(ss):
        h[i] = max([t - abs(s - s2) / RAMP * LH for s2, t in src if t > 0] + [0])
    # a road whose whole length is high keeps its deck (no dip in the middle)
    H.append(np.clip(h, 0, None))


def offset_strip(pts, hh, half, y0, col, lname, side_col=None):
    """a 3D ribbon along pts, at heights hh, half width 'half'; with deck sides when lifted"""
    n = len(pts)
    if n < 2: return
    d = np.gradient(pts, axis=0); d /= (np.linalg.norm(d, axis=1)[:, None] + 1e-9); nx, nz = -d[:, 1], d[:, 0]
    L = pts + np.column_stack([nx, nz]) * half; R = pts - np.column_stack([nx, nz]) * half
    for k0 in range(0, n - 1, 60):   # chunks, each in the tile of its start
        k1 = min(n, k0 + 61); tk = tile_of(*pts[k0])
        P = []; C = []; I = []; E = []
        m = k1 - k0
        for k in range(k0, k1):
            y = hh[k] + y0
            P += [L[k][0], y, L[k][1], R[k][0], y, R[k][1]]
        for j in range(m - 1):
            a = 2 * j; I += [a, a + 1, a + 2, a + 1, a + 3, a + 2]
        layer(tk, lname).add(P, col * (2 * m), I)
        if side_col and max(hh[k0:k1]) > 0.4:
            P = []; I = []
            for k in range(k0, k1):
                y = hh[k] + y0; yb = max(0.0, y - 1.4)
                P += [L[k][0], y + 0.9, L[k][1], L[k][0], yb, L[k][1], R[k][0], y + 0.9, R[k][1], R[k][0], yb, R[k][1]]
            for j in range(m - 1):
                a = 4 * j
                I += [a, a + 1, a + 4, a + 1, a + 5, a + 4, a + 2, a + 6, a + 3, a + 3, a + 6, a + 7]
            layer(tk, 'deck').add(P, side_col * (4 * m), I)
        # (v3) piers under lifted decks, every ~32 m: a concrete column on a wider cap, like a real overpass or bridge
        if side_col:
            for k in range(k0, k1):
                if k % 8 or hh[k] < 3.5: continue
                cx, cz = pts[k]; y = hh[k] + y0 - 1.4
                tx, tz = (pts[min(n - 1, k + 1)] - pts[max(0, k - 1)]); l = math.hypot(tx, tz) or 1; tx, tz = tx / l, tz / l
                for (a, b, y0b, y1b) in ((0.9, half * 0.45, 0, y - 0.8), (1.3, half * 0.9, y - 0.8, y)):
                    # a box: a = half-length along the road, b = half-width across it
                    C = [(cx + tx * a + tz * b, cz + tz * a - tx * b), (cx - tx * a + tz * b, cz - tz * a - tx * b),
                         (cx - tx * a - tz * b, cz - tz * a + tx * b), (cx + tx * a - tz * b, cz + tz * a + tx * b)]
                    P = []
                    for (qx, qz) in C: P += [qx, y0b, qz]
                    for (qx, qz) in C: P += [qx, y1b, qz]
                    I = []
                    for j in range(4):
                        j2 = (j + 1) % 4; I += [j, j2, 4 + j2, j, 4 + j2, 4 + j]
                    layer(tk, 'deck').add(P, [132, 136, 142] * 8, I)


# ground-level road shapes are merged (clean junctions); lifted parts are ribbons
ASPH = [58, 61, 68]
flat_parts = {k: [] for k in ('road', 'drive')}
for rd, (ss, pts, lvl), h in zip(roads, prof, H):
    up = h > 0.15
    # ground pieces
    runs = []; cur = []
    for i in range(len(ss)):
        if not up[i]: cur.append(pts[i])
        else:
            if len(cur) > 1: runs.append(cur)
            cur = [pts[i]] if False else []
    if len(cur) > 1: runs.append(cur)
    for rpts in runs:
        flat_parts['drive' if rd['sub'] in ('driveway', 'parking_aisle', 'alley') else 'road'].append(
            LineString(rpts).buffer(rd['w'] / 2, cap_style='round', join_style='round', quad_segs=3))
    # lifted pieces (with a little overlap into the ground pieces)
    i = 0
    while i < len(ss):
        if up[i]:
            j = i
            while j < len(ss) and up[j]: j += 1
            a, b = max(0, i - 1), min(len(ss), j + 1)
            offset_strip(pts[a:b], h[a:b], rd['w'] / 2, 0.06, ASPH, 'road', side_col=[150, 154, 162])
            rd.setdefault('lifted', []).append((a, b))
            i = j
        else: i += 1
print('merging road shapes...', flush=True)
road_u = unary_union(flat_parts['road']); drive_u = unary_union(flat_parts['drive']).difference(road_u)
flat_poly(road_u, 0.06, ASPH, 'road'); flat_poly(drive_u, 0.055, [70, 73, 80], 'road')

# sidewalks from the data (footway=sidewalk), and a narrow curb strip around every city road
sw = [r['g'] for r in seg if r['class'] == 'footway' and r.get('subclass') == 'sidewalk' and r['g'].geom_type == 'LineString']
sw_u = unary_union([g.buffer(1.1, cap_style='flat') for g in sw]).difference(road_u)
flat_poly(sw_u, 0.12, [168, 168, 162], 'walk')

# ---------------------------------------------------------------- markings
WHITE = [235, 235, 230]; YEL = [235, 190, 60]
# junction zones: where 3+ road ends or vertices meet, no lane lines (like real paint)
from collections import Counter
cnt = Counter()
for rd in roads:
    if rd['sub'] in ('driveway', 'parking_aisle'): continue
    for c in rd['g'].coords: cnt[(round(c[0], 1), round(c[1], 1))] += 1
JZ = unary_union([Point(k).buffer(9) for k, v in cnt.items() if v >= 3]) if cnt else None


def dashes(pts, hh, off, dash, gap, half_w, col):
    """a painted line 'off' metres to the side of the centre, dashed (gap>0) or solid"""
    if len(pts) < 2: return
    d = np.gradient(pts, axis=0); d /= (np.linalg.norm(d, axis=1)[:, None] + 1e-9); nrm = np.column_stack([-d[:, 1], d[:, 0]])
    c = pts + nrm * off
    seg_len = np.r_[0, np.cumsum(np.linalg.norm(np.diff(c, axis=0), axis=1))]
    Ltot = seg_len[-1]; per = dash + gap if gap > 0 else Ltot + 1
    s = 0.0
    while s < Ltot - 0.3:
        e = min(Ltot, s + (dash if gap > 0 else Ltot))
        ks = np.linspace(s, e, max(2, int((e - s) / 3) + 2))
        P = np.column_stack([np.interp(ks, seg_len, c[:, 0]), np.interp(ks, seg_len, c[:, 1])])
        Y = np.interp(ks, seg_len, hh)
        mid = Point(P[len(P) // 2])
        if JZ is None or not JZ.contains(mid):
            offset_strip(P, Y, half_w, 0.09, col, 'paint')
        s += per


for rd, (ss, pts, lvl), h in zip(roads, prof, H):
    cls, sub, w = rd['cls'], rd['sub'], rd['w']
    if sub in ('driveway', 'parking_aisle', 'alley') or cls in ('service', 'living_street'): continue
    if rd['ow']:
        lanes = max(1, round(w / 3.6))
        for k in range(1, lanes): dashes(pts, h, -w / 2 + k * w / lanes, 3, 9, 0.08, WHITE)
        if cls in ('motorway', 'trunk') or sub == 'link':
            dashes(pts, h, w / 2 - 0.4, 1, 0, 0.08, WHITE); dashes(pts, h, -w / 2 + 0.4, 1, 0, 0.08, YEL)
    else:
        if cls in ('primary', 'secondary', 'tertiary'):
            dashes(pts, h, 0.12, 1, 0, 0.06, YEL); dashes(pts, h, -0.12, 1, 0, 0.06, YEL)
            if w >= 12:
                dashes(pts, h, w / 4, 3, 9, 0.07, WHITE); dashes(pts, h, -w / 4, 3, 9, 0.07, WHITE)
        elif cls in ('residential', 'unclassified') and w >= 8: dashes(pts, h, 0, 3, 6, 0.06, YEL)

# crosswalks from the data: white bars along the crossing
for r in seg:
    if r['class'] == 'footway' and r.get('subclass') == 'crosswalk' and r['g'].geom_type == 'LineString':
        g = r['g']
        if g.length < 3 or not g.intersects(road_u): continue
        n = int(g.length / 1.0)
        for k in range(n):
            if k % 2: continue
            p = np.array(g.interpolate(k + 0.5).coords[0]); q = np.array(g.interpolate(min(g.length, k + 0.6)).coords[0])
            d = q - p; d /= (np.linalg.norm(d) + 1e-9); nrm = np.array([-d[1], d[0]])
            if not road_u.contains(Point(p)): continue
            P = np.array([p - nrm * 1.6, p + nrm * 1.6])
            offset_strip(P, np.array([0.0, 0.0]), 0.25, 0.1, WHITE, 'paint')

print('markings done', flush=True)
# ---------------------------------------------------------------- buildings
bl = load('bld')
MH = {}
for f in glob.glob(os.path.join(A.heights, 'heights_*.json')) if A.heights else []:
    MH.update(json.load(open(f)))
if os.path.exists(os.path.join(D, 'mtl_heights.json')): MH.update(json.load(open(os.path.join(D, 'mtl_heights.json'))))
print('city heights', len(MH))
PAL = {'house': ['#a8553f', '#b9a58a', '#d8d2c4', '#8f4a3a', '#c9b79a'], 'residential': ['#9c4f3d', '#b5a184', '#cfc6b3', '#8a8f96', '#a65a44'],
       'apartments': ['#c8c4bb', '#a7aab0', '#9c4f3d', '#d6d0c2', '#8f949b'], 'commercial': ['#b8bcc2', '#9aa3ad', '#d3d0c8'],
       'outbuilding': ['#8d8f93', '#a3a29c']}
DEFH = {'house': 7.5, 'detached': 7.5, 'residential': 8.5, 'apartments': 13, 'commercial': 7, 'retail': 6, 'industrial': 8,
        'outbuilding': 3.2, 'garage': 3, 'shed': 2.6, 'school': 10, 'church': 14}
for r in bl:
    g = r['g']
    if g.geom_type not in ('Polygon', 'MultiPolygon') or r.get('is_underground'): continue
    cls = r.get('class') or ''
    # City of Montréal laser heights first (2015). A mapped height wins when it is clearly taller: towers built since 2015,
    # or a tower whose footprint also covers its lower podium.
    hc = MH.get(r['id']); ho = r.get('height')
    hgt = ho if (ho and hc and ho >= 25 and ho > 1.4 * hc) else (hc or ho or ((r.get('num_floors') or 0) * 3.1 + 1.2 if r.get('num_floors') else None))
    area = g.area
    if not hgt: hgt = DEFH.get(cls, 6.5 if area < 150 else 9 if area < 900 else 12)
    if area < 40: hgt = min(hgt, 3.5)
    mh = r.get('min_height') or 0
    hid = int(hashlib.md5(r['id'].encode()).hexdigest()[:8], 16)
    fc = r.get('facade_color'); pal = PAL.get(cls) or PAL['residential']
    if fc and (max(hexrgb(fc)) - min(hexrgb(fc)) > 120): fc = None   # (a pure red or green wall in the data is almost always a mapping slip)
    if hgt > 35: pal = ['#8e9aa6', '#a9b4bd', '#6f7c8a', '#c3c7c9', '#7d8b96', '#b5aea3']   # towers: glass and concrete, not brick
    col = hexrgb(fc) if fc else hexrgb(pal[hid % len(pal)])
    roof = [int(c * 0.62) for c in col] if not fc else [110, 110, 112]
    rs = r.get('roof_shape') or ''
    pitched = rs in ('gabled', 'hipped', 'saltbox') or (cls in ('house', 'detached') and area < 260 and not rs)
    for P in (g.geoms if g.geom_type == 'MultiPolygon' else [g]):
        P = P.simplify(0.3)
        if P.is_empty or P.area < 10: continue
        ring = np.array(P.exterior.coords)[:-1]
        if Polygon(ring).exterior.is_ccw: ring = ring[::-1]
        tk = tile_of(*ring[0]); Ly = layer(tk, 'bld')
        top = hgt - (2.2 if pitched else 0)
        # walls (extra = running length along the wall, for the window pattern; negative = no windows)
        pts = []; cols = []; tri = []; ex = []; run = 0.0
        win = 1.0 if (area >= 40 and cls not in ('outbuilding', 'garage', 'shed')) else -1.0
        for i in range(len(ring)):
            a, b = ring[i], ring[(i + 1) % len(ring)]; ln = float(np.hypot(*(b - a)))
            base = len(pts) // 3
            pts += [a[0], mh, a[1], b[0], mh, b[1], b[0], top, b[1], a[0], top, a[1]]
            cols += col * 4; ex += [win * (run + 0.001), win * (run + ln), win * (run + ln), win * (run + 0.001)]
            tri += [base, base + 1, base + 2, base, base + 2, base + 3]; run += ln
        Ly.add(pts, cols, tri, ex)
        # roof: flat (with the earcut of the footprint), or a simple ridge over the long axis for houses
        if pitched:
            rect = P.minimum_rotated_rectangle; rc = np.array(rect.exterior.coords)[:4]
            e0 = rc[1] - rc[0]; e1 = rc[2] - rc[1]
            if np.linalg.norm(e0) < np.linalg.norm(e1): rc = np.roll(rc, -1, axis=0); e0, e1 = e1, e0
            m0 = (rc[0] + rc[3]) / 2; m1 = (rc[1] + rc[2]) / 2
            sh = 0.22 * (m1 - m0)
            r0, r1 = m0 + sh, m1 - sh
            q = [rc[0], rc[1], rc[2], rc[3]]
            pts = []; tri = []
            for v in q: pts += [float(v[0]), top, float(v[1])]
            pts += [float(r0[0]), hgt, float(r0[1]), float(r1[0]), hgt, float(r1[1])]
            tri = [0, 1, 5, 0, 5, 4, 2, 3, 4, 2, 4, 5, 1, 2, 5, 3, 0, 4]
            Ly.add(pts, roof * 6, tri, [-1.0] * 6)
        else:
            v = ring.astype(np.float32)
            try: t = earcut.triangulate_float32(v, np.array([len(v)], dtype=np.uint32))
            except Exception: continue
            pts = []
            for a in v: pts += [float(a[0]), hgt, float(a[1])]
            Ly.add(pts, roof * len(v), [int(x) for x in t], [-1.0] * len(v))

# ---------------------------------------------------------------- street furniture: points (lamps, lights, stops)
props = {}
for r in inf:
    if r['g'].geom_type != 'Point': continue
    k = {'street_lamp': 'lamp', 'traffic_signals': 'light', 'stop': 'stop'}.get(r['class'])
    if not k: continue
    x, z = r['g'].coords[0]
    pt = Point(x, z)
    if road_u.contains(pt):
        # (lights and stop signs are often mapped on the road's centre line: stand them at the kerb instead)
        from shapely.ops import nearest_points
        q = nearest_points(road_u.boundary, pt)[0]; dx, dz = q.x - x, q.y - z; L = math.hypot(dx, dz) or 1
        if L > 14: continue
        x, z = q.x + dx / L * 1.2, q.y + dz / L * 1.2
    props.setdefault(tile_of(x, z), []).append([k, round(x, 1), round(z, 1)])

# ---------------------------------------------------------------- trees (OpenStreetMap: single trees, tree rows, woods)
import random
random.seed(7)
ntree = 0
try:
    landf = load('land')
except FileNotFoundError:
    landf = []
def put_tree(x, z, kind):
    global ntree
    pt = Point(x, z)
    if road_u.contains(pt): return
    props.setdefault(tile_of(x, z), []).append([kind, round(x, 1), round(z, 1), random.randint(0, 99)]); ntree += 1
for r in landf:
    g = r['g']; c = r.get('class')
    if c == 'tree' and g.geom_type == 'Point':
        put_tree(g.x, g.y, 'tree')
    elif c == 'tree_row' and g.geom_type == 'LineString':
        for k in range(0, int(g.length), 8):
            q = g.interpolate(k); put_tree(q.x, q.y, 'tree')
    elif c in ('wood', 'forest') and g.geom_type in ('Polygon', 'MultiPolygon'):
        x0, z0, x1, z1 = g.bounds; n = 0
        for _ in range(int(min(500, g.area / 45))):
            x, z = random.uniform(x0, x1), random.uniform(z0, z1)
            if g.contains(Point(x, z)): put_tree(x, z, 'pine' if random.random() < 0.35 else 'tree'); n += 1
print('trees', ntree)

# ---------------------------------------------------------------- write
# Compact tile ('MK3Q', gzip): positions as int16 in 2 cm steps around the tile centre, colours u8, extra f32 only when used,
# indices u16 when they fit. The browser unzips with DecompressionStream and scales positions on the GPU.
Q = 0.02
os.makedirs(os.path.join(OUT, 't'), exist_ok=True)
if A.raw: os.makedirs(os.path.join(OUT, 'raw'), exist_ok=True)


def pad4(b): return b + b'\0' * ((4 - len(b) % 4) % 4)


def pack(hdr, bufs, magic):
    hj = json.dumps(hdr, separators=(',', ':')).encode(); hj += b' ' * ((4 - len(hj) % 4) % 4)
    return magic + struct.pack('<I', len(hj)) + hj + b''.join(pad4(b) for b in bufs)


written = []; tot = 0; totraw = 0
for tk, L in tiles.items():
    if not owned(tk): continue
    ox, oz = (tk[0] + 0.5) * T, (tk[1] + 0.5) * T
    hdr = {'v': 2, 'o': [ox, oz], 'q': Q, 'layers': [], 'props': props.get(tk, [])}; bufs = []
    rhdr = {'layers': [], 'props': props.get(tk, [])}; rbufs = []
    for name, ly in L.items():
        if not ly.i: continue
        P = np.array(ly.p, dtype=np.float64).reshape(-1, 3); C = np.array(ly.c, dtype=np.uint8); E = np.array(ly.e, dtype=np.float32); I = np.array(ly.i, dtype=np.uint32)
        P[:, 0] -= ox; P[:, 2] -= oz
        Pq = np.clip(np.round(P / Q), -32767, 32767).astype(np.int16)
        ue = bool(np.any(E != 0)); i16 = ly.n < 65536
        hdr['layers'].append({'name': name, 'nv': int(ly.n), 'ni': int(len(I)), 'e': int(ue), 'i16': int(i16)})
        bufs += [Pq.tobytes(), C.tobytes()] + ([E.tobytes()] if ue else []) + [I.astype(np.uint16 if i16 else np.uint32).tobytes()]
        if A.raw:
            P[:, 0] += ox; P[:, 2] += oz
            rhdr['layers'].append({'name': name, 'nv': int(ly.n), 'ni': int(len(I))})
            rbufs += [P.astype(np.float32).tobytes(), C.tobytes(), E.tobytes(), I.tobytes()]
    blob = gzip.compress(pack(hdr, bufs, b'MK3Q'), 9, mtime=0)
    fn = '%d_%d.bin.gz' % tk; open(os.path.join(OUT, 't', fn), 'wb').write(blob); tot += len(blob)
    if A.raw:
        rb = pack(rhdr, rbufs, b'MK3T'); open(os.path.join(OUT, 'raw', '%d_%d.wasm' % tk), 'wb').write(rb); totraw += len(rb)
    written.append([tk[0], tk[1], len(blob)])
os.makedirs(os.path.join(OUT, 'parts'), exist_ok=True)
json.dump({'tiles': written}, open(os.path.join(OUT, 'parts', ('%d_%d.json' % (TX0, TZ0)) if A.tiles else 'all.json'), 'w'))
print('tiles', len(written), 'MB', round(tot / 1e6, 2), ('raw MB %.1f' % (totraw / 1e6)) if A.raw else '')

# ---------------------------------------------------------------- road network for the car (lane matching), one file per tile
def lanes_dir(rd):
    """lanes in each direction, the same rule the paint uses"""
    w = rd['w']
    if rd['ow']: return max(1, round(w / 3.6))
    if rd['cls'] in ('primary', 'secondary', 'tertiary') and w >= 12: return 2
    return 1


os.makedirs(os.path.join(OUT, 'n'), exist_ok=True)
NT = {}; NET = []
for k, (rd, (ss, pts, lvl), h) in enumerate(zip(roads, prof, H)):
    rec = {'g': rd['id'], 'c': rd['cls'], 's': rd['sub'] or '', 'w': rd['w'], 'ow': 1 if rd['ow'] else 0, 'ln': lanes_dir(rd),
           'p': [round(float(v), 2) for xy in pts for v in xy], 'h': [round(float(v), 2) for v in h]}
    # a road goes in every owned tile it passes through, so the car always finds the roads around it
    tks = {tile_of(x, z) for x, z in pts}
    mine = [t for t in tks if owned(t)]
    for t in mine: NT.setdefault(t, []).append(rec)
    if mine or not A.tiles: NET.append(dict(rec, id=len(NET)))
nb = 0
for t, rs in NT.items():
    b = gzip.compress(json.dumps({'v': 2, 'roads': rs}, separators=(',', ':')).encode(), 9, mtime=0)
    open(os.path.join(OUT, 'n', '%d_%d.json.gz' % t), 'wb').write(b); nb += len(b)
if A.raw: json.dump({'v': 1, 'roads': NET}, open(os.path.join(OUT, 'raw', 'net.json'), 'w'), separators=(',', ':'))
print('net tiles', len(NT), 'MB', round(nb / 1e6, 2))
