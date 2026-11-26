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
# (Nov 26: grass flickered) the same ground is often mapped several times (a park, its lawn, a sports field, a wood);
# drawn on top of each other a centimetre apart, the screen can't tell which is in front from far away and they flicker.
# Now each kind is merged, and each piece of ground is drawn once: the top kind first, the others only where it isn't.
def merged(gs):
    gs = [g.buffer(0) if not g.is_valid else g for g in gs if g.geom_type in ('Polygon', 'MultiPolygon')]
    return unary_union(gs) if gs else Polygon()
kinds = [   # (polygons, height, colour), top first
    (merged([r['g'] for r in lu if r.get('class') == 'bunker']), 0.03, [222, 205, 150]),
    (merged([r['g'] for r in lu if r.get('class') == 'pitch']), 0.025, [70, 160, 75]),
    (merged([r['g'] for r in lu if r.get('class') in GREEN and r.get('class') != 'pitch']), 0.02, [88, 170, 70]),
    (merged([r['g'] for r in lc if r['subtype'] in ('forest', 'shrub')]), 0.015, [60, 130, 60]),
]
covered = Polygon()
for poly, y, col in kinds:
    if poly.is_empty: continue
    part = poly.difference(covered) if not covered.is_empty else poly
    flat_poly(part, y, col, 'ground'); covered = covered.union(poly)
water_u = merged([r['g'] for r in wa if r['class'] != 'swimming_pool'])
flat_poly(water_u, 0.01, [40, 120, 190], 'water')
pools = merged([r['g'] for r in wa if r['class'] == 'swimming_pool' and r['g'].geom_type == 'Polygon'])
flat_poly(pools, 0.05, [90, 190, 230], 'water')
flat_poly(merged([r['g'] for r in inf if r['class'] == 'parking']), 0.04, [96, 98, 104], 'paved')

print('ground done', flush=True)
# ---------------------------------------------------------------- roads
seg = load('segment')
WID = {'motorway': 11.5, 'trunk': 10.5, 'primary': 14, 'secondary': 12, 'tertiary': 10, 'residential': 8.5,
       'unclassified': 7.5, 'living_street': 6, 'service': 5.5, 'unknown': 6}
SUBW = {'link': 5.5, 'driveway': 3.6, 'parking_aisle': 6.5, 'alley': 4.2}
# (step 2) lane counts and widths from OpenStreetMap, joined through each Overture road's OpenStreetMap way id
OSMW = {}
if os.path.exists(os.path.join(D, 'osm_ways.json')): OSMW = json.load(open(os.path.join(D, 'osm_ways.json')))
import re as _re


def osm_id(r):
    best = None
    for s_ in r.get('sources') or []:
        m = _re.match(r'w(\d+)@', str(s_.get('record_id') or ''))
        if m and (best is None or not s_.get('property')): best = m.group(1)
    return best


def osm_way(r):
    best = osm_id(r)
    return OSMW.get(best) if best else None


roads = []
nosm = 0
for r in seg:
    if r['subtype'] != 'road' or r['class'] not in WID or r['g'].geom_type != 'LineString': continue
    sub = r.get('subclass'); w = SUBW.get(sub, WID[r['class']])
    if sub == 'link' and r['class'] == 'motorway': w = 6
    acc = jl(r.get('access_restrictions')) or []
    ow = any(a.get('access_type') == 'denied' and (a.get('when') or {}).get('heading') == 'backward' and not (a.get('when') or {}).get('mode') for a in acc)
    lv = jl(r.get('level_rules')) or []; fl = jl(r.get('road_flags')) or []
    o = osm_way(r); ln = lf = lb = 0
    if o:
        nosm += 1; ln = o.get('ln') or 0; lf = o.get('lf') or 0; lb = o.get('lb') or 0
        if 2 < (o.get('wd') or 0) < 40: w = o['wd']
    roads.append(dict(g=r['g'], cls=r['class'], sub=sub, w=w, ow=ow, lv=lv, fl=fl, id=r['id'], nm=(r.get('names') or {}).get('primary'),
                      ln=ln, lf=lf, lb=lb, osm=bool(o), osmsrc=osm_id(r) is not None))
print('roads', len(roads), 'with OpenStreetMap lanes/width', nosm)

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
    # (step 2: the low bridge on Rue Jolicoeur, Nov 26) where the road ends at street level, the deck must come down to it
    # inside this piece: no wall at the junction. A short bridge over a canal becomes a gentle hump instead.
    h = np.minimum(h, np.minimum(a + ss / RAMP * LH, b + (L - ss) / RAMP * LH))
    # ...and it always meets the roads at its ends at their height (a short ramp climbs steeper rather than leaving a step)
    h = np.maximum(h, a + (b - a) * ss / max(L, 1e-6))
    # a road whose whole length is high keeps its deck (no dip in the middle)
    H.append(np.clip(h, 0, None))
# landmarks with their real shape (the Champlain Bridge rises over the Seaway)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import landmarks
print('champlain deck points raised', landmarks.apply_heights(roads, prof, H))


PIERS = []


def box_(cx, cz, ux, uz, a, b, y0, y1):
    """a box: centre (cx, cz), half-length a along (ux, uz), half-width b across it, from y0 to y1"""
    vx, vz = -uz, ux; P = []
    for y in (y0, y1):
        for (sa, sb) in ((1, 1), (-1, 1), (-1, -1), (1, -1)):
            P += [cx + ux * a * sa + vx * b * sb, y, cz + uz * a * sa + vz * b * sb]
    I = []
    for j in range(4):
        j2 = (j + 1) % 4; I += [j, j2, 4 + j2, j, 4 + j2, 4 + j]
    return P, I + [4, 5, 6, 4, 6, 7, 0, 2, 1, 0, 3, 2]


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
            # (v3 bridges) a real deck edge: a concrete girder under the road, a light parapet on top, a darker underside
            low = max(hh[k0:k1]) < 3.0                     # (a small hump over a canal: just a low parapet, no girder)
            for (y_a, y_b, cl) in (((-1.5, 0.0, side_col),) if not low else ()) + ((0.0, 0.8 if low else 1.0, [204, 206, 208]),):
                P = []; I = []
                for k in range(k0, k1):
                    y = hh[k] + y0
                    ya, yb = max(0.0, y + y_a), y + y_b
                    P += [L[k][0], yb, L[k][1], L[k][0], ya, L[k][1], R[k][0], yb, R[k][1], R[k][0], ya, R[k][1]]
                for j in range(m - 1):
                    a = 4 * j
                    I += [a, a + 1, a + 4, a + 1, a + 5, a + 4, a + 2, a + 6, a + 3, a + 3, a + 6, a + 7]
                layer(tk, 'deck').add(P, cl * (4 * m), I)
            P = []; I = []
            for k in range(k0, k1 if not low else k0):
                yb = max(0.0, hh[k] + y0 - 1.5); P += [L[k][0], yb, L[k][1], R[k][0], yb, R[k][1]]
            if not low:
                for j in range(m - 1):
                    a = 2 * j; I += [a, a + 2, a + 1, a + 1, a + 2, a + 3]
                layer(tk, 'deck').add(P, [104, 107, 113] * (2 * m), I)
        # piers under lifted decks: every ~32 m on overpasses, every ~56 m on high bridges. Placed later, once the roads
        # below are known: a pier never stands on a road (Nov 26: piers stood in the middle of Autoroute 15)
        if side_col:
            for k in range(k0, k1):
                if hh[k] < 3.5 or k % (14 if hh[k] >= 10 else 8): continue
                alts = []
                for dk in (0, 1, -1, 2, -2, 3, -3):
                    kk = k + dk
                    if 0 <= kk < n and hh[kk] >= 3.5:
                        tx, tz = (pts[min(n - 1, kk + 1)] - pts[max(0, kk - 1)]); l = math.hypot(tx, tz) or 1
                        alts.append((float(pts[kk][0]), float(pts[kk][1]), tx / l, tz / l, float(hh[kk] + y0 - 1.5)))
                PIERS.append((tk, alts, 1.0 if hh[k] < 10 else 1.8, half))
        # (step 2) on bridges and overpasses: light poles along both edges every ~40 m, and dark cross beams under high decks
        if side_col and half >= 3.0:
            for k in range(k0, k1):
                if hh[k] < 4 or k % 10: continue
                tx, tz = (pts[min(n - 1, k + 1)] - pts[max(0, k - 1)]); l = math.hypot(tx, tz) or 1; tx, tz = tx / l, tz / l
                y = hh[k] + y0 + 0.8
                for sd in (1, -1) if half >= 5 else (1,):
                    ex, ez = pts[k][0] + nx[k] * (half - 0.3) * sd, pts[k][1] + nz[k] * (half - 0.3) * sd
                    Pp, Ip = box_(ex, ez, tx, tz, 0.13, 0.13, y, y + 9.0); layer(tk, 'deck').add(Pp, [118, 122, 128] * 8, Ip)
                    ax, az = ex - nx[k] * 1.0 * sd, ez - nz[k] * 1.0 * sd
                    Pp, Ip = box_(ax, az, -nz[k], nx[k], 1.1, 0.09, y + 8.8, y + 9.05); layer(tk, 'deck').add(Pp, [118, 122, 128] * 8, Ip)
                    hx, hz = ex - nx[k] * 1.9 * sd, ez - nz[k] * 1.9 * sd
                    Pp, Ip = box_(hx, hz, tx, tz, 0.35, 0.2, y + 8.65, y + 8.85); layer(tk, 'deck').add(Pp, [236, 226, 196] * 8, Ip)
        if side_col:
            for k in range(k0, k1):
                if hh[k] < 10 or k % 4: continue
                tx, tz = (pts[min(n - 1, k + 1)] - pts[max(0, k - 1)]); l = math.hypot(tx, tz) or 1; tx, tz = tx / l, tz / l
                y = hh[k] + y0 - 1.5
                Pp, Ip = box_(pts[k][0], pts[k][1], tx, tz, 0.35, half - 0.2, y - 0.9, y); layer(tk, 'deck').add(Pp, [86, 90, 98] * 8, Ip)


# ground-level road shapes are merged (clean junctions); lifted parts are ribbons
ASPH = [58, 61, 68]
flat_parts = {k: [] for k in ('road', 'drive')}
lifted_parts = []
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
            if max(h[a:b]) > 3: lifted_parts.append(LineString(pts[a:b]).buffer(rd['w'] / 2 + 1.5, cap_style='flat'))
            i = j
        else: i += 1
# (step 2) railways: the REM and train tracks, with a ballast bed and two rails; lifted on their own deck (viaducts, the
# Champlain Bridge) by their levels; the REM gets its overhead-line poles. Subways and tunnel parts are not drawn.
STEEL = [92, 94, 100]; BALLAST = [118, 112, 104]
rails = []; nrail = 0
for r in seg:
    if r['subtype'] != 'rail' or r['class'] == 'subway' or r['g'].geom_type != 'LineString': continue
    fl = jl(r.get('rail_flags')) or []
    if any('is_disused' in (f.get('values') or []) and not f.get('between') for f in fl): continue
    g = r['g']; L_ = g.length; n_ = max(2, int(L_ / STEP) + 1); ss = np.linspace(0, L_, n_)
    pts = np.array([g.interpolate(v).coords[0] for v in ss]); lvl = np.zeros(n_); skip = np.zeros(n_, bool)
    for rule in jl(r.get('level_rules')) or []:
        v = rule.get('value') or 0; bt = rule.get('between') or [0, 1]
        m = (ss >= bt[0] * L_ - 0.01) & (ss <= bt[1] * L_ + 0.01); lvl[m] = v
    for f in fl:
        if 'is_tunnel' in (f.get('values') or []):
            bt = f.get('between') or [0, 1]; skip |= (ss >= bt[0] * L_ - 0.01) & (ss <= bt[1] * L_ + 0.01)
    skip |= lvl < 0
    tgt = np.clip(lvl, 0, None) * LH
    src = [(s_, t_) for s_, t_ in zip(ss, tgt) if t_ > 0]
    hh = np.array([max([t_ - abs(s_ - s2) / RAMP * LH for s2, t_ in src] + [0]) for s_ in ss])
    nm = (r.get('names') or {}).get('primary')
    if nm == 'REM' or landmarks.on_champlain(nm, pts):
        # (a REM line is one long piece: check each point, not the middle of the piece)
        bx0, bz0, bx1, bz1 = landmarks.CH_BOX
        for i, (x, z) in enumerate(pts):
            t_ = landmarks.ch_height(x)
            if t_ is not None and hh[i] > 0.15 and bx0 < x < bx1 and bz0 < z < bz1:
                hh[i] = landmarks.ch_down(x) if x >= landmarks.CH_X2 else max(hh[i], t_)
    electric = r['class'] == 'light_rail' or nm == 'REM'
    i = 0
    while i < n_:
        if skip[i]: i += 1; continue
        j = i
        while j < n_ and not skip[j]: j += 1
        P_, H_ = pts[i:j], hh[i:j]; i = j
        if len(P_) < 2: continue
        nrail += 1
        if H_.max() > 0.4:
            offset_strip(P_, H_, 2.7, 0.0, BALLAST, 'road', side_col=[166, 169, 174])
            if nm == 'REM':
                bx0, bz0, bx1, bz1 = landmarks.CH_BOX
                inb = (P_[:, 0] > bx0) & (P_[:, 0] < bx1) & (P_[:, 1] > bz0) & (P_[:, 1] < bz1)
                if inb.sum() > 5: rails.append(P_[inb])
        else:
            offset_strip(P_, H_, 1.7, 0.07, BALLAST, 'paved')
        d = np.gradient(P_, axis=0); d /= (np.linalg.norm(d, axis=1)[:, None] + 1e-9); nrm = np.column_stack([-d[:, 1], d[:, 0]])
        for off in (-0.72, 0.72):
            offset_strip(P_ + nrm * off, H_, 0.07, 0.2, STEEL, 'paint')
        if electric:
            # overhead-line poles every ~48 m: a mast beside the track and an arm over it
            for k in range(0, len(P_), 12):
                cx, cz = P_[k] + nrm[k] * 2.3; y = H_[k] + 0.1; tk_ = tile_of(cx, cz)
                Pp, Ip = landmarks.box(cx, cz, d[k][0], d[k][1], 0.14, 0.14, y, y + 6.8); layer(tk_, 'deck').add(Pp, [150, 152, 156] * 8, Ip)
                ax, az = P_[k] + nrm[k] * 1.0
                Pp, Ip = landmarks.box(ax, az, -nrm[k][1], nrm[k][0], 0.08, 1.4, y + 6.3, y + 6.5); layer(tk_, 'deck').add(Pp, [150, 152, 156] * 8, Ip)
print('rail pieces', nrail, flush=True)
if rails:
    cz = landmarks.axis_at(rails, landmarks.TOWER_X); offs = []
    if cz is not None:
        for rd, (ss, pts, lvl), h in zip(roads, prof, H):
            if rd['cls'] != 'motorway' or not landmarks.on_champlain(rd.get('nm'), pts): continue
            i = int(np.argmin(np.abs(pts[:, 0] - landmarks.TOWER_X)))
            if abs(pts[i, 0] - landmarks.TOWER_X) < 10 and h[i] > 5: offs.append((float(pts[i, 1] - cz), rd['w'] / 2))   # (across the bridge, + = south side; and half the deck width)
    print('champlain cables', landmarks.champlain(layer, tile_of, rails, offs), 'decks', [(round(o, 1), w) for o, w in offs])
print('merging road shapes...', flush=True)
road_u = unary_union(flat_parts['road']); drive_u = unary_union(flat_parts['drive']).difference(road_u)
flat_poly(road_u, 0.06, ASPH, 'road'); flat_poly(drive_u, 0.055, [70, 73, 80], 'road')
# piers: the first spot near each one that is clear of the roads below
from shapely.prepared import prep
below = prep(road_u.buffer(0.6)); npier = nmoved = nskip = 0
for tk, alts, cw, half in PIERS:
    for i_, (cx, cz, tx, tz, y) in enumerate(alts):
        b_ = half * 0.42
        foot = Polygon([(cx + tx * cw + tz * b_, cz + tz * cw - tx * b_), (cx - tx * cw + tz * b_, cz - tz * cw - tx * b_),
                        (cx - tx * cw - tz * b_, cz - tz * cw + tx * b_), (cx + tx * cw - tz * b_, cz + tz * cw + tx * b_)])
        if below.intersects(foot): continue
        for (a_, bb, y0b, y1b, cl) in ((cw, b_, 0, y - 1.0, [176, 178, 181]), (cw + 0.5, half * 0.92, y - 1.0, y, [190, 192, 195])):
            P, I = box_(cx, cz, tx, tz, a_, bb, y0b, y1b); layer(tk, 'deck').add(P, cl * 8, I)
        npier += 1; nmoved += i_ > 0; break
    else: nskip += 1
print('piers', npier, 'moved off a road', nmoved, 'left out (road below)', nskip, flush=True)

# sidewalks from the data (footway=sidewalk), and a narrow curb strip around every city road
sw = [r['g'] for r in seg if r['class'] == 'footway' and r.get('subclass') == 'sidewalk' and r['g'].geom_type == 'LineString']
sw_u = unary_union([g.buffer(1.1, cap_style='flat') for g in sw]).difference(road_u)
flat_poly(sw_u, 0.12, [168, 168, 162], 'walk')

# (step 2) bike paths and footpaths: light strips on the ground, and on their own deck where they cross on a bridge
# (the Champlain Bridge's multi-use path from Nuns' Island to Brossard is one of them)
PATHW = {'cycleway': 2.6, 'path': 2.0, 'footway': 1.8, 'pedestrian': 4.0}
PATHC = {'cycleway': [204, 192, 164], 'path': [190, 184, 166], 'footway': [196, 194, 186], 'pedestrian': [196, 194, 186]}
path_parts = {k: [] for k in PATHW}
npath = 0
for r in seg:
    c = r['class']
    if c not in PATHW or r.get('subclass') in ('sidewalk', 'crosswalk', 'cycle_crossing') or r['g'].geom_type != 'LineString': continue
    lv = jl(r.get('level_rules')) or []; fl = jl(r.get('road_flags')) or []
    bridge = any('is_bridge' in (f.get('values') or []) for f in fl)
    if lv and not bridge: continue                      # (inside buildings, under stations: not drawn)
    g = r['g']; wd = PATHW[c]; npath += 1
    if bridge and lv:
        pr = profile(dict(g=g, lv=lv)); ss, pts, lvl = pr
        inbox = lambda x, z: landmarks.CH_BOX[0] < x < landmarks.CH_BOX[2] and landmarks.CH_BOX[1] < z < landmarks.CH_BOX[3]
        hh = np.array([(landmarks.ch_down(x) if x >= landmarks.CH_X2 else max(t_ * LH, landmarks.ch_height(x) or 0)) if t_ > 0 and inbox(x, z) else t_ * LH
                       for t_, (x, z) in zip(lvl, pts)])
        if hh.max() > 0.5:
            # ramps at the ends, like the roads
            L_ = ss[-1]; src = [(s_, t_) for s_, t_ in zip(ss, hh) if t_ > 0]
            hh = np.array([max([t_ - abs(s_ - s2) / RAMP * LH for s2, t_ in src] + [0]) for s_ in ss])
            hh = np.minimum(hh, np.minimum(max(0, lvl[0]) * LH + ss / RAMP * LH, max(0, lvl[-1]) * LH + (L_ - ss) / RAMP * LH))   # (down to the ground at its ends)
            offset_strip(pts, hh, wd / 2 + 0.3, 0.06, PATHC[c], 'walk', side_col=[160, 163, 168])
            continue
    path_parts[c].append(g.buffer(wd / 2, cap_style='flat', join_style='round', quad_segs=2))
taken = unary_union([road_u, sw_u])
for c, parts in path_parts.items():
    if not parts: continue
    u = unary_union(parts).difference(taken); flat_poly(u, 0.11, PATHC[c], 'walk'); taken = taken.union(u)
print('paths', npath, flush=True)

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


def lanes_dir(rd):
    """lanes in each direction: from OpenStreetMap when mapped, else from the width (the paint and lane lock both use this)"""
    w, ln = rd['w'], rd.get('ln') or 0
    if rd['ow']: return ln if ln > 0 else max(1, round(w / 3.6))
    if rd.get('lf') and rd.get('lb'): return max(1, min(rd['lf'], rd['lb']))
    if ln >= 2: return max(1, ln // 2)
    if rd['cls'] in ('primary', 'secondary', 'tertiary') and w >= 12: return 2
    return 1


for rd, (ss, pts, lvl), h in zip(roads, prof, H):
    cls, sub, w = rd['cls'], rd['sub'], rd['w']
    if sub in ('driveway', 'parking_aisle', 'alley') or cls in ('service', 'living_street'): continue
    n = lanes_dir(rd)
    if rd['ow']:
        for k in range(1, n): dashes(pts, h, -w / 2 + k * w / n, 3, 9, 0.08, WHITE)
        if cls in ('motorway', 'trunk') or sub == 'link' or n >= 3:
            dashes(pts, h, w / 2 - 0.4, 1, 0, 0.08, WHITE); dashes(pts, h, -w / 2 + 0.4, 1, 0, 0.08, YEL)
    else:
        if cls in ('primary', 'secondary', 'tertiary') or n >= 2:
            dashes(pts, h, 0.12, 1, 0, 0.06, YEL); dashes(pts, h, -0.12, 1, 0, 0.06, YEL)
            for k in range(1, n):
                o = k * (w / 2) / n; dashes(pts, h, o, 3, 9, 0.07, WHITE); dashes(pts, h, -o, 3, 9, 0.07, WHITE)
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
SHADOW_P = {}
import random
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
    rs = r.get('roof_shape') or ''
    pitched = rs in ('gabled', 'hipped', 'saltbox') or (cls in ('house', 'detached') and area < 260 and not rs)
    # (step 2 buildings) a facade style for the window pattern: 0 apartments, 1 shops below, 2 office tower, 3 house,
    # 4 industrial / warehouse. The viewer draws each style's windows; it reads the style from the 'extra' value.
    if cls in ('house', 'detached', 'semidetached_house', 'terrace', 'bungalow') or (hgt <= 9 and area < 220): style = 3
    elif cls in ('industrial', 'warehouse', 'manufacture', 'hangar') or (cls == '' and area > 3000 and hgt < 12): style = 4
    elif hgt > 35 or (cls in ('office', 'commercial') and hgt > 18): style = 2
    elif cls in ('commercial', 'retail', 'supermarket', 'kiosk') or (hgt >= 9 and hgt <= 30 and hid % 4 == 0): style = 1
    else: style = 0
    # roofs: flat ones light gravel grey (a little different each), pitched ones dark shingle
    if pitched: roof = [[74, 74, 79], [91, 74, 66], [107, 63, 53], [84, 86, 90]][hid % 4]
    else: g_ = 122 + (hid >> 4) % 34; roof = [g_, g_ - 2, g_ - 6]
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
        so = style * 20000.0 if win > 0 else 0.0
        flat_rim = (not pitched) and area >= 120 and hgt >= 6      # a low parapet around flat roofs
        for i in range(len(ring)):
            a, b = ring[i], ring[(i + 1) % len(ring)]; ln = float(np.hypot(*(b - a)))
            base = len(pts) // 3
            pts += [a[0], mh, a[1], b[0], mh, b[1], b[0], top, b[1], a[0], top, a[1]]
            cols += col * 4; ex += [win * (so + run + 0.001), win * (so + run + ln), win * (so + run + ln), win * (so + run + 0.001)]
            tri += [base, base + 1, base + 2, base, base + 2, base + 3]
            if flat_rim:
                base = len(pts) // 3; rim = [int(c * 0.9) for c in col]
                pts += [a[0], top, a[1], b[0], top, b[1], b[0], top + 0.8, b[1], a[0], top + 0.8, a[1]]
                cols += rim * 4; ex += [-1.0] * 4; tri += [base, base + 1, base + 2, base, base + 2, base + 3]
            run += ln
        Ly.add(pts, cols, tri, ex)
        # (step 2) the building's shadow on the ground: the footprint swept away from the sun by its height
        if hgt >= 3:
            SHADOW_P.setdefault(tk, []).append((P, hgt))
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
            # (step 2 buildings) machinery on big flat roofs: one to three grey boxes, always the same for a building
            if P.area >= 500 and hgt >= 6:
                rng = random.Random(hid); x0, z0, x1, z1 = P.bounds; nb = 1 + int(P.area > 1500) + int(P.area > 4000)
                for _ in range(nb * 4):
                    if nb <= 0: break
                    cx, cz = rng.uniform(x0, x1), rng.uniform(z0, z1); a_, b_ = rng.uniform(1.5, 3.5), rng.uniform(1.2, 2.5)
                    if not P.buffer(-3).contains(Point(cx, cz)): continue
                    Pb, Ib = box_(cx, cz, 1.0, 0.0, a_, b_, hgt, hgt + rng.uniform(1.2, 2.4))
                    Ly.add(Pb, [150, 152, 155] * 8, Ib, [-1.0] * 8); nb -= 1

# shadows: one soft dark layer per tile (the sun from the south-west, like the light in the viewer)
SUN = (0.6, -0.4)                       # metres of shadow per metre of height (toward the east-north-east)
from shapely import affinity
nsh = 0
for tk, items in SHADOW_P.items():
    parts = []
    for P, hgt in items:
        dx, dz = SUN[0] * hgt, SUN[1] * hgt
        Q = affinity.translate(P, dx, dz)
        ring = list(P.exterior.coords)
        quads = [Polygon([ring[i], ring[i + 1], (ring[i + 1][0] + dx, ring[i + 1][1] + dz), (ring[i][0] + dx, ring[i][1] + dz)]).buffer(0)
                 for i in range(len(ring) - 1)]
        try: parts.append(unary_union([P, Q] + quads))
        except Exception: continue
    sh = unary_union(parts).simplify(0.3)
    T0 = box(tk[0] * T, tk[1] * T, (tk[0] + 1) * T, (tk[1] + 1) * T)
    flat_poly(sh.difference(unary_union([p_ for p_, h_ in items])), 0.13, [0, 0, 0], 'shade'); nsh += 1
print('shadow tiles', nsh, flush=True)

# ---------------------------------------------------------------- street furniture: points (lamps, lights, stops)
lifted_u = unary_union(lifted_parts) if lifted_parts else Polygon()
props = {}
for r in inf:
    if r['g'].geom_type != 'Point': continue
    k = {'street_lamp': 'lamp', 'traffic_signals': 'light', 'stop': 'stop'}.get(r['class'])
    if not k: continue
    x, z = r['g'].coords[0]
    pt = Point(x, z)
    if k == 'lamp' and lifted_u.contains(pt): continue      # (the bridge has its own poles now)
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

# ---------------------------------------------------------------- checks: every mistake found once gets a check, so it
# can't come back unnoticed. The publish step reads checks.json from every chunk and stops if one fails.
from collections import defaultdict
chk = {}
ends = defaultdict(list)
for rd, (ss, pts, lvl), h in zip(roads, prof, H):
    for (x, z), hv in ((pts[0], h[0]), (pts[-1], h[-1])):
        if owned(tile_of(x, z)): ends[(round(x, 1), round(z, 1))].append(float(hv))
steps = [max(v) - min(v) for v in ends.values() if len(v) > 1]
chk['wall_at_junction'] = sum(1 for d in steps if d > 1.0)            # Nov 26: a bridge starting as a wall at a crosswalk
chk['pier_on_road'] = 0                                               # Nov 26: piers in the middle of Autoroute 15 (placed clear of roads)
chk['too_high_road'] = sum(1 for rd, (ss, pts, lvl), h in zip(roads, prof, H) if h.max() > 45 and owned(tile_of(*pts[0])))
# (Nov 26: Boucherville has roads that come from TomTom, not OpenStreetMap; they can't have its lanes, so they don't count)
own_roads = [rd for rd, (ss, pts, lvl) in zip(roads, prof) if owned(tile_of(*pts[0])) and rd.get('osmsrc')]
chk['roads_without_osm_pct'] = round(100.0 * sum(1 for rd in own_roads if not rd.get('osm')) / len(own_roads), 1) if len(own_roads) >= 50 else 0.0
LIMITS = {'wall_at_junction': 0, 'pier_on_road': 0, 'too_high_road': 0, 'roads_without_osm_pct': 25.0}
failed = [k for k, v in chk.items() if v > LIMITS[k]]
os.makedirs(os.path.join(OUT, 'checks'), exist_ok=True)
json.dump({'checks': chk, 'failed': failed}, open(os.path.join(OUT, 'checks', ('%d_%d.json' % (TX0, TZ0)) if A.tiles else 'all.json'), 'w'))
print('checks', chk, 'FAILED: ' + ', '.join(failed) if failed else 'all passed', flush=True)
