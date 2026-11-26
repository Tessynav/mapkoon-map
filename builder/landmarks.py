"""Landmarks: places people recognise, made with their real shape instead of the general rules.

Samuel-De Champlain Bridge (Montréal - Brossard, 2019): 3.4 km, a single white pylon on the Seaway dike, cables in a
semi-fan to both decks, the REM's transit deck between the two road decks, and the deck rising over the St. Lawrence
Seaway (ships need about 36 m under it). Positions are in the map's local metres (x east, z south)."""
import math
import numpy as np

# along the bridge: where it starts climbing, where it reaches the top, where it starts down, where it lands (x, metres)
CH_X0, CH_X1, CH_X2, CH_X3 = 3300.0, 4250.0, 4900.0, 5560.0
CH_LOW, CH_TOP = 13.0, 34.0          # deck height over the river, and over the Seaway (metres)
CH_BOX = (1700.0, -1500.0, 5800.0, -1000.0)   # x0, z0, x1, z1: the bridge's area
TOWER_X = 4460.0                     # the pylon stands on the Seaway dike
PYLON_H = 168.0


def ch_height(x):
    """deck height of the Champlain Bridge at x: level over the river, a long smooth climb, level over the Seaway, down"""
    def ease(u): u = min(1.0, max(0.0, u)); return u * u * (3 - 2 * u)
    if x < CH_X0 or x > CH_X3: return None
    if x < CH_X1: return CH_LOW + (CH_TOP - CH_LOW) * ease((x - CH_X0) / (CH_X1 - CH_X0))
    if x < CH_X2: return CH_TOP
    return CH_TOP - (CH_TOP - CH_LOW) * ease((x - CH_X2) / (CH_X3 - CH_X2))


def on_champlain(name, pts):
    """a piece of road (or track) that is the Champlain Bridge itself"""
    if not name or 'Samuel-De Champlain' not in name and name != 'REM': return False
    x0, z0, x1, z1 = CH_BOX
    c = pts.mean(axis=0)
    return x0 < c[0] < x1 and z0 < c[1] < z1


def apply_heights(roads, prof, H):
    """raise the Champlain Bridge's decks to the real profile (only where the data already has them lifted)"""
    n = 0
    for rd, (ss, pts, lvl), h in zip(roads, prof, H):
        if not on_champlain(rd.get('nm'), pts): continue
        for i, (x, z) in enumerate(pts):
            t = ch_height(x)
            if t is not None and h[i] > 0.15: h[i] = max(h[i], t); n += 1
    return n


def axis_at(rails, x):
    """the middle of the bridge at x: between the two REM tracks"""
    zs = []
    for pts in rails:
        xs = pts[:, 0]
        if xs.min() <= x <= xs.max():
            o = np.argsort(xs); zs.append(float(np.interp(x, xs[o], pts[o, 1])))
    return sum(zs) / len(zs) if zs else None


def box(cx, cz, ux, uz, a, b, y0, y1, a1=None, b1=None):
    """a (tapered) box: centre (cx, cz), half-sizes a along (ux, uz) and b across it; a1/b1 at the top"""
    a1 = a if a1 is None else a1; b1 = b if b1 is None else b1
    vx, vz = -uz, ux
    P = []
    for (A, B, y) in ((a, b, y0), (a1, b1, y1)):
        for (sa, sb) in ((1, 1), (-1, 1), (-1, -1), (1, -1)):
            P += [cx + ux * A * sa + vx * B * sb, y, cz + uz * A * sa + vz * B * sb]
    I = []
    for j in range(4):
        j2 = (j + 1) % 4; I += [j, j2, 4 + j2, j, 4 + j2, 4 + j]
    I += [4, 5, 6, 4, 6, 7]
    return P, I


def cable(p, q, w=0.18):
    """a thin cable from p to q (two crossed strips, so it shows from every side)"""
    p = np.array(p, float); q = np.array(q, float); d = q - p; d /= np.linalg.norm(d)
    s1 = np.cross(d, [0, 1, 0]); s1 /= (np.linalg.norm(s1) + 1e-9); s2 = np.cross(d, s1)
    P = []; I = []
    for k, s in enumerate((s1, s2)):
        for v in (p + s * w, p - s * w, q - s * w, q + s * w): P += [float(v[0]), float(v[1]), float(v[2])]
        b = 4 * k; I += [b, b + 1, b + 2, b, b + 2, b + 3]
    return P, I


def champlain(layer, tile_of, rails, road_offsets):
    """the pylon and its cables. rails: REM track points on the bridge; road_offsets: (how far each road deck's centre is
    from the middle of the bridge at the pylon, + = south side; half the deck's width), metres"""
    cz = axis_at(rails, TOWER_X)
    if cz is None: return 0
    za = axis_at(rails, TOWER_X - 50); zb = axis_at(rails, TOWER_X + 50)
    ux, uz = 100.0, (zb - za) if za is not None and zb is not None else 0.0; l = math.hypot(ux, uz); ux, uz = ux / l, uz / l
    vx, vz = -uz, ux                                   # across the bridge
    deck = ch_height(TOWER_X)
    WHITE = [238, 238, 234]; CAB = [214, 218, 224]
    tk = tile_of(TOWER_X, cz); P0 = []; C0 = []; I0 = []

    def add(P, I, col):
        b = len(P0) // 3; P0.extend(P); I0.extend(i + b for i in I); C0.extend(col * (len(P) // 3))

    # footing on the dike, then two slender legs either side of the transit deck, leaning in a little (a tuning fork)
    add(*box(TOWER_X, cz, ux, uz, 9, 14, 0, 4), [205, 205, 200])
    legs = []
    for side in (1, -1):
        bx, bz = TOWER_X + vx * 7.5 * side, cz + vz * 7.5 * side
        tx, tz = TOWER_X + vx * 5.0 * side, cz + vz * 5.0 * side
        # the leg in two parts so it can lean: base to deck, deck to top
        mid = deck + 6
        mx, mz = bx + (tx - bx) * mid / PYLON_H, bz + (tz - bz) * mid / PYLON_H
        P, I = box(bx, bz, ux, uz, 3.4, 2.2, 4, mid, 3.0, 2.0); add(P, I, WHITE)
        P, I = box(mx, mz, ux, uz, 3.0, 2.0, mid, PYLON_H, 2.0, 1.5)
        # (lean: move the top ring to the top point)
        for j in range(4, 8): P[3 * j] += tx - mx; P[3 * j + 2] += tz - mz
        add(P, I, WHITE); legs.append((side, tx, tz, mx, mz, mid))
    # cross beam under the decks
    add(*box(TOWER_X, cz, ux, uz, 2.5, 9.5, deck - 4.5, deck - 1.5), WHITE)
    # cables: from each leg to the inner edge of the road deck on its side, along both spans (a semi-fan)
    ncab = 0
    for (side, tx, tz, mx, mz, mid) in legs:
        offs = [(o, hw) for o, hw in road_offsets if o * side > 0]
        if not offs: continue
        inner = min(abs(o) - hw for o, hw in offs) + 0.6   # the deck's edge nearest the middle
        for (sgn, span, n) in ((-1, 215.0, 13), (1, 285.0, 17)):
            for i in range(n):
                f = (i + 1) / n
                along = sgn * (18 + (span - 18) * f)
                ax = TOWER_X + ux * along + vx * inner * side; az = cz + uz * along + vz * inner * side
                ay = (ch_height(ax) or deck) + 1.0
                hy = 96 + (PYLON_H - 8 - 96) * f            # farther cables start higher on the pylon
                u = (hy - mid) / (PYLON_H - mid)
                px = mx + (tx - mx) * u + ux * sgn * 1.6; pz = mz + (tz - mz) * u + uz * sgn * 1.6
                P, I = cable((px, hy, pz), (ax, ay, az)); add(P, I, CAB); ncab += 1
    layer(tk, 'lm').add(P0, C0, I0)
    return ncab
