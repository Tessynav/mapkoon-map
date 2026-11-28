"""Real road widths from the City of Montréal's curb outlines (Actifs de voirie, CC BY 4.0).

  python builder/city.py prep CITY_DIR out.json.gz
      reads the city's GeoJSON (roadway "chaussée" pieces and islands) and writes them small, in map metres.
  build.py then calls widths(roads, CITY): each road's width is measured across the real roadway, every 8 m,
  and the middle value is used when the measures agree. Roads the city doesn't have (other towns, private roads,
  parking aisles) keep their estimated width.
"""
import sys, os, json, gzip, glob, math, zipfile, io
import numpy as np

LON0, LAT0 = -73.5625, 45.4575
KX = 111320 * math.cos(math.radians(LAT0)); KY = 110540


def _feats(path):
    raw = open(path, 'rb').read()
    if raw[:2] == b'\x1f\x8b': raw = gzip.decompress(raw)
    if raw[:2] == b'PK':
        z = zipfile.ZipFile(io.BytesIO(raw)); raw = z.read([n for n in z.namelist() if n.lower().endswith(('.json', '.geojson'))][0])
    return json.loads(raw.decode('utf-8', 'replace'))['features']


def prep(src, out):
    res = {'c': [], 'i': [], 'credit': 'Contient des données de la Ville de Montréal, CC BY 4.0'}
    for key, pat in (('c', 'chaussee'), ('i', 'ilot')):
        files = [f for f in glob.glob(os.path.join(src, '*')) if pat in os.path.basename(f).lower() and f.lower().endswith('.zip.gz')]
        for f in files:
            for ft in _feats(f):
                g = ft.get('geometry')
                if not g: continue
                polys = [g['coordinates']] if g['type'] == 'Polygon' else g['coordinates'] if g['type'] == 'MultiPolygon' else []
                cat = (ft.get('properties') or {}).get('CATEGORIECHAUSSEE_REF') or (ft.get('properties') or {}).get('TYPEILOT_REF') or ''
                for p in polys:
                    rings = [[[round((x - LON0) * KX, 1), round(-(y - LAT0) * KY, 1)] for x, y, *_ in r] for r in p]
                    res[key].append([cat[:24], rings])
        print(key, len(res[key]), 'polygons from', len(files), 'file(s)')
    with gzip.open(out, 'wt') as fo: json.dump(res, fo, separators=(',', ':'))
    print('written', out, os.path.getsize(out) // 1024, 'KB')


def load(path, box=None):
    """the city polygons as shapely shapes (roadway minus islands), only inside box (x0, z0, x1, z1) when given"""
    from shapely.geometry import Polygon, box as sbox
    from shapely.ops import unary_union
    if not path or not os.path.exists(path): return None
    d = json.load(gzip.open(path, 'rt'))
    B = sbox(*box) if box else None

    def polys(key):
        out = []
        for cat, rings in d[key]:
            if len(rings[0]) < 4: continue
            xs = [p[0] for p in rings[0]]; zs = [p[1] for p in rings[0]]
            if box and (max(xs) < box[0] or min(xs) > box[2] or max(zs) < box[1] or min(zs) > box[3]): continue
            p = Polygon(rings[0], rings[1:])
            if not p.is_valid: p = p.buffer(0)
            if not p.is_empty: out.append(p)
        return out
    road = polys('c')
    if not road: return None
    U = unary_union([p.buffer(0.05) for p in road])      # (neighbour pieces of the same street touch: one surface)
    isl = polys('i')
    if isl: U = U.difference(unary_union(isl))           # medians and roundabout islands are not roadway
    return U


def widths(roads, U, step=5.0, keep_end=6.0, log=print):
    """sets rd['w'] to the measured curb-to-curb width where the city has the road; keeps rd['w0'], the estimate"""
    from shapely.geometry import LineString, Point
    from shapely.prepared import prep as sprep
    from shapely.strtree import STRtree
    if U is None or U.is_empty:
        log('city widths: no city data here'); return 0
    PU = sprep(U)
    lines = [rd['g'] for rd in roads]; tree = STRtree(lines)
    CAP = {'motorway': 18, 'trunk': 18, 'primary': 24, 'secondary': 20, 'tertiary': 18, 'residential': 16,
           'unclassified': 14, 'living_street': 10}
    done = asym = 0; stats = []; why = {}
    def no(k): why[k] = why.get(k, 0) + 1
    for i, rd in enumerate(roads):
        rd['w0'] = rd['w']
        if rd['cls'] not in CAP or rd['sub'] in ('driveway', 'parking_aisle', 'alley'): continue
        no('streets')
        g = rd['g']; L = g.length
        if L < 2 * keep_end + 4: no('short'); continue
        ws = []; offs = []
        for s in np.arange(keep_end, L - keep_end + 0.01, step):
            m = g.interpolate(s); a = g.interpolate(max(0, s - 2)); b = g.interpolate(min(L, s + 2))
            dx, dz = b.x - a.x, b.y - a.y; n = math.hypot(dx, dz)
            if n == 0 or not PU.contains(m): continue           # (inside an intersection, or not a city street)
            nx, nz = -dz / n, dx / n
            ray = LineString([(m.x - nx * 25, m.y - nz * 25), (m.x + nx * 25, m.y + nz * 25)])
            hit = ray.intersection(U); seg = None
            for q in getattr(hit, 'geoms', [hit]):
                if q.geom_type == 'LineString' and q.distance(m) < 0.01: seg = q
            if seg is None: continue
            p0, p1 = seg.coords[0], seg.coords[-1]
            dl = math.hypot(p0[0] - m.x, p0[1] - m.y); dr = math.hypot(p1[0] - m.x, p1[1] - m.y)
            # another road running alongside within the same surface (a divided road with only paint between, or
            # a service road next to it): stop half way to it
            for j in tree.query(ray):
                if j == i: continue
                o = lines[j]; x = o.intersection(ray)
                if x.is_empty or x.geom_type != 'Point': continue
                oc = np.array(o.coords); k = int(np.argmin(np.hypot(oc[:, 0] - x.x, oc[:, 1] - x.y)))
                k2 = min(k + 1, len(oc) - 1); k1 = max(k - 1, 0)
                ox, oz = oc[k2] - oc[k1]; on = math.hypot(ox, oz)
                if on == 0 or abs(ox * dx + oz * dz) / (on * n) < 0.85: continue   # (crossing, not alongside)
                d = math.hypot(x.x - m.x, x.y - m.y); side = (x.x - m.x) * nx + (x.y - m.y) * nz
                if side < 0: dl = min(dl, d / 2)
                else: dr = min(dr, d / 2)
            ws.append(dl + dr); offs.append((dr - dl) / 2)
        if len(ws) < 2: no('few samples (not a city street?)'); continue
        ws = np.array(ws); med = float(np.median(ws)); q1, q3 = np.percentile(ws, [25, 75])
        if q3 - q1 > max(3.0, 0.35 * med): no('measures disagree'); continue   # (keep the estimate)
        if not (3.0 <= med <= CAP[rd['cls']]): no('out of range'); continue
        rd['w'] = round(med, 1); rd['wsrc'] = 'city'; done += 1; stats.append(med - rd['w0'])
        if abs(float(np.median(offs))) > 1.0: asym += 1
    # short pieces (and ones whose measures disagreed) between measured pieces of the same street: the width of the
    # measured piece they continue straight from (no sudden narrow piece in the middle of a wide street)
    ends = {}
    for i, rd in enumerate(roads):
        c = rd['g'].coords
        for k, p in ((0, c[0]), (1, c[-1])): ends.setdefault((round(p[0]), round(p[1])), []).append((i, k))

    def dirv(i, k):
        c = np.array(roads[i]['g'].coords); a, b = (c[0], c[min(3, len(c) - 1)]) if k == 0 else (c[-1], c[max(-4, -len(c))])
        v = b - a; n = np.hypot(*v); return v / n if n else v
    filled = 0
    for _ in range(4):
        for i, rd in enumerate(roads):
            if rd.get('wsrc') or rd['cls'] not in CAP or rd['sub'] in ('driveway', 'parking_aisle', 'alley'): continue
            c = rd['g'].coords; best = None
            for k, p in ((0, c[0]), (1, c[-1])):
                for j, kj in ends.get((round(p[0]), round(p[1])), []):
                    o = roads[j]
                    if j == i or o.get('wsrc') not in ('city', 'city-next') or o['cls'] != rd['cls'] or o['ow'] != rd['ow']: continue
                    if float(np.dot(dirv(i, k), dirv(j, kj))) > -0.9: continue      # (not straight on)
                    best = o['w'] if best is None else (best + o['w']) / 2
            if best is not None: rd['w'] = best; rd['wsrc'] = 'city-next'; filled += 1
    if filled: log('city widths: %d more short pieces given the width of the street they continue' % filled)
    if stats:
        st = np.array(stats)
        log('city widths: %d of %d roads measured; vs the estimate: median %+.1f m, wider by >1.5 m %d, narrower by >1.5 m %d; '
            'off-centre by >1 m %d; %s' % (done, len(roads), float(np.median(st)), int((st > 1.5).sum()), int((st < -1.5).sum()), asym, why))
    return done


if __name__ == '__main__':
    if sys.argv[1] == 'prep': prep(sys.argv[2], sys.argv[3])
