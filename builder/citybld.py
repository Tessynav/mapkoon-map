"""City of Montréal property assessment units -> a small file of lots: [outline (map metres), floors, year built].

  python builder/citybld.py uef.(geojson|zip|gz) out.json.gz
"""
import sys, json, gzip, zipfile, io, math

LON0, LAT0 = -73.5625, 45.4575
KX = 111320 * math.cos(math.radians(LAT0)); KY = 110540


def read(path):
    raw = open(path, 'rb').read()
    if raw[:2] == b'\x1f\x8b': raw = gzip.decompress(raw)
    if raw[:2] == b'PK':
        z = zipfile.ZipFile(io.BytesIO(raw)); raw = z.read([n for n in z.namelist() if n.lower().endswith(('.json', '.geojson'))][0])
    return json.loads(raw.decode('utf-8', 'replace'))


d = read(sys.argv[1]); F = d['features']
print('features', len(F)); print('fields', list((F[0].get('properties') or {}).keys()) if F else [])
out = []; nf = ny = 0


def num(v):
    try: return int(float(v))
    except Exception: return 0


for f in F:
    p = f.get('properties') or {}; g = f.get('geometry')
    if not g: continue
    fl = num(p.get('ETAGE_HORS_SOL') or p.get('NOMBRE_ETAGE') or 0)
    yr = num(p.get('ANNEE_CONSTRUCTION') or p.get('annee_construction') or 0)
    if fl <= 0: continue
    nf += 1; ny += 1 if yr > 1800 else 0
    polys = [g['coordinates']] if g['type'] == 'Polygon' else g['coordinates'] if g['type'] == 'MultiPolygon' else []
    for poly in polys:
        ring = [[round((x - LON0) * KX, 1), round(-(y - LAT0) * KY, 1)] for x, y, *_ in poly[0]]
        if len(ring) >= 4: out.append([ring, fl, yr if yr > 1800 else 0, num(p.get('NOMBRE_LOGEMENT') or 0)])
print('lots with floors', nf, 'with year', ny, 'kept', len(out))
with gzip.open(sys.argv[2], 'wt') as fo: json.dump({'lots': out, 'credit': 'Ville de Montréal, CC BY 4.0'}, fo, separators=(',', ':'))
import os; print('written', sys.argv[2], os.path.getsize(sys.argv[2]) // 1024, 'KB')
