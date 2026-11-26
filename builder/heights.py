"""Real building heights from the City of Montréal laser surface model (MNS terrain+buildings 2015, 1 m, CC BY 4.0).

  python heights.py <borough id> plan.json idx.json out.json

Downloads that borough's file, reads the Overture buildings over it, and for each one:
height = typical roof level inside the footprint (60th percentile) minus the ground just around it (10th percentile)."""
import sys, os, json, glob, zipfile, shutil, tempfile, urllib.request
import numpy as np, rasterio
from rasterio.features import geometry_mask
from rasterio.windows import from_bounds
from shapely import wkb
import shapely
from pyproj import Transformer
sys.path.insert(0, os.path.dirname(__file__))
import overture
from plan import UA

bid, plan, idx, out = sys.argv[1], json.load(open(sys.argv[2])), json.load(open(sys.argv[3])), sys.argv[4]
url = [b['url'] for b in plan['boroughs'] if b['id'] == bid][0]
tmp = tempfile.mkdtemp()
zp = os.path.join(tmp, 'mns.zip')
for a in range(4):
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': UA}), timeout=300) as r, open(zp, 'wb') as f:
            shutil.copyfileobj(r, f, 2 ** 22)
        break
    except Exception as e:
        if a == 3: raise
        print('retry download', e, flush=True)
print(bid, 'downloaded MB', os.path.getsize(zp) // 2 ** 20, flush=True)
zipfile.ZipFile(zp).extractall(tmp); os.remove(zp)
tif = glob.glob(os.path.join(tmp, '**', '*.tif'), recursive=True)[0]
src = rasterio.open(tif); B = src.bounds
back = Transformer.from_crs(src.crs, 'EPSG:4326', always_xy=True)
lo = [back.transform(x, y) for x in (B.left, B.right) for y in (B.bottom, B.top)]
bbox = [min(p[0] for p in lo), min(p[1] for p in lo), max(p[0] for p in lo), max(p[1] for p in lo)]
overture.read(idx, bbox, tmp, ['bld'])
R = json.load(open(os.path.join(tmp, 'bld.json')))
T = Transformer.from_crs('EPSG:4326', src.crs, always_xy=True)
res = {}
for r in R:
    g = wkb.loads(bytes.fromhex(r['geometry']))
    if g.geom_type not in ('Polygon', 'MultiPolygon') or g.area * 1e10 < 20: continue
    gp = shapely.transform(g, lambda a: np.column_stack(T.transform(a[:, 0], a[:, 1])))
    x0, y0, x1, y1 = gp.bounds
    if x0 < B.left or x1 > B.right or y0 < B.bottom or y1 > B.top: continue
    win = from_bounds(x0 - 10, y0 - 10, x1 + 10, y1 + 10, src.transform).round_offsets().round_lengths()
    a = src.read(1, window=win); tr = src.window_transform(win)
    if a.size == 0: continue
    valid = a > -1000
    inner = gp.buffer(-0.8) if gp.area > 60 else gp
    if inner.is_empty: inner = gp
    m_in = ~geometry_mask([inner], a.shape, tr) & valid
    m_out = ~geometry_mask([gp.buffer(7).difference(gp.buffer(1.5))], a.shape, tr) & valid
    if m_in.sum() < 6 or m_out.sum() < 10: continue
    h = float(np.percentile(a[m_in], 60)) - float(np.percentile(a[m_out], 10))
    if 1.5 < h < 250: res[r['id']] = round(h, 1)
os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
json.dump(res, open(out, 'w'), separators=(',', ':'))
print(bid, 'buildings', len(R), 'heights', len(res), flush=True)
shutil.rmtree(tmp)
