"""Join the chunk outputs into one map: site/index.json lists every tile and its size."""
import sys, os, json, glob, time
sys.path.insert(0, os.path.dirname(__file__))
from plan import LON0, LAT0, KX, KY, T

site = sys.argv[1]; plan = json.load(open(sys.argv[2]))
tiles = []
for f in glob.glob(os.path.join(site, 'parts', '*.json')):
    tiles += json.load(open(f))['tiles']
tiles.sort()
nets = sorted(os.path.basename(f)[:-8] for f in glob.glob(os.path.join(site, 'n', '*.json.gz')))
idx = {'v': 2, 'format': 'MK3Q', 'lon0': LON0, 'lat0': LAT0, 'kx': KX, 'ky': KY, 'T': T, 'region': plan['region'],
       'bbox': plan['bbox'], 'built': time.strftime('%Y-%m-%d'), 'overture': json.load(open(sys.argv[3]))['release'],
       'credit': '© OpenStreetMap contributors · Overture Maps Foundation · Ville de Montréal (CC BY 4.0)',
       'tiles': tiles, 'net': nets}
json.dump(idx, open(os.path.join(site, 'index.json'), 'w'), separators=(',', ':'))
mb = sum(t[2] for t in tiles) / 1e6
print('tiles', len(tiles), 'MB %.1f' % mb, 'net tiles', len(nets))
# the builder's checks (walls at junctions, piers on roads, ...): any failed chunk stops the publish
bad = []
for f in glob.glob(os.path.join(site, 'checks', '*.json')):
    c = json.load(open(f))
    if c['failed']: bad.append((os.path.basename(f), c['failed'], c['checks']))
for b in bad: print('CHECK FAILED in chunk', *b)
if bad: sys.exit('map checks failed, not publishing (the live map stays as it was)')
if len(tiles) < 0.5 * len(plan['chunks']) * 8:   # far too few tiles: stop before replacing a good map
    sys.exit('too few tiles, not publishing')
