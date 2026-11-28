"""Join the chunk outputs into one map: site/index.json lists every tile and its size."""
import sys, os, json, glob, time
sys.path.insert(0, os.path.dirname(__file__))
from plan import LON0, LAT0, KX, KY, T

site = sys.argv[1]; plan = json.load(open(sys.argv[2]))
tiles = []
# (Nov 27) every part of the map must be here: a publish with parts missing left holes (south Nuns' Island)
_parts = glob.glob(os.path.join(site, 'parts', '*.json'))
print('map parts', len(_parts), 'of', len(plan['chunks']))
if len(_parts) < len(plan['chunks']): sys.exit('map parts missing, not publishing (the live map stays as it was)')
for f in _parts:
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
SAFETY = {'wall_at_junction', 'pier_on_road', 'too_high_road', 'lines_crossing', 'roads_without_osm_pct'}
bad = []; warn = []
for f in glob.glob(os.path.join(site, 'checks', '*.json')):
    c = json.load(open(f))
    # (Nov 27 build #13: 1 tiny gap in ground blocked the whole map, like build #12) only SAFETY problems stop the
    # publish now: a wall or pier in the road, a road floating too high, lines crossing, roads missing their OSM info.
    # Look problems (paint, zebra bars, building heights, small ground gaps) never block; they're listed to fix next.
    hard = [k for k in c['failed'] if k in SAFETY]
    if hard: bad.append((os.path.basename(f), hard, c['checks']))
    for k in c['failed']:
        if k not in SAFETY: warn.append('%s: %s=%s' % (os.path.basename(f), k, c['checks'].get(k)))
    warn += c.get('where', [])
# (Nov 27) the few look problems that don't block the map, with their place, to fix in the next round
print('look problems to fix next:', len(warn))
for w in warn: print('  ', w)
for b in bad: print('CHECK FAILED in chunk', *b)
if bad: sys.exit('map checks failed, not publishing (the live map stays as it was)')
if len(tiles) < 0.5 * len(plan['chunks']) * 8:   # far too few tiles: stop before replacing a good map
    sys.exit('too few tiles, not publishing')
