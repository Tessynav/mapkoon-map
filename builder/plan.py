"""Split a region into chunks of tiles (each built by its own job) and list the City of Montréal height files.

  python plan.py <region> <outdir>     -> outdir/plan.json, and 'chunks=' / 'boroughs=' lines for GitHub Actions"""
import sys, os, json, math, urllib.request

LON0, LAT0 = -73.5625, 45.4575
KX = 111320 * math.cos(math.radians(LAT0)); KY = 110540
T = 500.0
CH = 8        # tiles per chunk side (4 km)
PAD = 300.0   # metres of data read around each chunk, so shapes that cross the edge are whole
UA = 'Mozilla/5.0 (X11; Linux x86_64) MapKoon map builder'
REGIONS = {
    # Montréal island, Laval's south edge and the South Shore (Longueuil, Brossard, Saint-Lambert, La Prairie, Boucherville)
    'montreal': [-73.98, 45.37, -73.38, 45.71],
    # the small test area used for the preview (Verdun, Nuns' Island, Pointe-Saint-Charles)
    'test': [-73.59, 45.44, -73.49, 45.48],
}


def to_xz(lon, lat): return (lon - LON0) * KX, -(lat - LAT0) * KY
def to_ll(x, z): return x / KX + LON0, LAT0 - z / KY


def chunk_bbox(t):
    """lon/lat box of a chunk's tiles plus the padding"""
    tx0, tz0, tx1, tz1 = t
    w, n = to_ll(tx0 * T - PAD, tz0 * T - PAD); e, s = to_ll(tx1 * T + PAD, tz1 * T + PAD)
    return [round(w, 6), round(s, 6), round(e, 6), round(n, 6)]


def boroughs():
    """the 33 laser surface files (one per borough or town), from the city's open data catalogue"""
    u = 'https://donnees.montreal.ca/api/3/action/package_show?id=modele-numerique-de-surface-mns'
    d = json.load(urllib.request.urlopen(urllib.request.Request(u, headers={'User-Agent': UA}), timeout=60))
    out = []
    for r in d['result']['resources']:
        if r['url'].endswith('.zip'):
            out.append({'id': r['url'].rsplit('/', 1)[1][:-4].replace('mnsterrainbatiment_2015_1m_', ''), 'url': r['url']})
    return out


if __name__ == '__main__':
    reg, outdir = sys.argv[1], sys.argv[2]; os.makedirs(outdir, exist_ok=True)
    W, S, E, N = REGIONS[reg]
    x0, z0 = to_xz(W, N); x1, z1 = to_xz(E, S)
    TX0, TZ0, TX1, TZ1 = math.floor(x0 / T), math.floor(z0 / T), math.floor(x1 / T) + 1, math.floor(z1 / T) + 1
    chunks = []
    for cx in range(TX0, TX1, CH):
        for cz in range(TZ0, TZ1, CH):
            t = [cx, cz, min(cx + CH, TX1), min(cz + CH, TZ1)]
            chunks.append({'id': '%d_%d' % (cx, cz), 'tiles': ','.join(map(str, t)), 'bbox': ','.join(map(str, chunk_bbox(t)))})
    bs = boroughs() if reg == 'montreal' else [b for b in boroughs() if b['id'] in ('verdun', 'sud-ouest')]
    json.dump({'region': reg, 'bbox': [W, S, E, N], 'tiles': [TX0, TZ0, TX1, TZ1], 'chunks': chunks, 'boroughs': bs},
              open(os.path.join(outdir, 'plan.json'), 'w'), indent=1)
    print('chunks=' + json.dumps([c['id'] for c in chunks]))
    print('boroughs=' + json.dumps([b['id'] for b in bs]))
    sys.stderr.write('%d chunks, %d height files\n' % (len(chunks), len(bs)))
