"""Overture Maps data for an area, read straight from the public bucket (no account, no key).

  python overture.py index W,S,E,N idx.json      which file pieces (row groups) cover the area, with their boxes
  python overture.py read  idx.json W,S,E,N dir   the features inside a box -> dir/<name>.json (geometry as WKB hex)

Only the file footers are read for the index, and only the matching pieces for a read (HTTP range requests)."""
import sys, re, json, time, urllib.request, urllib.parse, concurrent.futures as cf
import fsspec, pyarrow.parquet as pq, pyarrow.compute as pc

BASE = 'https://overturemaps-us-west-2.s3.us-west-2.amazonaws.com/'
# our name -> (theme, type)
TYPES = {'bld': ('buildings', 'building'), 'bpart': ('buildings', 'building_part'), 'segment': ('transportation', 'segment'), 'land_use': ('base', 'land_use'),
         'water': ('base', 'water'), 'land_cover': ('base', 'land_cover'), 'infrastructure': ('base', 'infrastructure'),
         'land': ('base', 'land')}


def ls(prefix, delim=False):
    keys, pre, tok = [], [], None
    while True:
        u = BASE + '?list-type=2&prefix=' + prefix + ('&delimiter=/' if delim else '') + ('&continuation-token=' + urllib.parse.quote(tok) if tok else '')
        x = urllib.request.urlopen(u, timeout=60).read().decode()
        keys += re.findall(r'<Key>([^<]+)</Key>', x); pre += re.findall(r'<Prefix>([^<]+)</Prefix>', x)
        m = re.search(r'<NextContinuationToken>([^<]+)</NextContinuationToken>', x)
        if not m: return keys, pre
        tok = m.group(1)


def latest_release():
    _, pre = ls('release/', True)
    rel = sorted(p.split('/')[1] for p in pre if re.match(r'release/\d{4}-\d{2}-\d{2}\.\d+/$', p))
    return rel[-1]


def fsys(bs): return fsspec.filesystem('http', client_kwargs={'trust_env': True}, block_size=bs)


def index(bbox, out):
    W, S, E, N = bbox; rel = latest_release(); print('Overture release', rel, flush=True)
    fs = fsys(2 ** 20); res = {'release': rel, 'types': {}}

    def chk(k):
        err = None
        for a in range(4):
            try:
                md = pq.ParquetFile(fs.open(BASE + k, 'rb')).metadata
                cols = {md.schema.column(i).path: i for i in range(md.num_columns)}
                hits = []
                for g in range(md.num_row_groups):
                    rg = md.row_group(g); st = lambda c: rg.column(cols[c]).statistics
                    x0, x1, y0, y1 = st('bbox.xmin').min, st('bbox.xmax').max, st('bbox.ymin').min, st('bbox.ymax').max
                    if x0 <= E and x1 >= W and y0 <= N and y1 >= S: hits.append([g, x0, y0, x1, y1])
                return k, hits
            except Exception as e:
                err = e; time.sleep(2 + 3 * a)
        raise RuntimeError('could not read %s: %s' % (k, err))

    for name, (theme, typ) in TYPES.items():
        keys = [k for k in ls('release/%s/theme=%s/type=%s/' % (rel, theme, typ))[0] if k.endswith('.parquet')]
        r = {}
        with cf.ThreadPoolExecutor(8) as ex:
            for k, h in ex.map(chk, keys):
                if h: r[k] = h
        res['types'][name] = r
        print(name, len(keys), 'files,', sum(len(v) for v in r.values()), 'pieces in the area', flush=True)
    json.dump(res, open(out, 'w'))


def read(idx, bbox, outdir, names=None):
    import os
    W, S, E, N = bbox; os.makedirs(outdir, exist_ok=True); fs = fsys(4 * 2 ** 20)
    for name, files in idx['types'].items():
        if names and name not in names: continue
        rows = []
        for k, gs in files.items():
            gs = [g[0] for g in gs if g[1] <= E and g[3] >= W and g[2] <= N and g[4] >= S]
            if not gs: continue
            for a in range(4):
                try:
                    t = pq.ParquetFile(fs.open(BASE + k, 'rb')).read_row_groups(gs); break
                except Exception as e:
                    if a == 3: raise
                    print('retry', k[-40:], e, flush=True); time.sleep(3 + 5 * a)
            b = t.column('bbox'); f = pc.struct_field
            m = pc.and_(pc.and_(pc.less_equal(f(b, 'xmin'), E), pc.greater_equal(f(b, 'xmax'), W)),
                        pc.and_(pc.less_equal(f(b, 'ymin'), N), pc.greater_equal(f(b, 'ymax'), S)))
            for r in t.filter(m).to_pylist():
                r['geometry'] = r['geometry'].hex() if r.get('geometry') else None
                rows.append(r)
        json.dump(rows, open(os.path.join(outdir, name + '.json'), 'w'), default=str)
        print(name, len(rows), flush=True)


if __name__ == '__main__':
    if sys.argv[1] == 'index': index([float(v) for v in sys.argv[2].split(',')], sys.argv[3])
    else: read(json.load(open(sys.argv[2])), [float(v) for v in sys.argv[3].split(',')], sys.argv[4],
               sys.argv[5].split(',') if len(sys.argv) > 5 else None)
