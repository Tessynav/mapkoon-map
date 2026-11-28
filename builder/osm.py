"""Lane counts and road widths from OpenStreetMap (Overture's roads don't carry them).

  python osm.py W,S,E,N out.json [source.osm.pbf]

Downloads the Québec extract from Geofabrik (unless a file is given), cuts the area with osmium, and keeps for every road
its OpenStreetMap way id -> {ln: lanes, lf/lb: lanes forward/backward, wd: width (m), ow: one-way, tl: lane arrows,
lm: 0 when mapped with no painted lines}; and for every crosswalk way {xc: crossing, xm: crossing:markings, xr: crossing_ref}.
The builder joins these to Overture's roads through each road's OpenStreetMap source id."""
import sys, os, re, json, subprocess, urllib.request, shutil
import osmium

SRC = 'https://download.geofabrik.de/north-america/canada/quebec-latest.osm.pbf'
CLS = {'motorway': 'motorway', 'trunk': 'trunk', 'primary': 'primary', 'secondary': 'secondary', 'tertiary': 'tertiary',
       'unclassified': 'minor', 'residential': 'minor', 'living_street': 'minor', 'service': 'service'}
LINKS = {'motorway_link', 'trunk_link', 'primary_link', 'secondary_link', 'tertiary_link'}
W2 = {'motorway': 11, 'trunk': 10, 'primary': 13, 'secondary': 12, 'tertiary': 11, 'minor': 9.5, 'service': 5.5}
W1 = {'motorway': 10.5, 'trunk': 9.5, 'primary': 8, 'secondary': 7.5, 'tertiary': 6.5, 'minor': 5.5, 'service': 4}
TURN = {'left': 'L', 'sharp_left': 'L', 'slight_left': 'l', 'through': 'T', 'slight_right': 'r', 'right': 'R',
        'sharp_right': 'R', 'reverse': 'U'}


def lead_int(s):
    m = re.match(r'\s*(\d+)', s or ''); return int(m.group(1)) if m else 0


def lead_float(s):
    m = re.match(r'\s*(\d*\.?\d+)', s or ''); return float(m.group(1)) if m else None


def lanes_code(v):
    if not v: return ''
    out = [''.join(TURN[x] for x in lane.strip().split(';') if x.strip() in TURN) for lane in v.split('|')]
    return '|'.join(out) if any(out) else ''


def width(cls, ow, ramp, lanes, t):
    """the same rule as the v2 road builder: a mapped width, else from the lanes, else from the road class"""
    wd = lead_float(t.get('width'))
    if wd and 2 < wd < 40: return wd
    if lanes > 0: return max(3.5 if ow else 6, lanes * 3.4 + (2 if cls == 'minor' and not ow else 0))
    if ramp: return 6
    return W1.get(cls, 6) if ow else W2.get(cls, 8)


def _num(v):
    try: return float(str(v).split(';')[0].replace('m', '').replace(',', '.').strip())
    except Exception: return None


class Bld(osmium.SimpleHandler):
    """buildings and building parts with a height or floors: outline (lon, lat), height, base height, part or not"""
    def __init__(self):
        super().__init__(); self.out = []

    def area(self, a):
        t = a.tags; part = 'building:part' in t and t.get('building:part') != 'no'
        if not part and 'building' not in t: return
        h = _num(t.get('height')); lv = _num(t.get('building:levels'))
        mh = _num(t.get('min_height')); ml = _num(t.get('building:min_level'))
        if not h and not lv: return
        if not h: h = lv * 3.1 + 1.2 + (_num(t.get('roof:levels')) or 0) * 2.5
        if h <= 0 or h > 400: return
        base = mh if mh is not None else (ml * 3.1 if ml else 0)
        try:
            for outer in a.outer_rings():
                ring = [[round(n.lon, 7), round(n.lat, 7)] for n in outer]
                if len(ring) >= 4: self.out.append({'r': ring, 'h': round(h, 1), 'b': round(base, 1), 'p': 1 if part else 0, 'id': ('w' if a.from_way() else 'r') + str(a.orig_id())})
        except Exception:
            pass


class H(osmium.SimpleHandler):
    def __init__(self):
        super().__init__(); self.out = {}

    def way(self, w):
        t = w.tags; hw = t.get('highway')
        # (step 1, Nov 27) crosswalk ways: their crossing type, so only really painted crosswalks are drawn
        if hw in ('footway', 'cycleway', 'path') and (t.get('footway') == 'crossing' or t.get('cycleway') == 'crossing' or t.get('path') == 'crossing'):
            rec = {}
            for k, o in (('crossing', 'xc'), ('crossing:markings', 'xm'), ('crossing_ref', 'xr')):
                if t.get(k): rec[o] = t.get(k)[:20]
            if rec: self.out[w.id] = rec
            return
        if not hw or t.get('area') == 'yes' or (hw not in CLS and hw not in LINKS): return
        ramp = hw in LINKS; cls = CLS[hw[:-5] if ramp else hw]; sv = t.get('service', '')
        ow = 1 if (t.get('oneway') in ('yes', '1', 'true') or t.get('junction') == 'roundabout' or hw == 'motorway') else (-1 if t.get('oneway') in ('-1', 'reverse') else 0)
        ln = lead_int(t.get('lanes'))
        wd = width(cls, ow, ramp, ln, t)
        if hw == 'service' and 'driveway' in sv: wd = 3.5
        elif hw == 'service' and re.search(r'parking_aisle|drive-through', sv): wd = 5.5 if not ow else 3.5
        tl = lanes_code(t.get('turn:lanes')) if ow else ''
        rec = {'ln': ln, 'wd': round(wd, 1), 'ow': ow}
        lf, lb = lead_int(t.get('lanes:forward')), lead_int(t.get('lanes:backward'))
        if lf: rec['lf'] = lf
        if lb: rec['lb'] = lb
        if tl: rec['tl'] = tl
        if t.get('lane_markings') == 'no': rec['lm'] = 0     # (step 1: a street mapped with no painted lines: no centre line)
        self.out[w.id] = rec


if __name__ == '__main__':
    bb, out = sys.argv[1], sys.argv[2]
    src = sys.argv[3] if len(sys.argv) > 3 else None
    work = os.path.dirname(os.path.abspath(out)) or '.'
    if not src:
        src = os.path.join(work, 'source.osm.pbf')
        # (Nov 26: one build failed on "Network is unreachable" for a few seconds; the 4 tries came right after each other)
        # wait longer between tries: 30 s, 1, 2, 4 minutes
        import time
        for a in range(5):
            try:
                with urllib.request.urlopen(SRC, timeout=600) as r, open(src, 'wb') as f: shutil.copyfileobj(r, f, 2 ** 22)
                break
            except Exception as e:
                if a == 4: raise
                print('retry download in', 30 * 2 ** a, 's:', e, flush=True); time.sleep(30 * 2 ** a)
        print('downloaded MB', os.path.getsize(src) // 2 ** 20, flush=True)
    cut = os.path.join(work, 'area.osm.pbf')
    subprocess.run(['osmium', 'extract', '-b', bb, '-s', 'smart', '--overwrite', '-o', cut, src], check=True)
    h = H(); h.apply_file(cut)
    json.dump(h.out, open(out, 'w'), separators=(',', ':'))
    # (Nov 28: Pointe-Nord's towers, 25 to 36 floors, are mapped as building parts with their floors; Overture has the
    # outline but no height, so they came out 4-16 floors) every building and building part with floors or a height
    try:
        B = Bld(); B.apply_file(cut, locations=True)
        json.dump(B.out, open(os.path.join(os.path.dirname(out) or '.', 'osm_bld.json'), 'w'), separators=(',', ':'))
        print('OpenStreetMap buildings and parts with floors or a height', len(B.out), 'parts', sum(1 for b in B.out if b['p']), flush=True)
    except Exception as e:
        print('OpenStreetMap building heights skipped:', e, flush=True)
    print('roads with OpenStreetMap details', sum(1 for v in h.out.values() if 'ln' in v), 'with lanes', sum(1 for v in h.out.values() if v.get('ln')),
          'crosswalks with a type', sum(1 for v in h.out.values() if 'ln' not in v), flush=True)
    for f in (cut,) + (() if len(sys.argv) > 3 else (src,)): os.remove(f)
