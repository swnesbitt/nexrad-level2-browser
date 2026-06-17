"""
NEXRAD Level 2 — 0.5° sweep browser (SAILS-aware)
Hugging Face Space (Gradio)

Pulls one hour of archived NEXRAD Level 2 volumes from the AWS Open Data
buckets, extracts every ~0.5° sweep (including SAILS / MESO-SAILS
re-insertions), and displays them gate-natively: each sweep's polar grid is
shipped to the browser as a lossless grayscale PNG and reprojected per-pixel
in a WebGL fragment shader (4/3-earth beam model), so the native gate
geometry is preserved at any zoom level.
"""

import base64
import bz2
import collections
import datetime as dt
import hashlib
import time as time_mod
import gzip
import html as html_mod
import io
import json
import os
import re
import shutil
import tempfile
import threading
import zipfile

try:
    import shapefile as _pyshp   # pyshp, for live warning polygons
except Exception:                # pragma: no cover
    _pyshp = None
import traceback
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import (ProcessPoolExecutor, ThreadPoolExecutor,
                                as_completed)

import matplotlib

matplotlib.use("Agg")

import cmweather  # noqa: F401  (registers ChaseSpectral & friends)
import cmasher    # noqa: F401  (registers cmr.* colormaps, e.g. cmr.fusion_r)
import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import pyart
import xradar as xd
from PIL import Image

# Fast Rust dealiaser (region_dealias) when available; identical to pyart's
# region-based algorithm. Falls back to pyart if it's not installed or the
# call hits an unsupported path (e.g. ref_vel_field). See
# github.com/swnesbitt/region-dealias
try:
    import region_dealias as _region_dealias  # noqa: F401
    _HAVE_REGION_DEALIAS = True
except Exception:
    _HAVE_REGION_DEALIAS = False

# label shown in the progress bar so it's clear which dealiaser ran
_DEALIAS_ENGINE = "Rust region-dealias" if _HAVE_REGION_DEALIAS else "Py-ART"
print(f"[region-dealias] velocity dealias engine = {_DEALIAS_ENGINE}"
      + (f" v{_region_dealias.__version__}" if _HAVE_REGION_DEALIAS else ""),
      flush=True)


def _dealias_region_based(radar, **kwargs):
    """region_dealias when present, else pyart; pyart fallback on NotImplemented."""
    if _HAVE_REGION_DEALIAS:
        try:
            return _region_dealias.dealias_region_based(radar, **kwargs)
        except NotImplementedError:
            pass
    return pyart.correct.dealias_region_based(radar, **kwargs)
from xradar.io.backends import nexrad_level2 as _nx2

# --- patch: the last LDM record's size field is negative (signed) by spec;
# xradar reads it as unsigned -> MemoryError on most modern volumes.
_orig_init_record = _nx2.NEXRADLevel2File.init_record


def _init_record_fixed(self, recnum):
    try:
        return _orig_init_record(self, recnum)
    except (MemoryError, OverflowError):
        ldm = 0 if recnum < 134 else ((recnum - 134) // 120) + 1
        if ldm >= len(self.bz2_record_indices):
            return False
        start = self.bz2_record_indices[ldm]
        size = abs(int(self._fh[start:start + 4].view(">i4")[0]))
        if self._fp is not None:
            self._fp.seek(start + 4)
            compressed = self._fp.read(size)
        else:
            compressed = self._fh[start + 4:start + 4 + size].tobytes()
        dec = bz2.BZ2Decompressor()
        self._ldm[ldm] = np.frombuffer(dec.decompress(compressed),
                                       dtype=np.uint8)
        return _orig_init_record(self, recnum)


_nx2.NEXRADLevel2File.init_record = _init_record_fixed

# --- patch 2: legacy Message 1 declares range-to-first-gate as signed
# halfwords (they're negative for split cuts, e.g. -375 m), but xradar reads
# them unsigned -> velocity sweeps land ~65 km downrange in pre-2008 data.
_nx2.MSG_1["sur_range_first"] = _nx2.SINT2
_nx2.MSG_1["doppler_range_first"] = _nx2.SINT2

# --- link previews: Gradio's `head=` is injected client-side, so social
# scrapers only see the static defaults baked into its index.html template.
# Patch the template on disk at startup.
APP_TITLE = "NEXRAD level 2 browser"
APP_DESC = ("Gate-native WebGL browsing of archived WSR-88D 0.5° scans "
            "(SAILS-aware) from the AWS NEXRAD Level 2 archive.")
THUMB_URL = ("https://huggingface.co/spaces/snesbitt/nexrad-level2-browser/"
             "resolve/main/thumbnail.png")


def _patch_og_template():
    try:
        import gradio as _g
        p = os.path.join(os.path.dirname(_g.__file__),
                         "templates", "frontend", "index.html")
        html = open(p).read()
        html = html.replace('content="Gradio"', f'content="{APP_TITLE}"')
        html = html.replace("Click to try out the app!", APP_DESC)
        html = html.replace(
            "https://raw.githubusercontent.com/gradio-app/gradio/main/js/"
            "_website/src/lib/assets/img/header-image.jpg", THUMB_URL)
        open(p, "w").write(html)
    except Exception:
        pass  # non-fatal: previews fall back to Gradio defaults


_patch_og_template()

from sites import SITES  # (icao, city, state) for all WSR-88D sites

SITE_CHOICES = [(f"{icao} — {city.lower()}, {st.lower()}", icao)
                for icao, city, st in SITES]

# ----------------------------------------------------------------------------- config

BUCKETS = [
    "https://noaa-nexrad-level2.s3.amazonaws.com",
    "https://unidata-nexrad-level2.s3.amazonaws.com",
]

FIELDS = {
    "Reflectivity": dict(
        pyart="reflectivity", cmap="ChaseSpectral", vmin=-30, vmax=80,
        units="dBZ", label="Horizontal reflectivity factor", tick=10,
    ),
    "Radial velocity": dict(
        pyart="velocity", cmap="cmr.fusion_r", vmin=-40, vmax=40,
        units="m/s", label="Radial velocity", tick=10,
    ),
    "Differential reflectivity": dict(
        pyart="differential_reflectivity", cmap="HomeyerRainbow",
        vmin=-2, vmax=8, units="dB", label="Differential reflectivity", tick=2,
    ),
    "Correlation coefficient": dict(
        pyart="cross_correlation_ratio", cmap="RefDiff",
        vmin=0.5, vmax=1.05, units="", label="Correlation coefficient", tick=0.1,
    ),
}

DEALIAS_NAME = "Radial velocity (dealiased)"
DEALIAS_CFG = dict(
    pyart="velocity", cmap="cmr.fusion_r", vmin=-64, vmax=64,
    units="m/s", label="Dealiased radial velocity", tick=16,
)


def _field_cfg(fn):
    return DEALIAS_CFG if fn == DEALIAS_NAME else FIELDS[fn]


ELEV_MAX = 0.75          # deg — treat sweeps below this as the 0.5° split cut
MAX_FRAMES = 60          # safety cap on rendered sweeps
N_PROC = max(1, min(2, os.cpu_count() or 1))   # decode workers (CPU-bound)

# persistent raw-volume cache (survives across requests; LRU-pruned)
VOL_CACHE_DIR = os.path.join(tempfile.gettempdir(), "nexrad_vol_cache")
os.makedirs(VOL_CACHE_DIR, exist_ok=True)
VOL_CACHE_BYTES = 2 << 30  # 2 GiB
VOL_CACHE_MAX_AGE_S = 24 * 3600  # also evict raw volumes older than 24 h


# ---- live Level 2 chunk feed (Phase 1) --------------------------------------
# Same objects the NewNEXRADLevel2ObjectFilterable SNS topic announces, read
# anonymously: keys are SITE/VOL#/YYYYMMDD-HHMMSS-NNN-{S|I|E}.
CHUNK_BUCKET = "https://unidata-nexrad-level2-chunks.s3.amazonaws.com"
CHUNK_DIR = os.path.join(VOL_CACHE_DIR, "chunks")
os.makedirs(CHUNK_DIR, exist_ok=True)
_CHUNK_STATE = {}          # site -> {vol:int -> meta dict}
_CHUNK_LOCK = threading.Lock()
_S3NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}


def _s3_page(prefix, delimiter=None):
    url = (f"{CHUNK_BUCKET}/?list-type=2&max-keys=1000"
           f"&prefix={urllib.request.quote(prefix)}")
    if delimiter:
        url += f"&delimiter={delimiter}"
    root = ET.fromstring(_http_get(url, timeout=30))
    keys = [k.text for k in root.findall(".//s3:Contents/s3:Key", _S3NS)]
    pres = [p.text for p in
            root.findall(".//s3:CommonPrefixes/s3:Prefix", _S3NS)]
    return keys, pres


_CHUNK_RE = re.compile(r"(\d{8})-(\d{6})-(\d{3})-([SIE])$")


def _chunk_list_vol(site, vol):
    """List one volume dir -> meta(start, keys ordered, complete)."""
    keys, _ = _s3_page(f"{site}/{vol}/")
    parsed = []
    for k in keys:
        m = _CHUNK_RE.search(k)
        if m:
            parsed.append((int(m.group(3)), m.group(4), k))
    if not parsed:
        return None
    parsed.sort()
    m0 = _CHUNK_RE.search(parsed[0][2])
    start = dt.datetime.strptime(m0.group(1) + m0.group(2), "%Y%m%d%H%M%S")
    return dict(vol=vol, start=start,
                keys=[p[2] for p in parsed],
                complete=any(p[1] == "E" for p in parsed))


def _chunk_recent_vols(site, window_min=65):
    """Volumes whose scan started in the trailing window, oldest->newest.
    One dir listing + probes of only the newest (and still-open) volumes."""
    _, pres = _s3_page(f"{site}/", delimiter="/")
    nums = sorted(int(p.split("/")[1]) for p in pres if
                  p.split("/")[1].isdigit())
    if not nums:
        return []
    # handle 0-999 rotation: if the set spans the wrap, low numbers are new
    eff = ([n + 1000 if (max(nums) - min(nums) > 800 and n < 500) else n
            for n in nums])
    order = [n for _, n in sorted(zip(eff, nums), reverse=True)]
    now = dt.datetime.utcnow()
    state = _CHUNK_STATE.setdefault(site, {})
    out = []
    for vol in order[:18]:
        meta = state.get(vol)
        if meta is None or not meta["complete"]:
            meta = _chunk_list_vol(site, vol)
            if meta is None:
                continue
            with _CHUNK_LOCK:
                state[vol] = meta
        age_min = (now - meta["start"]).total_seconds() / 60.0
        if age_min <= window_min:
            out.append(meta)
        elif age_min > window_min + 20:
            break          # everything older is out of window too
    with _CHUNK_LOCK:       # drop state for vols far outside the window
        for vol in [v for v, m in state.items()
                    if (now - m["start"]).total_seconds() > 5400]:
            state.pop(vol, None)
    return sorted(out, key=lambda m: m["start"])


def _sync_chunks(site, meta):
    """Download missing chunks for one volume; returns ordered local paths."""
    vdir = os.path.join(CHUNK_DIR, site, str(meta["vol"]))
    os.makedirs(vdir, exist_ok=True)
    paths = []
    todo = []
    for k in meta["keys"]:
        p = os.path.join(vdir, os.path.basename(k))
        paths.append(p)
        if not os.path.exists(p):
            todo.append((k, p))
    def _dl(job):
        k, p = job
        try:
            data = _http_get(f"{CHUNK_BUCKET}/{urllib.request.quote(k)}",
                             timeout=60)
            tmp = p + ".part"
            with open(tmp, "wb") as f:
                f.write(data)
            os.replace(tmp, p)
        except Exception:
            pass
    if todo:
        with ThreadPoolExecutor(max_workers=8) as ex:
            list(ex.map(_dl, todo))
    return [p for p in paths if os.path.exists(p)]


def fetch_live_chunks(site, window_min=65, progress=None):
    """Phase-1 API: trailing-window volumes as ordered local chunk paths.
    Returns [dict(vol, start, complete, paths)] oldest->newest."""
    vols = _chunk_recent_vols(site, window_min)
    out = []
    for i, meta in enumerate(vols):
        if progress:
            progress(0.05 + 0.9 * (i + 1) / max(1, len(vols)),
                     f"Syncing live chunks {i + 1}/{len(vols)} "
                     f"(vol {meta['vol']})…")
        paths = _sync_chunks(site, meta)
        if paths:
            out.append(dict(vol=meta["vol"], start=meta["start"],
                            complete=meta["complete"], paths=paths))
    return out


def _prune_chunk_cache(max_age_s=7200):
    try:
        now = time_mod.time()
        for site_d in os.listdir(CHUNK_DIR):
            sdir = os.path.join(CHUNK_DIR, site_d)
            for vol_d in os.listdir(sdir):
                vdir = os.path.join(sdir, vol_d)
                if now - os.path.getmtime(vdir) > max_age_s:
                    shutil.rmtree(vdir, ignore_errors=True)
    except Exception:
        pass


CHUNK_SITES = {"KILX"}        # sites on the low-latency chunk feed
_CHUNK_FRAMES = {}            # (site, vol) -> dict(n, complete, frames)
_CHUNK_DEAL = {}              # (site, vol) -> dealiased frames (complete vols)


def _concat_chunks(paths, out_path):
    """S+I+...+E chunks concatenated form a valid archive volume file."""
    if not os.path.exists(out_path):
        tmp = out_path + ".part"
        with open(tmp, "wb") as w:
            for p in paths:
                with open(p, "rb") as r:
                    shutil.copyfileobj(r, w)
        os.replace(tmp, out_path)
    return out_path


def dealias_volume_file(path, vol_label):
    """Py-ART region-based dealiasing of one local archive-format file."""
    frames = []
    try:
        radar = pyart.io.read_nexrad_archive(path)
    except Exception:
        return frames
    fixed = radar.fixed_angle["data"]

    def _has(sweep, f):
        if f not in radar.fields:
            return False
        return not np.all(np.ma.getmaskarray(radar.get_field(sweep, f)))

    cands = [s for s in range(radar.nsweeps)
             if fixed[s] < ELEV_MAX and _has(s, "velocity")]
    if not cands:
        return frames
    try:
        nyq = radar.instrument_parameters["nyquist_velocity"]["data"]
        meta = []
        for s in cands:
            i0, i1 = radar.get_start_end(s)
            meta.append(dict(s=s, t=sweep_datetime(radar, s),
                             nyq=float(np.median(nyq[i0:i1 + 1]))))
        cands = [m["s"] for m in _dedup_doppler(meta, lambda m: m["nyq"])]
    except Exception:
        pass
    try:
        sub = radar.extract_sweeps(cands)
        corr = _dealias_region_based(
            sub, vel_field="velocity", keep_original=False)
        sub.fields["velocity"]["data"] = corr["data"]
        polar_frame._vol = vol_label
        _vcp = radar.metadata.get("vcp_pattern")
        polar_frame._vcp = f"VCP-{_vcp}" if _vcp else ""
        for s in range(sub.nsweeps):
            fr = polar_frame(sub, s, DEALIAS_CFG)
            if fr is not None:
                frames.append(fr)
    except Exception:
        pass
    del radar
    return frames


# ---- live warnings (real-time mode) ----------------------------------------
WARN_SRC = ("https://mesonet.agron.iastate.edu/data/gis/shape/4326/us/"
            "current_ww.zip")
WARN_JSON = os.path.join(VOL_CACHE_DIR, "warnings.json")
WARN_URL = "/gradio_api/file=" + WARN_JSON

# city label database (GeoNames-derived, shipped with the repo) is served
# from the static cache dir alongside warnings.json
CITIES_SRC = os.path.join(os.path.dirname(__file__), "cities.json")
CITIES_JSON = os.path.join(VOL_CACHE_DIR, "cities.json")
try:
    if os.path.exists(CITIES_SRC):
        shutil.copyfile(CITIES_SRC, CITIES_JSON)
except Exception:
    pass
CITIES_URL = "/gradio_api/file=" + CITIES_JSON


def _fetch_warnings():
    """Pull IEM current watches/warnings, keep storm-based TOR/SVR warnings,
    write a GeoJSON FeatureCollection next to the volume cache."""
    if _pyshp is None:
        return
    buf = _http_get(WARN_SRC, timeout=30)
    z = zipfile.ZipFile(io.BytesIO(buf))
    base = [n for n in z.namelist() if n.endswith(".shp")][0][:-4]
    rdr = _pyshp.Reader(shp=io.BytesIO(z.read(base + ".shp")),
                        dbf=io.BytesIO(z.read(base + ".dbf")),
                        shx=io.BytesIO(z.read(base + ".shx")))
    now = dt.datetime.utcnow()
    feats = []
    for sr in rdr.iterShapeRecords():
        d = sr.record.as_dict()
        # TO = tornado, SV = severe t'storm, FF = flash flood — all storm-based
        # polygon warnings (GTYPE=P). (Flood Warning FL.W is county-based.)
        if not (d.get("SIG") == "W" and d.get("PHENOM") in ("TO", "SV", "FF")
                and d.get("GTYPE") == "P"):
            continue
        # drop warnings already past their expiration (UTC YYYYMMDDHHMM) so
        # they vanish from the loop instead of lingering until IEM republishes
        exp = str(d.get("EXPIRED") or "")
        try:
            if len(exp) >= 12 and \
                    dt.datetime.strptime(exp[:12], "%Y%m%d%H%M") <= now:
                continue
        except Exception:
            pass
        feats.append(dict(
            type="Feature",
            properties=dict(ph=d.get("PHENOM"), wfo=d.get("WFO"),
                            etn=d.get("ETN"), exp=exp),
            geometry=sr.shape.__geo_interface__))
    feats.sort(key=lambda f: (str(f["properties"]["wfo"] or ""),
                              f["properties"]["etn"] or 0))
    # etag from the (expiry-filtered) content, NOT the raw zip — so the client
    # redraws and clears a warning the moment it drops out of the valid set
    etag = hashlib.md5(
        json.dumps(feats, sort_keys=True).encode()).hexdigest()[:12]
    out = dict(type="FeatureCollection",
               etag=etag,
               updated=now.isoformat() + "Z",
               features=feats)
    tmp = WARN_JSON + ".part"
    with open(tmp, "w") as f:
        json.dump(out, f)
    os.replace(tmp, WARN_JSON)


def _warn_poller():
    while True:
        try:
            _fetch_warnings()
        except Exception:
            pass
        time_mod.sleep(60)


# small static assets that live in VOL_CACHE_DIR but must never be LRU-pruned
# (cities.json is copied once at startup and never re-touched, so without this
# guard it sorts oldest and gets evicted as radar volumes fill the 2 GiB cap)
_CACHE_KEEP = {"cities.json", "warnings.json"}


def _prune_vol_cache():
    try:
        files = [(os.path.getmtime(p), os.path.getsize(p), p)
                 for p in (os.path.join(VOL_CACHE_DIR, f)
                           for f in os.listdir(VOL_CACHE_DIR))
                 if os.path.basename(p) not in _CACHE_KEEP
                 and os.path.isfile(p)]
        files.sort(reverse=True)
        now = time_mod.time()
        total = 0
        for mt, sz, p in files:
            total += sz
            if total > VOL_CACHE_BYTES or (now - mt) > VOL_CACHE_MAX_AGE_S:
                os.remove(p)
    except Exception:
        pass

YEARS = [str(y) for y in range(1991, dt.datetime.utcnow().year + 1)][::-1]
MONTHS = [f"{m:02d}" for m in range(1, 13)]
DAYS = [f"{d:02d}" for d in range(1, 32)]
HOURS = [f"{h:02d}:00" for h in range(24)]

# ----------------------------------------------------------------------------- s3 access


def _http_get(url, timeout=60):
    req = urllib.request.Request(url, headers={"User-Agent": "nexrad-browser/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def list_hour_keys(site, date, hour):
    """Return (bucket, [keys]) for volumes starting in the given UTC hour."""
    prefix = f"{date:%Y/%m/%d}/{site}/{site}{date:%Y%m%d}_{hour:02d}"
    last_err = None
    any_ok = False
    for bucket in BUCKETS:
        url = f"{bucket}/?list-type=2&prefix={urllib.request.quote(prefix)}&max-keys=200"
        try:
            xml = _http_get(url, timeout=30)
        except Exception as e:  # 403 / network — try next bucket
            last_err = e
            continue
        ns = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
        root = ET.fromstring(xml)
        any_ok = True
        keys = [k.text for k in root.findall(".//s3:Contents/s3:Key", ns)]
        keys = [k for k in keys if not k.endswith("_MDM") and "NXL2" not in k]
        if keys:
            return bucket, sorted(keys)
    if last_err and not any_ok:
        raise RuntimeError(f"Could not list archive buckets: {last_err}")
    return None, []


def download_volume(bucket, key, dest_dir):
    path = os.path.join(dest_dir, os.path.basename(key))
    if not os.path.exists(path):
        data = _http_get(f"{bucket}/{urllib.request.quote(key)}", timeout=120)
        with open(path, "wb") as f:
            f.write(data)
    return path


def _safe_download(bucket, key, dest_dir):
    try:
        download_volume(bucket, key, dest_dir)
    except Exception:
        pass


# ----------------------------------------------------------------------------- radar processing


def sweep_datetime(radar, sweep):
    units = radar.time["units"]  # "seconds since YYYY-MM-DDTHH:MM:SSZ"
    m = re.search(r"since\s+([0-9T:\-]+)", units)
    base = dt.datetime.fromisoformat(m.group(1).rstrip("Z"))
    s = radar.sweep_start_ray_index["data"][sweep]
    return base + dt.timedelta(seconds=float(radar.time["data"][s]))


def _scale_u8(data, mask, field_cfg):
    """float field -> uint8 (0 = no data, 1..255 spans vmin..vmax)."""
    scaled = np.clip(
        (data - field_cfg["vmin"]) / (field_cfg["vmax"] - field_cfg["vmin"]), 0, 1)
    vals = (scaled * 254 + 1).astype(np.uint8)
    vals[mask] = 0
    return vals


# gates whose reflectivity is below this (or with no reflectivity return at
# all) are treated as "no return" and rendered fully transparent — applied to
# every field, so V/ZDR/CC noise outside echo disappears too
Z_MIN_DBZ = -30.0


def _noreturn_mask(z):
    """True where there is no usable return (Z missing or < Z_MIN_DBZ)."""
    zf = np.ma.filled(np.ma.masked_invalid(z), -1e3)
    return ~(zf >= Z_MIN_DBZ)


def _regrid_az(az, vals):
    """Gather rays onto a uniform azimuth grid (every bin takes its nearest
    ray, so no empty spokes). Returns (grid, naz)."""
    daz = np.median(np.abs(np.diff(np.unwrap(np.radians(az))))) * 180 / np.pi
    naz = 720 if daz < 0.75 else 360
    order = np.argsort(az)
    az_s = az[order]
    centers = (np.arange(naz) + 0.5) * (360.0 / naz)
    idx = np.searchsorted(az_s, centers)
    lo = (idx - 1) % len(az_s)
    hi = idx % len(az_s)
    d_lo = np.abs((centers - az_s[lo] + 180) % 360 - 180)
    d_hi = np.abs((centers - az_s[hi] + 180) % 360 - 180)
    nearest = np.where(d_lo <= d_hi, lo, hi)
    return vals[order[nearest]], naz


def polar_frame(radar, sweep, field_cfg):
    """Pack one sweep onto a uniform azimuth grid -> gate-native frame dict.

    The uint8 grid (rows=azimuth bins, cols=gates) is PNG-compressed
    losslessly; 0 = no data, 1..255 spans [vmin, vmax].
    """
    fname = field_cfg["pyart"]
    data = radar.get_field(sweep, fname)
    mask = np.ma.getmaskarray(data)
    if mask.all():
        return None

    s_idx = radar.sweep_start_ray_index["data"][sweep]
    e_idx = radar.sweep_end_ray_index["data"][sweep]
    az = radar.azimuth["data"][s_idx:e_idx + 1]

    rng = radar.range["data"]
    dr = float(rng[1] - rng[0])
    r0 = float(rng[0]) - dr / 2.0          # edge of first gate
    ngates = data.shape[1]

    try:  # censor gates with no usable return (same-sweep reflectivity);
        # skip when this sweep has no Z at all (legacy Doppler cuts)
        z = radar.get_field(sweep, "reflectivity")
        if not np.all(np.ma.getmaskarray(z)):
            mask = mask | _noreturn_mask(z)
    except Exception:
        pass
    vals = _scale_u8(np.ma.filled(data, field_cfg["vmin"]), mask, field_cfg)
    grid, naz = _regrid_az(az, vals)

    buf = io.BytesIO()
    Image.fromarray(grid, mode="L").save(buf, format="PNG")

    t = sweep_datetime(radar, sweep)
    el = float(radar.fixed_angle["data"][sweep])
    vcp = getattr(polar_frame, "_vcp", "")
    vcp_part = f"{vcp}  •  " if vcp else ""
    return dict(
        img=base64.b64encode(buf.getvalue()).decode(),
        naz=naz, ngates=ngates, r0=r0, dr=dr, el=el,
        maxr=r0 + ngates * dr, time=t,
        label=(f"{t:%Y-%m-%d %H:%M:%S}Z  •  {el:.1f}°  •  {vcp_part}"
               f"sweep {sweep}  •  {polar_frame._vol}"),
    )


def colorbar_cfg(field_cfg, n=256):
    """Colormap stops + range; drawn client-side and used as the WebGL LUT."""
    cm = plt.get_cmap(field_cfg["cmap"])
    stops = [matplotlib.colors.to_hex(cm(i / (n - 1))) for i in range(n)]
    unit = f" ({field_cfg['units']})" if field_cfg["units"] else ""
    return dict(stops=stops, vmin=field_cfg["vmin"], vmax=field_cfg["vmax"],
                tick=field_cfg["tick"], units=field_cfg["units"],
                label=f"{field_cfg['label']}{unit}")


XR_FIELD = {"reflectivity": "DBZH", "velocity": "VRADH",
            "differential_reflectivity": "ZDR",
            "cross_correlation_ratio": "RHOHV"}


def _gunzip(path):
    if not path.endswith(".gz"):
        return path
    out = path[:-3]
    if not os.path.exists(out):
        with gzip.open(path, "rb") as fi, open(out, "wb") as fo:
            shutil.copyfileobj(fi, fo)
    return out


def _np_dt(t64):
    return dt.datetime.utcfromtimestamp(
        int(np.datetime64(t64, "ns").astype("int64")) / 1e9)


def polar_frame_xr(ds, fvar, field_cfg, el, sweep_name, vol, vcp=""):
    data = ds[fvar].values
    finite = np.isfinite(data)
    if not finite.any():
        return None
    az = ds["azimuth"].values
    rng = ds["range"].values
    dr = float(rng[1] - rng[0])
    r0 = float(rng[0]) - dr / 2.0
    ngates = data.shape[1]

    nodata = ~finite
    rsv = _reserved_mask(ds, fvar, data)
    if rsv is not None:  # below-threshold / range-folded codes
        nodata = nodata | rsv
    if "DBZH" in ds:  # censor gates with no usable return
        zdat = ds["DBZH"].values
        zbad = _noreturn_mask(zdat)
        zrsv = _reserved_mask(ds, "DBZH", zdat)
        if zrsv is not None:
            zbad = zbad | zrsv
        nodata = nodata | zbad
    vals = _scale_u8(np.nan_to_num(data, nan=field_cfg["vmin"]),
                     nodata, field_cfg)
    grid, naz = _regrid_az(az, vals)

    buf = io.BytesIO()
    Image.fromarray(grid, mode="L").save(buf, format="PNG")
    t = _np_dt(ds["time"].values.min())
    vcp_part = f"{vcp}  •  " if vcp else ""
    return dict(
        img=base64.b64encode(buf.getvalue()).decode(),
        naz=naz, ngates=ngates, r0=r0, dr=dr, el=el,
        maxr=r0 + ngates * dr, time=t,
        label=(f"{t:%Y-%m-%d %H:%M:%S}Z  •  {el:.1f}°  •  {vcp_part}"
               f"{sweep_name}  •  {vol}"),
    )


def _reserved_mask(ds, fvar, data):
    """NEXRAD moment codes 0 (below threshold) and 1 (range folded) are
    reserved, but xradar decodes them as physical values (no _FillValue in
    its attrs). Mask anything at/below code 1 using the CF encoding."""
    enc = getattr(ds[fvar], "encoding", {}) or {}
    sf, ao = enc.get("scale_factor"), enc.get("add_offset")
    if sf is None or ao is None:
        return None
    return data <= (ao + 1.5 * abs(sf))


def _dedup_doppler(cands, nyq_of, window_s=90):
    """Collapse MPDA-style multi-PRF bursts: within a cluster of Doppler
    cuts closer than window_s, keep only the highest-Nyquist cut — but only
    when the Nyquists actually differ (>2 m/s spread). Equal-Nyquist cuts
    are SAILS/MRLE revisits and are all kept."""
    out = []
    cluster = []

    def flush():
        if not cluster:
            return
        nv = [nyq_of(s) for s in cluster]
        if max(nv) - min(nv) > 2.0:
            out.append(cluster[int(np.argmax(nv))])
        else:
            out.extend(cluster)

    for s in cands:  # cands are in scan order
        if cluster and (s["t"] - cluster[0]["t"]).total_seconds() > window_s:
            flush()
            cluster = []
        cluster.append(s)
    flush()
    return out


def _process_xradar(path, key, cfgs):
    """Read once with xradar (swnesbitt fork); render every requested field.
    Returns ({field_name: [frames]}, site_ll)."""
    dtree = xd.io.open_nexradlevel2_datatree(_gunzip(path))
    return _tree_to_frames(dtree, os.path.basename(key), cfgs)


def process_chunks(paths, vol_label, cfgs):
    """Decode a live chunk list (S/I/E pieces of one volume) — the fork
    assembles partial volumes; incomplete trailing sweeps are dropped."""
    dtree = xd.io.open_nexradlevel2_datatree(list(paths),
                                             incomplete_sweep="drop")
    return _tree_to_frames(dtree, vol_label, cfgs)


def _tree_to_frames(dtree, vol, cfgs):
    root = dtree.ds
    site_ll = (float(root["latitude"].values), float(root["longitude"].values))
    vcp = str(root.attrs.get("scan_name", "") or "")
    dyn = str(root.attrs.get("dynamic_scan_type", "") or "")
    if dyn and dyn.lower() not in ("none", "false", "", "standard"):
        vcp = f"{vcp} ({dyn})" if vcp else dyn

    sweeps = []
    for name in sorted((c for c in dtree.children if c.startswith("sweep_")),
                       key=lambda n: int(n.split("_")[1])):
        ds = dtree[name].ds
        el = float(ds["sweep_fixed_angle"].values)
        if el >= ELEV_MAX:
            continue
        has_v = "VRADH" in ds and bool(np.isfinite(ds["VRADH"].values).any())
        sweeps.append(dict(name=name, ds=ds, el=el, has_v=has_v,
                           t=_np_dt(ds["time"].values.min())))

    out = {}
    for fname, cfg in cfgs.items():
        fvar = XR_FIELD[cfg["pyart"]]
        cands = [s for s in sweeps
                 if fvar in s["ds"]
                 and bool(np.isfinite(s["ds"][fvar].values).any())]
        if cfg["pyart"] != "velocity":
            # surveillance (long-range) cuts only — never the shorter-range
            # Doppler split cuts; fall back for merged-cut VCPs
            surv = [s for s in cands if not s["has_v"]]
            cands = surv or cands
        else:
            # MPDA (VCP 121) runs several near-simultaneous Doppler cuts at
            # different PRFs: keep only the highest-Nyquist member of each
            # cluster. With reserved codes masked, max |V| == the Nyquist.
            def _est_nyq(s):
                v = s["ds"][fvar].values
                rsv = _reserved_mask(s["ds"], fvar, v)
                if rsv is not None:
                    v = np.where(rsv, np.nan, v)
                return float(np.nanmax(np.abs(v)))
            cands = _dedup_doppler(cands, _est_nyq)
        frames = []
        for s in cands:
            try:
                fr = polar_frame_xr(s["ds"], fvar, cfg, s["el"],
                                    s["name"], vol, vcp)
            except Exception:
                continue
            if fr is not None:
                frames.append(fr)
        out[fname] = frames
    del dtree
    return out, site_ll


def dealias_volume(bucket, key, dest_dir):
    """Region-based dealiasing of the volume's 0.5° Doppler cuts (Py-ART).
    Returns a list of frames for the dealiased-velocity field."""
    frames = []
    try:
        path = download_volume(bucket, key, dest_dir)
        radar = pyart.io.read_nexrad_archive(path)
    except Exception:
        return frames
    fixed = radar.fixed_angle["data"]

    def _has(sweep, f):
        if f not in radar.fields:
            return False
        return not np.all(np.ma.getmaskarray(radar.get_field(sweep, f)))

    cands = [s for s in range(radar.nsweeps)
             if fixed[s] < ELEV_MAX and _has(s, "velocity")]
    if not cands:
        return frames
    try:  # MPDA: keep highest-Nyquist cut per cluster
        nyq = radar.instrument_parameters["nyquist_velocity"]["data"]
        meta = []
        for s in cands:
            i0, i1 = radar.get_start_end(s)
            meta.append(dict(s=s, t=sweep_datetime(radar, s),
                             nyq=float(np.median(nyq[i0:i1 + 1]))))
        cands = [m["s"] for m in _dedup_doppler(meta, lambda m: m["nyq"])]
    except Exception:
        pass
    try:
        sub = radar.extract_sweeps(cands)
        corr = _dealias_region_based(
            sub, vel_field="velocity", keep_original=False)
        sub.fields["velocity"]["data"] = corr["data"]
        polar_frame._vol = os.path.basename(key)
        _vcp = radar.metadata.get("vcp_pattern")
        polar_frame._vcp = f"VCP-{_vcp}" if _vcp else ""
        for s in range(sub.nsweeps):
            fr = polar_frame(sub, s, DEALIAS_CFG)
            if fr is not None:
                frames.append(fr)
    except Exception:
        pass
    del radar
    return frames


def process_volume(bucket, key, cfgs, dest_dir):
    """Download + read one volume (xradar fork first, Py-ART fallback).
    Decodes every requested field in a single read; raw files stay cached."""
    empty = {fn: [] for fn in cfgs}
    try:
        path = download_volume(bucket, key, dest_dir)
    except Exception:
        return empty, None
    try:
        return _process_xradar(path, key, cfgs)
    except Exception:
        pass  # fall back to Py-ART below
    try:
        return _process_pyart(path, key, cfgs)
    except Exception:
        return empty, None


def _process_pyart(path, key, cfgs):
    """Fallback reader via Py-ART."""
    out = {fn: [] for fn in cfgs}
    try:
        radar = pyart.io.read_nexrad_archive(path, delay_field_loading=True)
    except Exception:
        return out, None
    site_ll = (float(radar.latitude["data"][0]),
               float(radar.longitude["data"][0]))
    fixed = radar.fixed_angle["data"]
    polar_frame._vol = os.path.basename(key)
    _vcp = radar.metadata.get("vcp_pattern")
    polar_frame._vcp = f"VCP-{_vcp}" if _vcp else ""

    def _has(sweep, f):
        if f not in radar.fields:
            return False
        return not np.all(np.ma.getmaskarray(radar.get_field(sweep, f)))

    lows = [s for s in range(radar.nsweeps) if fixed[s] < ELEV_MAX]
    for fname, cfg in cfgs.items():
        if cfg["pyart"] not in radar.fields:
            continue
        cands = [s for s in lows if _has(s, cfg["pyart"])]
        if cfg["pyart"] != "velocity":
            surv = [s for s in cands if not _has(s, "velocity")]
            cands = surv or cands
        else:
            try:
                nyq = radar.instrument_parameters["nyquist_velocity"]["data"]
                meta = []
                for s in cands:
                    i0, i1 = radar.get_start_end(s)
                    meta.append(dict(s=s, t=sweep_datetime(radar, s),
                                     nyq=float(np.median(nyq[i0:i1 + 1]))))
                cands = [m["s"] for m in
                         _dedup_doppler(meta, lambda m: m["nyq"])]
            except Exception:
                pass
        for sweep in cands:
            try:
                fr = polar_frame(radar, sweep, cfg)
            except Exception:
                continue
            if fr is not None:
                out[fname].append(fr)
    del radar
    return out, site_ll


# ----------------------------------------------------------------------------- leaflet + webgl page

LEAFLET_PAGE = """<!DOCTYPE html>
<html><head>
<meta charset="utf-8"/>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
 @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@600;700&family=Source+Sans+3:wght@400;600;700&display=swap');
 :root{--illini-orange:#FF5F05;--illini-blue:#13294B;--illini-blue-2:#1D3866;
       --illini-blue-3:#25457F;--storm-light:#C8C6C7}
 html,body{margin:0;height:100%;background:var(--illini-blue)}
 #map{position:absolute;inset:0 0 86px 0}
 #glcv{position:absolute;inset:0;pointer-events:none;z-index:400}
 #bar{position:absolute;left:0;right:0;bottom:0;height:86px;
      background:var(--illini-blue);border-top:4px solid var(--illini-orange);
      color:#fff;font:13px 'Source Sans 3','Source Sans Pro',system-ui,sans-serif;
      display:flex;flex-direction:column;justify-content:center;gap:6px;
      padding:6px 14px;box-sizing:border-box}
 #row1{display:flex;align-items:center;gap:10px}
 #slider{flex:1;accent-color:var(--illini-orange)}
 #op{accent-color:var(--illini-orange);width:90px}
 #label{font-variant-numeric:tabular-nums;white-space:nowrap;overflow:hidden;
        text-overflow:ellipsis;color:var(--storm-light)}
 button{background:var(--illini-orange);color:#fff;border:0;border-radius:4px;
        padding:4px 12px;cursor:pointer;font-size:13px;
        font-family:'Montserrat',sans-serif;font-weight:700}
 button:hover{background:#E25504}
 a{color:#FF8136}
 #share{position:absolute;top:80px;left:10px;z-index:1000;width:34px;height:34px;
        display:flex;align-items:center;justify-content:center;cursor:pointer;
        background:var(--illini-blue-2);color:#fff;border:1px solid var(--illini-blue-3);
        border-radius:4px;box-shadow:0 1px 5px rgba(0,0,0,.4)}
 #share:hover{background:var(--illini-orange)}
 #toast{position:absolute;top:88px;left:52px;z-index:1000;display:none;
        background:var(--illini-blue);border:1px solid var(--illini-blue-3);
        color:#fff;font:12px 'Source Sans 3',sans-serif;padding:4px 10px;
        border-radius:4px}
 #ovmenu{position:absolute;top:10px;right:10px;z-index:1000;
       background:rgba(19,41,75,.92);border:1px solid var(--illini-blue-3);
       border-top:4px solid var(--illini-orange);border-radius:4px;
       padding:10px;color:#fff;font:12px 'Source Sans 3',sans-serif;
       display:flex;flex-direction:column;gap:8px;width:118px}
 #ovmenu b{font-family:'Montserrat',sans-serif;font-size:10px;
       letter-spacing:.06em;text-transform:uppercase;opacity:.85}
 #ovmenu label{display:flex;gap:6px;align-items:flex-start;cursor:pointer;
       line-height:1.25}
 #ovmenu input{accent-color:var(--illini-orange);margin-top:1px}
 #cbar{position:absolute;top:50%;right:10px;transform:translateY(-50%);z-index:1000;
       background:rgba(19,41,75,.92);border:1px solid var(--illini-blue-3);
       border-top:4px solid var(--illini-orange);border-radius:4px;
       padding:10px 8px;color:#fff;
       font:10px 'Source Sans 3',system-ui,sans-serif;
       display:flex;gap:6px;align-items:stretch}
 #cblabel{writing-mode:vertical-rl;transform:rotate(180deg);text-align:center;
          font-size:11px;opacity:.9}
 #cbticks{position:relative;width:34px;margin:0}
 #cbticks div{position:absolute;left:0;transform:translateY(-50%);
              font-variant-numeric:tabular-nums}
</style></head><body>
<div id="map"></div>
<div id="share" title="Copy share link">
 <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor"
      stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
   <circle cx="18" cy="5" r="3"/><circle cx="6" cy="12" r="3"/>
   <circle cx="18" cy="19" r="3"/>
   <line x1="8.6" y1="10.5" x2="15.4" y2="6.5"/>
   <line x1="8.6" y1="13.5" x2="15.4" y2="17.5"/>
 </svg>
</div>
<div id="toast">Link copied</div>
<div id="cbar"><div id="cblabel"></div>
  <canvas id="cbcanvas" width="16" height="320"></canvas>
  <div id="cbticks"></div></div>
<div id="ovmenu"><b>Overlays</b>
  <label><input type="checkbox" id="ck-counties" checked/><span>County boundaries</span></label>
  <label><input type="checkbox" id="ck-interstates" checked/><span>Highways</span></label>
</div>
<div id="bar">
 <div id="row1">
   <button id="play">&#9654;</button>
   <input id="slider" type="range" min="0" max="__MAXIDX__" value="0" step="1"/>
   <span style="opacity:.7">opacity</span>
   <input id="op" type="range" min="10" max="100" value="100"/>
 </div>
 <div id="label"></div>
</div>
<script>
const frames = __FRAMES__;
const cb = __CBAR__;
const SITE = [__SLAT__, __SLON__];          // deg
const SHARE_URL = "__SHARE__";

// ---------------------------------------------------------------- colorbar
(function(){
  const cv = document.getElementById('cbcanvas'), ctx = cv.getContext('2d');
  const g = ctx.createLinearGradient(0, cv.height, 0, 0);
  cb.stops.forEach((c, i) => g.addColorStop(i / (cb.stops.length - 1), c));
  ctx.fillStyle = g; ctx.fillRect(0, 0, cv.width, cv.height);
  document.getElementById('cblabel').textContent = cb.label;
  const ticks = document.getElementById('cbticks');
  const t0 = Math.ceil(cb.vmin / cb.tick) * cb.tick;
  for (let v = t0; v <= cb.vmax + 1e-9; v += cb.tick) {
    const val = +v.toFixed(6);
    const frac = (val - cb.vmin) / (cb.vmax - cb.vmin);
    const d = document.createElement('div');
    d.style.top = (100 - frac * 100) + '%';
    d.textContent = '' + (cb.tick < 1 ? val.toFixed(1) : Math.round(val));
    ticks.appendChild(d);
  }
})();

// ---------------------------------------------------------------- leaflet
const map = L.map('map', {zoomControl: true}).setView(SITE, 8);
L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png',
  {attribution: '&copy; OpenStreetMap &copy; CARTO', subdomains: 'abcd',
   maxZoom: 14}).addTo(map);
L.circleMarker(SITE, {radius: 5, color: '#fff', weight: 2, fillColor: '#e33',
  fillOpacity: 1}).addTo(map).bindTooltip('__SITE__');
// white ring at maximum unambiguous range of the current sweep
const ring = L.circle(SITE, {radius: frames.length ? frames[0].maxr : 300000,
  color: '#fff', weight: 1.5, opacity: 0.9, fill: false,
  interactive: false}).addTo(map);

// fit to max range of frame 0
(function(){
  const mr = frames.length ? frames[0].maxr : 300000;
  const dLat = mr / 111320, dLon = mr / (111320 * Math.cos(SITE[0] * Math.PI / 180));
  map.fitBounds([[SITE[0] - dLat, SITE[1] - dLon], [SITE[0] + dLat, SITE[1] + dLon]],
                {padding: [8, 8]});
})();

// ---------------------------------------------------------------- webgl layer
const glcv = document.createElement('canvas');
glcv.id = 'glcv';
document.getElementById('map').appendChild(glcv);
const gl = glcv.getContext('webgl', {premultipliedAlpha: false, alpha: true});

const VS = `
attribute vec2 aPos;
varying vec2 vUV;
void main(){ vUV = aPos * 0.5 + 0.5; gl_Position = vec4(aPos, 0., 1.); }`;

const FS = `
precision highp float;
varying vec2 vUV;
uniform sampler2D uData, uCmap;
uniform vec2 uMercMin, uMercMax;     // SW / NE of canvas in mercator meters
uniform vec2 uSite;                  // lon0, lat0 (radians)
uniform float uEl, uR0, uDr, uNgates, uNaz, uMaxR, uOpacity;
const float R = 6378137.0;
const float RE = 6371000.0;
const float K = 1.3333333;
const float PI = 3.141592653589793;

void main(){
  vec2 merc = mix(uMercMin, uMercMax, vUV);
  float lat = 2.0 * atan(exp(merc.y / R)) - PI / 2.0;
  float lon = merc.x / R;
  float dLon = lon - uSite.x;
  float lat0 = uSite.y;
  // great-circle ground distance (haversine) and azimuth from site
  float sdl = sin(dLon * 0.5), sdp = sin((lat - lat0) * 0.5);
  float a = sdp * sdp + cos(lat0) * cos(lat) * sdl * sdl;
  float s = 2.0 * RE * asin(min(1.0, sqrt(a)));
  if (s > uMaxR * 1.05) discard;
  float az = atan(sin(dLon) * cos(lat),
                  cos(lat0) * sin(lat) - sin(lat0) * cos(lat) * cos(dLon));
  az = mod(degrees(az) + 360.0, 360.0);
  // invert 4/3-earth beam model: find slant range r whose ground arc = s
  float Rk = K * RE;
  float r = s;
  for (int i = 0; i < 3; i++) {
    float h = sqrt(r * r + Rk * Rk + 2.0 * r * Rk * sin(uEl)) - Rk;
    float sg = Rk * asin(clamp(r * cos(uEl) / (Rk + h), -1.0, 1.0));
    r = r * (1.0 + (s - sg) / max(sg, 1.0));
  }
  float gate = (r - uR0) / uDr;
  if (gate < 0.0 || gate >= uNgates) discard;
  vec2 tc = vec2((floor(gate) + 0.5) / uNgates, (az / 360.0));
  float v = texture2D(uData, tc).r;
  if (v < 0.002) discard;                       // 0 = no data
  vec3 col = texture2D(uCmap, vec2((v * 255.0 - 1.0) / 254.0, 0.5)).rgb;
  gl_FragColor = vec4(col, uOpacity);
}`;

function shader(type, src){
  const sh = gl.createShader(type);
  gl.shaderSource(sh, src); gl.compileShader(sh);
  if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS))
    throw new Error(gl.getShaderInfoLog(sh));
  return sh;
}
const prog = gl.createProgram();
gl.attachShader(prog, shader(gl.VERTEX_SHADER, VS));
gl.attachShader(prog, shader(gl.FRAGMENT_SHADER, FS));
gl.linkProgram(prog); gl.useProgram(prog);
gl.bindBuffer(gl.ARRAY_BUFFER, gl.createBuffer());
gl.bufferData(gl.ARRAY_BUFFER,
  new Float32Array([-1,-1, 1,-1, -1,1, 1,1]), gl.STATIC_DRAW);
const aPos = gl.getAttribLocation(prog, 'aPos');
gl.enableVertexAttribArray(aPos);
gl.vertexAttribPointer(aPos, 2, gl.FLOAT, false, 0, 0);
const U = {};
['uData','uCmap','uMercMin','uMercMax','uSite','uEl','uR0','uDr','uNgates',
 'uNaz','uMaxR','uOpacity'].forEach(n => U[n] = gl.getUniformLocation(prog, n));

// colormap LUT texture
(function(){
  const px = new Uint8Array(cb.stops.length * 4);
  cb.stops.forEach((hex, i) => {
    px[i*4]   = parseInt(hex.slice(1,3),16);
    px[i*4+1] = parseInt(hex.slice(3,5),16);
    px[i*4+2] = parseInt(hex.slice(5,7),16);
    px[i*4+3] = 255;
  });
  const t = gl.createTexture();
  gl.activeTexture(gl.TEXTURE1);
  gl.bindTexture(gl.TEXTURE_2D, t);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, cb.stops.length, 1, 0,
                gl.RGBA, gl.UNSIGNED_BYTE, px);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
})();

// per-frame polar textures, decoded lazily from data-URL PNGs
const texCache = new Array(frames.length).fill(null);
function frameTexture(i, cbk){
  if (texCache[i]) { cbk(texCache[i]); return; }
  const img = new (window.Image)();
  img.onload = () => {
    const t = gl.createTexture();
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, t);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.LUMINANCE, gl.LUMINANCE,
                  gl.UNSIGNED_BYTE, img);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    texCache[i] = t; cbk(t);
  };
  img.src = 'data:image/png;base64,' + frames[i].img;
}

const PROJ = L.Projection.SphericalMercator;
let idx = -1, opacity = 1.0, raf = 0;

function draw(){
  raf = 0;
  if (idx < 0) return;
  const f = frames[idx];
  frameTexture(idx, (tex) => {
    const sz = map.getSize(), dpr = window.devicePixelRatio || 1;
    if (glcv.width !== sz.x * dpr || glcv.height !== sz.y * dpr) {
      glcv.width = sz.x * dpr; glcv.height = sz.y * dpr;
      glcv.style.width = sz.x + 'px'; glcv.style.height = sz.y + 'px';
    }
    gl.viewport(0, 0, glcv.width, glcv.height);
    gl.clearColor(0, 0, 0, 0); gl.clear(gl.COLOR_BUFFER_BIT);
    const sw = PROJ.project(map.containerPointToLatLng([0, sz.y]));
    const ne = PROJ.project(map.containerPointToLatLng([sz.x, 0]));
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.uniform1i(U.uData, 0); gl.uniform1i(U.uCmap, 1);
    gl.uniform2f(U.uMercMin, sw.x, sw.y);
    gl.uniform2f(U.uMercMax, ne.x, ne.y);
    gl.uniform2f(U.uSite, SITE[1] * Math.PI / 180, SITE[0] * Math.PI / 180);
    gl.uniform1f(U.uEl, f.el * Math.PI / 180);
    gl.uniform1f(U.uR0, f.r0); gl.uniform1f(U.uDr, f.dr);
    gl.uniform1f(U.uNgates, f.ngates); gl.uniform1f(U.uNaz, f.naz);
    gl.uniform1f(U.uMaxR, f.maxr); gl.uniform1f(U.uOpacity, opacity);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  });
}
function requestDraw(){ if (!raf) raf = requestAnimationFrame(draw); }
map.on('move zoom zoomend moveend resize viewreset', requestDraw);

// ---------------------------------------------------------------- controls
const slider = document.getElementById('slider'),
      label = document.getElementById('label'),
      playBtn = document.getElementById('play'),
      op = document.getElementById('op');
let playing = false, timer = null;

function show(i){
  if (i === idx) return;
  idx = i;
  label.textContent = (i + 1) + '/' + frames.length + '  —  ' + frames[i].label;
  slider.value = i;
  ring.setRadius(frames[i].maxr);
  requestDraw();
}
slider.addEventListener('input', e => show(+e.target.value));
op.addEventListener('input', e => { opacity = e.target.value / 100; requestDraw(); });
playBtn.addEventListener('click', () => {
  playing = !playing;
  playBtn.innerHTML = playing ? '&#10074;&#10074;' : '&#9654;';
  if (playing) timer = setInterval(() => show((idx + 1) % frames.length), 450);
  else clearInterval(timer);
});
document.addEventListener('keydown', e => {
  if (e.key === 'ArrowRight') show(Math.min(idx + 1, frames.length - 1));
  if (e.key === 'ArrowLeft')  show(Math.max(idx - 1, 0));
});
document.getElementById('share').addEventListener('click', () => {
  const done = () => { const t = document.getElementById('toast');
    t.style.display = 'block'; setTimeout(() => t.style.display = 'none', 1600); };
  if (navigator.clipboard && navigator.clipboard.writeText)
    navigator.clipboard.writeText(SHARE_URL).then(done,
      () => window.prompt('Copy share link:', SHARE_URL));
  else window.prompt('Copy share link:', SHARE_URL);
});
// ---------------------------------------------------------------- overlays
map.createPane('refpane');
map.getPane('refpane').style.zIndex = 450;       // above the radar canvas
map.getPane('refpane').style.pointerEvents = 'none';
const COUNTY_URL =
  'https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json';
let countyLayer = null, countyLoading = false;
const roadLayer = L.tileLayer(
  'https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Transportation/MapServer/tile/{z}/{y}/{x}',
  {pane: 'refpane', attribution: 'Esri', opacity: 0.9});
document.getElementById('ck-counties').addEventListener('change', e => {
  const lbl = e.target.nextElementSibling;
  if (e.target.checked) {
    if (countyLayer) { countyLayer.addTo(map); return; }
    if (countyLoading) return;
    countyLoading = true; lbl.textContent = 'County boundaries…';
    fetch(COUNTY_URL).then(r => r.json()).then(gj => {
      countyLayer = L.geoJSON(gj, {pane: 'refpane', style:
        {color: '#C8C6C7', weight: 0.6, opacity: 0.7, fill: false}});
      if (document.getElementById('ck-counties').checked)
        countyLayer.addTo(map);
      lbl.textContent = 'County boundaries';
    }).catch(() => { lbl.textContent = 'Counties (load failed)'; })
      .finally(() => { countyLoading = false; });
  } else if (countyLayer) map.removeLayer(countyLayer);
});
document.getElementById('ck-interstates').addEventListener('change', e => {
  if (e.target.checked) roadLayer.addTo(map);
  else map.removeLayer(roadLayer);
});

// preload all frame textures in the background
frames.forEach((_, i) => setTimeout(() => frameTexture(i, () => {}), 50 * i));
show(0);
</script></body></html>"""


def build_page(frames, cbar, site, slat, slon, share_url=""):
    payload = json.dumps(
        [dict(img=f["img"], naz=f["naz"], ngates=f["ngates"], r0=f["r0"],
              dr=f["dr"], el=f["el"], maxr=f["maxr"], label=f["label"])
         for f in frames]
    )
    page = (LEAFLET_PAGE
            .replace("__FRAMES__", payload)
            .replace("__CBAR__", json.dumps(cbar))
            .replace("__MAXIDX__", str(len(frames) - 1))
            .replace("__SLAT__", f"{slat:.5f}")
            .replace("__SLON__", f"{slon:.5f}")
            .replace("__SITE__", site)
            .replace("__SHARE__", share_url))
    return (f'<iframe allow="clipboard-write" '
            f'style="width:100%;height:calc(100vh - 245px);'
            f'min-height:420px;border:0;border-radius:4px" '
            f'srcdoc="{html_mod.escape(page)}"></iframe>')


QUAD = "All fields (4-panel)"
ZVF = "Z+V (2-panel)"

QUAD_PAGE = """<!DOCTYPE html>
<html><head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
 @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@600;700&family=Source+Sans+3:wght@400;600;700&display=swap');
 :root{--io:#FF5F05;--ib:#13294B;--ib2:#1D3866;--ib3:#25457F;--sl:#C8C6C7}
 html,body{margin:0;height:100%;background:var(--ib)}
 :root{--barh:86px}
 #grid{position:absolute;inset:0 0 var(--barh) 0;display:grid;gap:4px;
       grid-template-columns:1fr;grid-template-rows:1fr;
       background:#0c1d38}
 #grid.quad{grid-template-columns:1fr 1fr;grid-template-rows:1fr 1fr}
 #grid.zv{grid-template-columns:1fr;grid-template-rows:1fr 1fr}
 #grid.zv.zvlr{grid-template-columns:1fr 1fr;grid-template-rows:1fr}
 .panel{position:relative;overflow:hidden}
 .pmap{position:absolute;inset:0}
 /* radar canvas lives in a Leaflet pane (z 250): tiles(200) < radar(250)
    < ring/vectors(400) < markers(600) < county/road overlays(650) */
 canvas.gl{pointer-events:none}
 .chip{position:absolute;top:6px;left:6px;z-index:1000;color:#fff;
       background:rgba(19,41,75,.92);border-left:3px solid var(--io);
       padding:3px 9px;font:700 11px 'Montserrat',sans-serif;
       letter-spacing:.04em;text-transform:uppercase;border-radius:2px}
 .pcbar{position:absolute;top:50%;right:6px;transform:translateY(-50%);
       z-index:1000;background:rgba(19,41,75,.92);
       border:1px solid var(--ib3);border-top:3px solid var(--io);
       border-radius:4px;padding:6px 5px;color:#fff;
       font:9px 'Source Sans 3',sans-serif;display:flex;gap:4px}
 .pcl{writing-mode:vertical-rl;transform:rotate(180deg);text-align:center;
       font-size:9px;opacity:.9}
 .pct{position:relative;width:26px}
 .pct div{position:absolute;left:0;transform:translateY(-50%);
       font-variant-numeric:tabular-nums}
 #bar{position:absolute;left:0;right:0;bottom:0;height:var(--barh);
      background:var(--ib);border-top:4px solid var(--io);color:#fff;
      font:13px 'Source Sans 3',system-ui,sans-serif;display:flex;
      flex-direction:column;justify-content:center;gap:6px;
      padding:6px 14px;box-sizing:border-box}
 #row1{display:flex;align-items:center;gap:10px;flex-wrap:nowrap}
 #slider{flex:1;accent-color:var(--io)}
 #op{accent-color:var(--io);width:90px}
 #label{font-variant-numeric:tabular-nums;white-space:nowrap;overflow:hidden;
        text-overflow:ellipsis;color:var(--sl)}
 button{background:var(--io);color:#fff;border:0;border-radius:4px;
        padding:4px 12px;cursor:pointer;font-size:13px;
        font-family:'Montserrat',sans-serif;font-weight:700}
 button:hover{background:#E25504}
 #refresh{font-size:16px;line-height:1;padding:4px 9px}
 #refresh.busy{animation:spin .8s linear infinite;pointer-events:none;opacity:.7}
 @keyframes spin{to{transform:rotate(360deg)}}
 #share{position:absolute;top:116px;left:10px;z-index:1100;width:34px;
        height:34px;display:flex;align-items:center;justify-content:center;
        cursor:pointer;background:var(--ib2);color:#fff;
        border:1px solid var(--ib3);border-radius:4px;
        box-shadow:0 1px 5px rgba(0,0,0,.4)}
 #share:hover{background:var(--io)}
 #toast{position:absolute;top:124px;left:52px;z-index:1100;display:none;
        background:var(--ib);border:1px solid var(--ib3);color:#fff;
        font:12px 'Source Sans 3',sans-serif;padding:4px 10px;
        border-radius:4px}
 .ck{display:flex;gap:5px;align-items:center;cursor:pointer;
     white-space:nowrap;color:var(--sl);font-size:12px}
 .ck input{accent-color:var(--io)}
 .nodata{position:absolute;inset:0;z-index:900;display:flex;
     align-items:center;justify-content:center;text-align:center;
     pointer-events:none;color:var(--sl);background:rgba(12,29,56,.45);
     font:600 13px 'Montserrat',sans-serif;letter-spacing:.05em;
     text-transform:uppercase}
 #export{position:absolute;top:158px;left:10px;z-index:1100;width:34px;
     height:34px;display:flex;align-items:center;justify-content:center;
     cursor:pointer;background:var(--ib2);color:#fff;
     border:1px solid var(--ib3);border-radius:4px;
     box-shadow:0 1px 5px rgba(0,0,0,.4)}
 #export:hover{background:var(--io)}
 #exmenu{position:absolute;top:158px;left:52px;z-index:1100;display:none;
     flex-direction:column;gap:6px;background:rgba(19,41,75,.95);
     border:1px solid var(--ib3);border-top:4px solid var(--io);
     border-radius:4px;padding:10px;color:#fff;
     font:12px 'Source Sans 3',sans-serif}
 #exmenu b{font-family:'Montserrat',sans-serif;font-size:10px;
     letter-spacing:.06em;text-transform:uppercase;opacity:.85}
 #exmenu button{text-align:left;font-family:'Source Sans 3',sans-serif;
     font-weight:600;font-size:12px;padding:6px 10px}
 @media (max-width:700px){
   :root{--barh:132px}
   #row1{flex-wrap:wrap;gap:8px;row-gap:6px}
   #slider{flex:1 1 100%;order:10}
   #modes button{padding:6px 10px;font-size:12px}
   #play{padding:6px 14px}
   .chip{font-size:9px;padding:2px 6px}
   .pcbar{padding:4px 3px;gap:3px}
   .pcbar canvas{height:120px;width:8px}
   .panel.solo .pcbar canvas{height:150px}
   .pct{width:22px}.pcl{font-size:8px}
   #share,#export{width:42px;height:42px}
   #share{top:104px}
   #export{top:154px}
   #exmenu{top:154px;left:60px}
   #label{font-size:11px}
   #bar{gap:4px;padding:6px 10px}
 }
 .leaflet-top.leaflet-left{top:30px}   /* keep zoom clear of the field chip */
 .panel{display:none}
 .panel .leaflet-control-zoom{display:none}
 .panel.zc .leaflet-control-zoom{display:block}
 .panel.solo .pcbar canvas{height:300px}
 #modes{display:flex;gap:4px;margin-left:4px}
 #modes button{padding:3px 9px;font-size:11px;background:var(--ib2);
   border:1px solid var(--ib3)}
 #modes button.act{background:var(--io);border-color:var(--io)}
</style></head><body>
<div id="grid"></div>
<div id="share" title="Copy share link">
 <svg viewBox="0 0 24 24" width="18" height="18" fill="none"
      stroke="currentColor" stroke-width="2" stroke-linecap="round">
   <circle cx="18" cy="5" r="3"/><circle cx="6" cy="12" r="3"/>
   <circle cx="18" cy="19" r="3"/>
   <line x1="8.6" y1="10.5" x2="15.4" y2="6.5"/>
   <line x1="8.6" y1="13.5" x2="15.4" y2="17.5"/>
 </svg>
</div>
<div id="toast">Link copied</div>
<div id="export" title="Export image / loop">
 <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor"
      stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
   <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"/>
   <circle cx="12" cy="13" r="4"/>
 </svg>
</div>
<div id="exmenu"><b>Export view</b>
 <button data-k="still4k">Still &middot; 4K (3840&times;2160)</button>
 <button data-k="stillig">Still &middot; Instagram reel (1080&times;1920)</button>
 <button data-k="loop4k">Loop &middot; 4K movie</button>
 <button data-k="loopig">Loop &middot; Instagram reel movie</button>
</div>
<div id="bar">
 <div id="row1">
   <button id="play">&#9654;</button>
   <button id="refresh" title="Refresh now: load new scans + warnings"
           style="display:none">&#x21bb;</button>
   <span id="modes"></span>
   <input id="slider" type="range" min="0" max="0" value="0" step="1"/>
   <label class="ck"><input type="checkbox" id="ck-counties" checked/><span>Counties</span></label>
   <label class="ck"><input type="checkbox" id="ck-interstates" checked/><span>Highways</span></label>
   <label class="ck"><input type="checkbox" id="ck-dealias"/><span>Dealias V</span></label>
   <span style="opacity:.7">opacity</span>
   <input id="op" type="range" min="10" max="100" value="100"/>
 </div>
 <div id="label"></div>
</div>
<script>
const DATA = __DATA__;
const SITE = [__SLAT__, __SLON__];
const SITE_ID = "__SITE__";
// reflect the loaded radar site in the browser tab title (the tab shows the
// top document's title; the bundle is same-origin with the app page)
try { parent.document.title =
        SITE_ID + " \\u00b7 NEXRAD Level 2 \\u2014 0.5\\u00b0 browser"; }
catch (e) { try { document.title = SITE_ID; } catch (e2) {} }
const SHARE_BASE = "__SHAREBASE__";
const QUADF = "All fields (4-panel)";
const ZVF = "Z+V (2-panel)";
const INIT_VIEW = __VIEW__;   // [lat, lon, zoom] from a share link, or null
const INIT_DEAL = __DEAL__;   // start with dealiased velocity shown
const RT = __RT__;            // real-time (trailing hour) mode
const WARN_URL = "__WARNURL__";
const CITIES_URL = "__CITIESURL__";
const LIVE_URL = "__LIVEURL__";        // full live frame set (fetched on change)
const LIVE_TAG_URL = "__LIVETAGURL__"; // tiny etag sidecar (polled cheaply)
const DEALF = "Radial velocity (dealiased)";
let DEAL = DATA.find(d => d.name === DEALF) || null;
const PANEL_DATA = DATA.filter(d => d.name !== DEALF);
let RAWV = PANEL_DATA.find(d => d.name === 'Radial velocity') || null;
let mode = __MODE__;
let zvOrient = 'tb';   // 2x1 orientation: 'tb' (top/bottom) or 'lr' (left/right)
if (mode === DEALF) mode = 'Radial velocity';
if (mode !== 'quad' && mode !== 'zv' &&
    !PANEL_DATA.some(fd => fd.name === mode))
  mode = PANEL_DATA[0].name;
// 2x1 mode requires both reflectivity and velocity to be present
if (mode === 'zv' && !(PANEL_DATA.some(d => d.name === 'Reflectivity') &&
                       PANEL_DATA.some(d => d.name === 'Radial velocity')))
  mode = PANEL_DATA[0].name;
const VS = `attribute vec2 aPos; varying vec2 vUV;
void main(){ vUV = aPos*0.5+0.5; gl_Position = vec4(aPos,0.,1.); }`;
const FS = `
precision highp float;
varying vec2 vUV;
uniform sampler2D uData, uCmap;
uniform vec2 uMercMin, uMercMax, uSite;
uniform float uEl, uR0, uDr, uNgates, uNaz, uMaxR, uOpacity;
const float R = 6378137.0; const float RE = 6371000.0;
const float K = 1.3333333; const float PI = 3.141592653589793;
void main(){
  vec2 merc = mix(uMercMin, uMercMax, vUV);
  float lat = 2.0*atan(exp(merc.y/R)) - PI/2.0;
  float lon = merc.x/R;
  float dLon = lon - uSite.x; float lat0 = uSite.y;
  float sdl = sin(dLon*0.5), sdp = sin((lat-lat0)*0.5);
  float a = sdp*sdp + cos(lat0)*cos(lat)*sdl*sdl;
  float s = 2.0*RE*asin(min(1.0,sqrt(a)));
  if (s > uMaxR*1.05) discard;
  float az = atan(sin(dLon)*cos(lat),
                  cos(lat0)*sin(lat)-sin(lat0)*cos(lat)*cos(dLon));
  az = mod(degrees(az)+360.0, 360.0);
  float Rk = K*RE; float r = s;
  for (int i=0;i<3;i++){
    float h = sqrt(r*r+Rk*Rk+2.0*r*Rk*sin(uEl))-Rk;
    float sg = Rk*asin(clamp(r*cos(uEl)/(Rk+h),-1.0,1.0));
    r = r*(1.0+(s-sg)/max(sg,1.0));
  }
  float gate = (r-uR0)/uDr;
  if (gate < 0.0 || gate >= uNgates) discard;
  float v = texture2D(uData, vec2((floor(gate)+0.5)/uNgates, az/360.0)).r;
  if (v < 0.002) discard;
  vec3 col = texture2D(uCmap, vec2((v*255.0-1.0)/254.0, 0.5)).rgb;
  gl_FragColor = vec4(col, uOpacity);
}`;
const PROJ = L.Projection.SphericalMercator;
let opacity = 1.0, idx = -1, playing = false, timer = null, syncing = false;
const panels = [];

function makePanel(fd, first){
  const wrap = document.createElement('div'); wrap.className = 'panel';
  document.getElementById('grid').appendChild(wrap);
  const mdiv = document.createElement('div'); mdiv.className = 'pmap';
  wrap.appendChild(mdiv);
  const cv = document.createElement('canvas'); cv.className = 'gl';
  const st = {fd: fd};   // swappable dataset (raw <-> dealiased velocity)
  const chip = document.createElement('div'); chip.className = 'chip';
  chip.textContent = fd.name; wrap.appendChild(chip);
  if (!fd.frames.length) {
    const nd = document.createElement('div'); nd.className = 'nodata';
    nd.textContent = (fd.name.indexOf('reflectivity') > 0 ||
                      fd.name.indexOf('Correlation') === 0)
      ? 'Pre-polarimetric upgrade data' : 'No data for this hour';
    wrap.appendChild(nd);
  }
  // mini colorbar (repaintable)
  const cwrap = document.createElement('div'); cwrap.className = 'pcbar';
  const cl = document.createElement('div'); cl.className = 'pcl';
  const ccv = document.createElement('canvas'); ccv.width = 10; ccv.height = 180;
  const ct = document.createElement('div'); ct.className = 'pct';
  cwrap.appendChild(cl); cwrap.appendChild(ccv); cwrap.appendChild(ct);
  wrap.appendChild(cwrap);
  function paintCbar(cb){
    cl.textContent = cb.label;
    const cctx = ccv.getContext('2d');
    cctx.clearRect(0,0,ccv.width,ccv.height);
    const g = cctx.createLinearGradient(0, ccv.height, 0, 0);
    cb.stops.forEach((c,i)=>g.addColorStop(i/(cb.stops.length-1), c));
    cctx.fillStyle = g; cctx.fillRect(0,0,ccv.width,ccv.height);
    ct.innerHTML = '';
    const t0 = Math.ceil(cb.vmin/cb.tick)*cb.tick;
    for (let v=t0; v<=cb.vmax+1e-9; v+=cb.tick){
      const val = +v.toFixed(6);
      const d = document.createElement('div');
      d.style.top = (100-(val-cb.vmin)/(cb.vmax-cb.vmin)*100)+'%';
      d.textContent = ''+(cb.tick<1?val.toFixed(1):Math.round(val));
      ct.appendChild(d);
    }
  }
  const cb = fd.cbar;
  paintCbar(cb);
  // map (zoom control on all; CSS shows it only on the first visible panel)
  const map = L.map(mdiv, {center:SITE, zoom:7, zoomControl:true,
                           attributionControl:first});
  // city labels pane (canvas) above overlays, below warnings
  map.createPane('citypane');
  map.getPane('citypane').style.zIndex = 655;
  map.getPane('citypane').style.pointerEvents = 'none';
  const cityCv = document.createElement('canvas');
  map.getPane('citypane').appendChild(cityCv);
  function cityDraw(){
    if (!window.CITIES) return;
    const sz = map.getSize(), dpr = window.devicePixelRatio || 1;
    if (cityCv.width !== sz.x*dpr || cityCv.height !== sz.y*dpr){
      cityCv.width = sz.x*dpr; cityCv.height = sz.y*dpr;
      cityCv.style.width = sz.x+'px'; cityCv.style.height = sz.y+'px';
    }
    L.DomUtil.setPosition(cityCv, map.containerPointToLayerPoint([0,0]));
    const g = cityCv.getContext('2d');
    g.setTransform(dpr,0,0,dpr,0,0);
    g.clearRect(0,0,sz.x,sz.y);
    const z = map.getZoom(), b = map.getBounds();
    const minPop = cityPopMin(z);
    const placed = [];
    let n = 0;
    g.textBaseline = 'middle'; g.textAlign = 'left';
    g.font = '600 12px Arial, Helvetica, sans-serif';
    for (let i = 0; i < window.CITIES.length && n < 110; i++){
      const c = window.CITIES[i];           // [name, lat, lon, pop]
      if (c[3] < minPop) { if (minPop > 0) continue; }
      if (c[1] < b.getSouth() || c[1] > b.getNorth() ||
          c[2] < b.getWest()  || c[2] > b.getEast()) continue;
      const pt = map.latLngToContainerPoint([c[1], c[2]]);
      let ok = true;
      for (let j = 0; j < placed.length; j++){
        if (Math.abs(pt.x-placed[j][0]) < 92 &&
            Math.abs(pt.y-placed[j][1]) < 18){ ok = false; break; }
      }
      if (!ok) continue;
      placed.push([pt.x, pt.y]); n++;
      g.fillStyle = '#FFFFFF';
      g.beginPath(); g.arc(pt.x, pt.y, 2.4, 0, 2*Math.PI); g.fill();
      g.lineWidth = 3; g.strokeStyle = 'rgba(7,16,32,.85)';
      g.strokeText(c[0], pt.x+6, pt.y);
      g.fillText(c[0], pt.x+6, pt.y);
    }
  }
  map.on('move zoom zoomend moveend resize viewreset', cityDraw);
  // warnings pane sits above everything else
  map.createPane('warnpane');
  map.getPane('warnpane').style.zIndex = 660;
  map.getPane('warnpane').style.pointerEvents = 'none';
  // radar canvas pane sits between basemap tiles and all vector panes
  map.createPane('radarpane');
  map.getPane('radarpane').style.zIndex = 250;
  map.getPane('radarpane').style.pointerEvents = 'none';
  map.getPane('radarpane').appendChild(cv);
  // reference overlays (counties / roads) draw above everything
  map.createPane('refpane');
  map.getPane('refpane').style.zIndex = 650;
  map.getPane('refpane').style.pointerEvents = 'none';
  L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_nolabels/{z}/{x}/{y}{r}.png',
    {attribution:'&copy; OSM &copy; CARTO', subdomains:'abcd', maxZoom:14})
    .addTo(map);
  L.circleMarker(SITE,{radius:4,color:'#fff',weight:2,fillColor:'#e33',
    fillOpacity:1}).addTo(map).bindTooltip('__SITE__');
  const ring = L.circle(SITE,{radius:fd.frames.length?fd.frames[0].maxr:3e5,
    color:'#fff',weight:1.2,opacity:.9,fill:false,interactive:false})
    .addTo(map);
  // webgl
  const gl = cv.getContext('webgl',{premultipliedAlpha:false,alpha:true});
  function sh(t,src){ const s=gl.createShader(t); gl.shaderSource(s,src);
    gl.compileShader(s); return s; }
  const prog = gl.createProgram();
  gl.attachShader(prog, sh(gl.VERTEX_SHADER,VS));
  gl.attachShader(prog, sh(gl.FRAGMENT_SHADER,FS));
  gl.linkProgram(prog); gl.useProgram(prog);
  gl.bindBuffer(gl.ARRAY_BUFFER, gl.createBuffer());
  gl.bufferData(gl.ARRAY_BUFFER,
    new Float32Array([-1,-1,1,-1,-1,1,1,1]), gl.STATIC_DRAW);
  const aPos = gl.getAttribLocation(prog,'aPos');
  gl.enableVertexAttribArray(aPos);
  gl.vertexAttribPointer(aPos,2,gl.FLOAT,false,0,0);
  const U = {};
  ['uData','uCmap','uMercMin','uMercMax','uSite','uEl','uR0','uDr','uNgates',
   'uNaz','uMaxR','uOpacity'].forEach(n=>U[n]=gl.getUniformLocation(prog,n));
  const px = new Uint8Array(cb.stops.length*4);
  cb.stops.forEach((hex,i)=>{px[i*4]=parseInt(hex.slice(1,3),16);
    px[i*4+1]=parseInt(hex.slice(3,5),16);
    px[i*4+2]=parseInt(hex.slice(5,7),16); px[i*4+3]=255;});
  const ct2 = gl.createTexture();
  gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, ct2);
  gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA,cb.stops.length,1,0,gl.RGBA,
                gl.UNSIGNED_BYTE,px);
  ['TEXTURE_MIN_FILTER','TEXTURE_MAG_FILTER'].forEach(p=>
    gl.texParameteri(gl.TEXTURE_2D, gl[p], gl.LINEAR));
  ['TEXTURE_WRAP_S','TEXTURE_WRAP_T'].forEach(p=>
    gl.texParameteri(gl.TEXTURE_2D, gl[p], gl.CLAMP_TO_EDGE));
  let texCache = {};
  function texFor(i, cbk){
    if (texCache[i]) { cbk(texCache[i]); return; }
    const img = new (window.Image)();
    img.onload = ()=>{
      const t = gl.createTexture();
      gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D,t);
      gl.texImage2D(gl.TEXTURE_2D,0,gl.LUMINANCE,gl.LUMINANCE,
                    gl.UNSIGNED_BYTE,img);
      ['TEXTURE_MIN_FILTER','TEXTURE_MAG_FILTER'].forEach(p=>
        gl.texParameteri(gl.TEXTURE_2D, gl[p], gl.NEAREST));
      ['TEXTURE_WRAP_S','TEXTURE_WRAP_T'].forEach(p=>
        gl.texParameteri(gl.TEXTURE_2D, gl[p], gl.CLAMP_TO_EDGE));
      texCache[i] = t; cbk(t);
    };
    img.src = 'data:image/png;base64,' + st.fd.frames[i].img;
  }
  let raf = 0;
  function draw(){
    raf = 0;
    const pi = Math.min(idx, st.fd.frames.length-1);
    if (pi < 0) return;
    const f = st.fd.frames[pi];
    texFor(pi, (tex)=>{
      const sz = map.getSize(), dpr = window.devicePixelRatio||1;
      if (cv.width!==sz.x*dpr||cv.height!==sz.y*dpr){
        cv.width=sz.x*dpr; cv.height=sz.y*dpr;
        cv.style.width=sz.x+'px'; cv.style.height=sz.y+'px';
      }
      // counter the map-pane's pan transform so the canvas stays
      // screen-aligned while living inside the pane stack
      L.DomUtil.setPosition(cv, map.containerPointToLayerPoint([0,0]));
      gl.viewport(0,0,cv.width,cv.height);
      gl.clearColor(0,0,0,0); gl.clear(gl.COLOR_BUFFER_BIT);
      const sw = PROJ.project(map.containerPointToLatLng([0,sz.y]));
      const ne = PROJ.project(map.containerPointToLatLng([sz.x,0]));
      gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D,tex);
      gl.uniform1i(U.uData,0); gl.uniform1i(U.uCmap,1);
      gl.uniform2f(U.uMercMin,sw.x,sw.y); gl.uniform2f(U.uMercMax,ne.x,ne.y);
      gl.uniform2f(U.uSite,SITE[1]*Math.PI/180,SITE[0]*Math.PI/180);
      gl.uniform1f(U.uEl,f.el*Math.PI/180);
      gl.uniform1f(U.uR0,f.r0); gl.uniform1f(U.uDr,f.dr);
      gl.uniform1f(U.uNgates,f.ngates); gl.uniform1f(U.uNaz,f.naz);
      gl.uniform1f(U.uMaxR,f.maxr); gl.uniform1f(U.uOpacity,opacity);
      gl.enable(gl.BLEND); gl.blendFunc(gl.SRC_ALPHA,gl.ONE_MINUS_SRC_ALPHA);
      gl.drawArrays(gl.TRIANGLE_STRIP,0,4);
    });
  }
  function requestDraw(){ if(!raf) raf=requestAnimationFrame(draw); }
  map.on('move zoom zoomend moveend resize viewreset', requestDraw);
  function setData(nf){
    st.fd = nf; texCache = {};
    chip.textContent = nf.name;
    paintCbar(nf.cbar);
    const f = nf.frames[Math.min(Math.max(idx,0), nf.frames.length-1)];
    if (f) ring.setRadius(f.maxr);
    if (idx >= 0) refreshLabel(idx);
    requestDraw();
  }
  return {map, get fd(){ return st.fd; }, name: fd.name, ring,
          requestDraw, texFor, wrap, setData, cityDraw};
}

PANEL_DATA.forEach((fd,i)=>panels.push(makePanel(fd, i===0)));

function cityPopMin(z){
  if (z >= 12) return 0;
  if (z >= 11) return 2000;
  if (z >= 10) return 6000;
  if (z >= 9)  return 15000;
  if (z >= 8)  return 40000;
  if (z >= 7)  return 90000;
  if (z >= 6)  return 250000;
  return 600000;
}
fetch(CITIES_URL).then(r=>r.json()).then(arr=>{
  window.CITIES = arr;
  panels.forEach(p=>p.cityDraw());
}).catch(()=>{});

// view sync
panels.forEach(p=>{
  p.map.on('move', ()=>{
    if (syncing) return;
    syncing = true;
    const c = p.map.getCenter(), z = p.map.getZoom();
    panels.forEach(o=>{ if(o!==p) o.map.setView(c,z,{animate:false}); });
    syncing = false;
  });
});

// view-mode switcher: every field's textures live in this one page, so
// toggling Z/V/ZDR/CC/2x2 is instant — no reload, no refetch
const SHORT = {'Reflectivity':'Z','Radial velocity':'V',
  'Radial velocity (dealiased)':'V\\u2020',
  'Differential reflectivity':'ZDR','Correlation coefficient':'CC'};
const modesEl = document.getElementById('modes');
function mkBtn(txt, m){
  const b = document.createElement('button');
  b.textContent = txt; b.dataset.m = m;
  b.onclick = ()=>setMode(m);
  modesEl.appendChild(b);
}
PANEL_DATA.forEach(fd=>mkBtn(SHORT[fd.name]||fd.name, fd.name));
const HAS_ZV = PANEL_DATA.some(d=>d.name==='Reflectivity') &&
               PANEL_DATA.some(d=>d.name==='Radial velocity');
if (HAS_ZV) mkBtn('Z+V', 'zv');
if (PANEL_DATA.length > 1) mkBtn('2\\u00d72', 'quad');
// the two fields shown in 2x1 mode (match on stable panel name, not fd.name,
// so a dealiased velocity panel still counts as the velocity panel)
const ZV_SET = {'Reflectivity':1, 'Radial velocity':1};
function setMode(m){
  // clicking Z+V while already in 2x1 flips its orientation (tb <-> lr)
  if (m === 'zv' && mode === 'zv') zvOrient = (zvOrient === 'tb' ? 'lr' : 'tb');
  mode = m;
  const grid = document.getElementById('grid');
  let zcDone = false;
  panels.forEach(p=>{
    const vis = (m === 'quad') || (m === 'zv' && ZV_SET[p.name]) ||
                (p.fd.name === m);
    p.wrap.style.display = vis ? 'block' : 'none';
    p.wrap.classList.toggle('solo', vis && m !== 'quad' && m !== 'zv');
    const zc = vis && !zcDone;
    p.wrap.classList.toggle('zc', zc);
    if (zc) zcDone = true;
  });
  grid.classList.toggle('quad', m === 'quad');
  grid.classList.toggle('zv', m === 'zv');
  grid.classList.toggle('zvlr', m === 'zv' && zvOrient === 'lr');
  document.querySelectorAll('#modes button').forEach(b=>
    b.classList.toggle('act', b.dataset.m === m));
  setTimeout(()=>{
    panels.forEach(p=>{ p.map.invalidateSize(false); p.requestDraw(); });
    if (idx >= 0) refreshLabel(idx);
  }, 60);
}

// initial view
setMode(mode);
setTimeout(function(){
  panels.forEach(p=>p.map.invalidateSize(false));
  const vis = panels.find(p=>p.wrap.style.display !== 'none') || panels[0];
  if (INIT_VIEW && INIT_VIEW.length === 3) {
    vis.map.setView([INIT_VIEW[0], INIT_VIEW[1]], INIT_VIEW[2]);
    return;
  }
  vis.map.setView(SITE, 8);   // default zoom on load
}, 90);

// controls
let nmax = Math.max.apply(null, PANEL_DATA.map(fd=>fd.frames.length));
const slider = document.getElementById('slider');
slider.max = nmax-1;
const label = document.getElementById('label'),
      playBtn = document.getElementById('play'),
      op = document.getElementById('op');
function refreshLabel(i){
  const ap = panels.find(p=>p.wrap.style.display !== 'none') || panels[0];
  const f = ap.fd.frames[Math.min(i, ap.fd.frames.length-1)];
  label.textContent = (i+1)+'/'+nmax+'  —  '+(f ? f.label : '');
}
function show(i){
  if (i===idx) return;
  idx = i;
  slider.value = i;
  refreshLabel(i);
  panels.forEach(p=>{
    const f = p.fd.frames[Math.min(i, p.fd.frames.length-1)];
    if (f) p.ring.setRadius(f.maxr);
    p.requestDraw();
  });
}
slider.addEventListener('input', e=>show(+e.target.value));
op.addEventListener('input', e=>{opacity=e.target.value/100;
  panels.forEach(p=>p.requestDraw());});
const FRAME_MS = 450, END_DWELL = 4;   // hold the last frame 4x before looping
function playStep(){
  const next = (idx+1) % nmax;
  show(next);
  timer = setTimeout(playStep,
                     FRAME_MS * (next === nmax-1 ? END_DWELL : 1));
}
playBtn.addEventListener('click', ()=>{
  playing=!playing;
  playBtn.innerHTML = playing?'&#10074;&#10074;':'&#9654;';
  if(playing) timer = setTimeout(playStep,
                     FRAME_MS * (idx === nmax-1 ? END_DWELL : 1));
  else clearTimeout(timer);
});
document.addEventListener('keydown', e=>{
  if(e.key==='ArrowRight') show(Math.min(idx+1,nmax-1));
  if(e.key==='ArrowLeft') show(Math.max(idx-1,0));
});
document.getElementById('share').addEventListener('click', ()=>{
  const vis = panels.find(p=>p.wrap.style.display !== 'none') || panels[0];
  const c = vis.map.getCenter(), z = vis.map.getZoom();
  let url = SHARE_BASE + '&field=' +
    encodeURIComponent(mode === 'quad' ? QUADF : (mode === 'zv' ? ZVF : mode)) +
    '&lat=' + c.lat.toFixed(4) + '&lon=' + c.lng.toFixed(4) + '&zoom=' + z;
  const ckd = document.getElementById('ck-dealias');
  if (ckd && ckd.checked) url += '&dealias=1';
  // SHARE_BASE is a bare query (?...) when the server didn't know the Space
  // host (always so in live mode) — prepend the app's origin+path so the
  // copied link is a full, openable URL
  if (!/^https?:/i.test(url)){
    let base = '';
    try { base = parent.location.origin + parent.location.pathname; }
    catch (e) { base = location.origin + location.pathname; }
    url = base + url;
  }
  const done=()=>{const t=document.getElementById('toast');
    t.style.display='block'; setTimeout(()=>t.style.display='none',1600);};
  if(navigator.clipboard&&navigator.clipboard.writeText)
    navigator.clipboard.writeText(url).then(done,
      ()=>window.prompt('Copy share link:',url));
  else window.prompt('Copy share link:',url);
});

// overlays applied to all panels
const COUNTY_URL =
  'https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json';
let countyLayers = null, countyLoading = false, countyGJ = null;
const roadLayers = panels.map(p=>L.tileLayer(
  'https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Transportation/MapServer/tile/{z}/{y}/{x}',
  {attribution:'Esri', opacity:1.0, pane:'refpane'}));
document.getElementById('ck-counties').addEventListener('change', e=>{
  const lbl = e.target.nextElementSibling;
  if (e.target.checked){
    if (countyLayers){ countyLayers.forEach((l,i)=>l.addTo(panels[i].map));
      return; }
    if (countyLoading) return;
    countyLoading = true; lbl.textContent='Counties…';
    fetch(COUNTY_URL).then(r=>r.json()).then(gj=>{
      countyGJ = gj;
      countyLayers = panels.map(()=>L.geoJSON(gj,{pane:'refpane',style:
        {color:'#C8C6C7',weight:0.7,opacity:1.0,fill:false}}));
      if (document.getElementById('ck-counties').checked)
        countyLayers.forEach((l,i)=>l.addTo(panels[i].map));
      lbl.textContent='Counties';
    }).catch(()=>{lbl.textContent='Counties (load failed)';})
      .finally(()=>{countyLoading=false;});
  } else if (countyLayers)
    countyLayers.forEach((l,i)=>panels[i].map.removeLayer(l));
});
document.getElementById('ck-interstates').addEventListener('change', e=>{
  if (e.target.checked) roadLayers.forEach((l,i)=>l.addTo(panels[i].map));
  else roadLayers.forEach((l,i)=>panels[i].map.removeLayer(l));
});
// both overlays start enabled
document.getElementById('ck-counties').dispatchEvent(new Event('change'));
document.getElementById('ck-interstates').dispatchEvent(new Event('change'));

// dealias toggle: swap the velocity panel between raw and dealiased.
// If this hour hasn't been dealiased server-side yet, click the hidden
// Gradio trigger in the parent page to reprocess (page will update).
const ckDeal = document.getElementById('ck-dealias');
const vPanel = panels.find(p=>p.name === 'Radial velocity') || null;
if (!vPanel || !RAWV || !RAWV.frames.length)
  ckDeal.parentElement.style.display = 'none';
function applyDeal(on){
  if (!vPanel) return;
  if (on){
    if (DEAL && DEAL.frames.length){
      vPanel.setData(DEAL);
    } else {
      exToast('Reprocessing hour with region-based dealiasing\\u2026', true);
      try {
        // adding the dealiased field rebuilds the iframe; hand the current
        // frame to the new bundle (via the shared parent) so it doesn't reset
        parent.__restoreIdx = idx;
        const host = parent.document.getElementById('dax-trigger');
        (host.querySelector('button') || host).click();
      } catch (err) {
        exToast('Could not start dealiasing'); ckDeal.checked = false;
      }
    }
  } else if (vPanel.fd.name === DEALF){
    vPanel.setData(RAWV);
  }
}
ckDeal.addEventListener('change', e=>applyDeal(e.target.checked));
if (INIT_DEAL && DEAL && DEAL.frames.length){
  ckDeal.checked = true; applyDeal(true);
}

// ------------------------------------------------------------- real-time
if (RT){
  // Live updates merge new frames IN PLACE — no iframe reload. The old design
  // reloaded the whole ~40MB bundle every tick (parent rt-refresh click /
  // server Timer), blanking the map ("grey-out"). Now we poll a tiny etag
  // sidecar; only when it changes do we fetch the full frame set and swap
  // each panel's data via setData, preserving pan/zoom/playback/mode.
  let liveTag = __LIVETAG0__;
  async function fetchTag(){
    try {
      const r = await fetch(LIVE_TAG_URL + '?t=' + Date.now(), {cache:'no-store'});
      if (r.ok) return await r.json();   // {etag, ts}
    } catch (err) {}
    return null;
  }
  function applyLive(j){
    if (!j || !j.fields) return;
    const byName = {};
    j.fields.forEach(fd=>{ byName[fd.name] = fd; });
    if (byName[DEALF]) DEAL = byName[DEALF];
    if (byName['Radial velocity']) RAWV = byName['Radial velocity'];
    const oldNmax = nmax;
    panels.forEach(p=>{
      // a velocity panel currently showing dealiased data keeps showing it
      if (p.name === 'Radial velocity' && p.fd.name === DEALF){
        if (DEAL && DEAL.frames.length) p.setData(DEAL);
        return;
      }
      const nf = byName[p.name];
      if (nf) p.setData(nf);
    });
    nmax = Math.max.apply(null, panels.map(p=>p.fd.frames.length));
    slider.max = Math.max(0, nmax-1);
    if (idx >= oldNmax-1 && nmax-1 !== idx) show(nmax-1);   // follow newest
    else { if (idx>=0) refreshLabel(idx); panels.forEach(p=>p.requestDraw()); }
  }
  async function pollLive(){
    const t = await fetchTag();
    if (!t || t.etag === liveTag) return false;
    try {
      const r2 = await fetch(LIVE_URL + '?t=' + Date.now(), {cache:'no-store'});
      if (!r2.ok) return false;
      const j = await r2.json();
      if (j && j.etag){ liveTag = j.etag; applyLive(j); return true; }
    } catch (err) {}
    return false;
  }
  setInterval(pollLive, 30 * 1000);
  // poll current TOR/SVR warnings every minute; redraw on change
  let warnTag = null, warnLayers = [];
  async function pollWarnings(){
    try {
      const r = await fetch(WARN_URL + '?t=' + Date.now(),
                            {cache: 'no-store'});
      if (!r.ok) return;
      const j = await r.json();
      if (j.etag === warnTag) return;
      warnTag = j.etag;
      window.WARN_GJ = j;          // stash for the export compositor
      warnLayers.forEach(l=>l.remove());
      // backup guard: hide any warning already past its UTC expiry
      // (YYYYMMDDHHMM, fixed-width so string compare is safe)
      const nd = new Date(), pad = n => String(n).padStart(2, '0');
      const nowZ = '' + nd.getUTCFullYear() + pad(nd.getUTCMonth()+1) +
        pad(nd.getUTCDate()) + pad(nd.getUTCHours()) + pad(nd.getUTCMinutes());
      const WARN_W = 2.5, WARN_BUF = 0.25 * (96/72);   // 0.25pt black buffer
      const keep = f=> !f.properties.exp || f.properties.exp > nowZ;
      warnLayers = [];
      panels.forEach(p=>{
        // black casing first (drawn under), colored warning line on top
        const casing = L.geoJSON(j, {pane: 'warnpane', filter: keep,
          style: ()=>({color:'#000', weight: WARN_W + 2*WARN_BUF,
                       opacity: 1, fill: false})});
        const line = L.geoJSON(j, {pane: 'warnpane', filter: keep,
          style: f=>({color: warnColor(f.properties.ph),
                      weight: WARN_W, opacity: 1, fill: false})});
        casing.addTo(p.map); line.addTo(p.map);
        warnLayers.push(casing, line);
      });
    } catch (err) {}
  }
  pollWarnings();
  setInterval(pollWarnings, 60 * 1000);

  // forced-refresh button: warnings update instantly; ask the server to
  // re-decode the live hour now (picks up any new scans), then fast-poll the
  // frame set until it lands — all in place, no iframe reload
  const refreshBtn = document.getElementById('refresh');
  if (refreshBtn){
    refreshBtn.style.display = '';
    let refreshing = false;
    refreshBtn.addEventListener('click', async ()=>{
      if (refreshing) return;
      refreshing = true; refreshBtn.classList.add('busy');
      pollWarnings();
      const base = await fetchTag();
      const baseTs = base && base.ts ? base.ts : 0;
      try {
        const h = parent.document.getElementById('rt-force');
        (h.querySelector('button') || h).click();
      } catch (err) {}
      const t0 = Date.now();
      while (Date.now() - t0 < 15000){
        await new Promise(r=>setTimeout(r, 500));
        const t = await fetchTag();
        if (!t) continue;
        if (t.etag !== liveTag){ await pollLive(); break; }   // new scans merged
        if (baseTs && t.ts && t.ts > baseTs) break;           // server done, no change
      }
      pollWarnings();
      refreshBtn.classList.remove('busy'); refreshing = false;
    });
  }
}

// ---------------------------------------------------------------- export
function activePanel(){
  return panels.find(p=>p.wrap.style.display!=='none') || panels[0];
}
function exToast(t, sticky){
  const el = document.getElementById('toast');
  el.textContent = t; el.style.display = 'block';
  clearTimeout(exToast._t);
  if (!sticky) exToast._t = setTimeout(()=>{
    el.style.display='none'; el.textContent='Link copied';}, 2600);
}
function loadImg(src, cors){
  return new Promise((res, rej)=>{
    const im = new (window.Image)();
    if (cors) im.crossOrigin = 'anonymous';
    im.onload = ()=>res(im);
    im.onerror = ()=>rej(new Error('image load failed'));
    im.src = src;
  });
}
const MERC_R = 6378137, MERC_FULL = 2*Math.PI*MERC_R;
function mercX(lon){ return MERC_R*lon*Math.PI/180; }
function mercY(lat){ return MERC_R*Math.log(Math.tan(Math.PI/4+lat*Math.PI/360)); }

async function drawTiles(ctx, tpl, win, w, subs){
  const mpp = (win.x1-win.x0)/w;
  let z = Math.round(Math.log2(MERC_FULL/(256*mpp)));
  z = Math.max(2, Math.min(18, z));
  const n = Math.pow(2, z), ts = MERC_FULL/n;
  const tx0 = Math.floor((win.x0+MERC_FULL/2)/ts),
        tx1 = Math.floor((win.x1+MERC_FULL/2)/ts),
        ty0 = Math.floor((MERC_FULL/2-win.y1)/ts),
        ty1 = Math.floor((MERC_FULL/2-win.y0)/ts);
  const jobs = [];
  for (let tx=tx0; tx<=tx1; tx++) for (let ty=ty0; ty<=ty1; ty++){
    if (ty<0 || ty>=n) continue;
    const xm = ((tx%n)+n)%n;
    const url = tpl.replace('{z}',z).replace('{x}',xm).replace('{y}',ty)
                   .replace('{s}', subs ? subs[(tx+ty)%subs.length] : '');
    const px = (tx*ts-MERC_FULL/2-win.x0)/mpp;
    const py = (win.y1-(MERC_FULL/2-ty*ts))/mpp;
    jobs.push(loadImg(url, true)
      .then(im=>ctx.drawImage(im, px, py, ts/mpp+0.6, ts/mpp+0.6))
      .catch(()=>{}));
  }
  await Promise.all(jobs);
}

function drawCitiesExport(ctx, win, w, h){
  if (!window.CITIES) return;
  const mpp = (win.x1-win.x0)/w;
  const zEff = Math.round(Math.log2(MERC_FULL/(256*mpp)));
  const minPop = cityPopMin(zEff);
  const s = Math.min(w, h);
  const fs = Math.max(12, Math.round(s*0.016));
  ctx.save();
  ctx.font = '600 '+fs+'px Arial, Helvetica, sans-serif';
  ctx.textBaseline = 'middle'; ctx.textAlign = 'left';
  const placed = [];
  let n = 0;
  for (let i = 0; i < window.CITIES.length && n < 120; i++){
    const c = window.CITIES[i];
    if (c[3] < minPop && minPop > 0) continue;
    const mx = mercX(c[2]), my = mercY(c[1]);
    if (mx < win.x0 || mx > win.x1 || my < win.y0 || my > win.y1) continue;
    const px = (mx-win.x0)/mpp, py = (win.y1-my)/mpp;
    let ok = true;
    for (let j = 0; j < placed.length; j++){
      if (Math.abs(px-placed[j][0]) < fs*8 &&
          Math.abs(py-placed[j][1]) < fs*1.6){ ok = false; break; }
    }
    if (!ok) continue;
    placed.push([px, py]); n++;
    ctx.fillStyle = '#FFFFFF';
    ctx.beginPath(); ctx.arc(px, py, fs*0.2, 0, 2*Math.PI); ctx.fill();
    ctx.lineWidth = fs*0.25; ctx.strokeStyle = 'rgba(7,16,32,.85)';
    ctx.strokeText(c[0], px+fs*0.5, py);
    ctx.fillText(c[0], px+fs*0.5, py);
  }
  ctx.restore();
}

function drawCounties(ctx, win, w){
  if (!countyGJ) return;
  const mpp = (win.x1-win.x0)/w;
  ctx.save();
  ctx.strokeStyle = '#C8C6C7'; ctx.globalAlpha = 0.9;
  ctx.lineWidth = Math.max(1, w/2200);
  const drawRing = ring=>{
    ctx.beginPath();
    ring.forEach((pt,i)=>{
      const x = (mercX(pt[0])-win.x0)/mpp, y = (win.y1-mercY(pt[1]))/mpp;
      if (i) ctx.lineTo(x,y); else ctx.moveTo(x,y);
    });
    ctx.stroke();
  };
  countyGJ.features.forEach(f=>{
    const g = f.geometry; if (!g) return;
    if (g.type === 'Polygon') g.coordinates.forEach(drawRing);
    else if (g.type === 'MultiPolygon')
      g.coordinates.forEach(p=>p.forEach(drawRing));
  });
  ctx.restore();
}

// warning outline colors: tornado red, severe-tstorm yellow, flash-flood green
function warnColor(ph){
  return ph === 'TO' ? '#FF2A2A' : (ph === 'FF' ? '#2EE66B' : '#FFE12A');
}

function drawWarningsExport(ctx, win, w){
  const gj = window.WARN_GJ;
  if (!gj || !gj.features || !gj.features.length) return;
  // drop any warning already past its UTC expiry (YYYYMMDDHHMM string compare)
  const nd = new Date(), pad = n => String(n).padStart(2, '0');
  const nowZ = '' + nd.getUTCFullYear() + pad(nd.getUTCMonth()+1) +
    pad(nd.getUTCDate()) + pad(nd.getUTCHours()) + pad(nd.getUTCMinutes());
  const mpp = (win.x1-win.x0)/w;
  const colorW = Math.max(2, w/500);    // bold so it reads on 4K stills/movies
  const buf = 0.25 * (96/72) * (w/900); // 0.25pt black buffer, scaled to export
  ctx.save();
  ctx.lineJoin = 'round';
  const ringPath = ring=>{
    ctx.beginPath();
    ring.forEach((pt,i)=>{
      const x = (mercX(pt[0])-win.x0)/mpp, y = (win.y1-mercY(pt[1]))/mpp;
      if (i) ctx.lineTo(x,y); else ctx.moveTo(x,y);
    });
    ctx.closePath();
  };
  const strokeGeom = g=>{
    if (g.type === 'Polygon') g.coordinates.forEach(r=>{ ringPath(r); ctx.stroke(); });
    else if (g.type === 'MultiPolygon')
      g.coordinates.forEach(p=>p.forEach(r=>{ ringPath(r); ctx.stroke(); }));
  };
  const each = cb=>gj.features.forEach(f=>{
    const pr = f.properties || {};
    if (pr.exp && pr.exp <= nowZ) return;
    const g = f.geometry; if (!g) return;
    cb(pr, g);
  });
  // pass 1: black casing (0.25pt buffer outside the colored stroke)
  ctx.strokeStyle = '#000'; ctx.lineWidth = colorW + 2*buf;
  each((pr, g)=>strokeGeom(g));
  // pass 2: colored warning line on top
  ctx.lineWidth = colorW;
  each((pr, g)=>{ ctx.strokeStyle = warnColor(pr.ph);
                  strokeGeom(g); });
  ctx.restore();
}

function makeExportGL(w, h, cb){
  const cnv = document.createElement('canvas');
  cnv.width = w; cnv.height = h;
  const gl = cnv.getContext('webgl', {premultipliedAlpha:false, alpha:true,
                                      preserveDrawingBuffer:true});
  function sh(t,src){ const s=gl.createShader(t); gl.shaderSource(s,src);
    gl.compileShader(s); return s; }
  const prog = gl.createProgram();
  gl.attachShader(prog, sh(gl.VERTEX_SHADER,VS));
  gl.attachShader(prog, sh(gl.FRAGMENT_SHADER,FS));
  gl.linkProgram(prog); gl.useProgram(prog);
  gl.bindBuffer(gl.ARRAY_BUFFER, gl.createBuffer());
  gl.bufferData(gl.ARRAY_BUFFER,
    new Float32Array([-1,-1,1,-1,-1,1,1,1]), gl.STATIC_DRAW);
  const aPos = gl.getAttribLocation(prog,'aPos');
  gl.enableVertexAttribArray(aPos);
  gl.vertexAttribPointer(aPos,2,gl.FLOAT,false,0,0);
  const U = {};
  ['uData','uCmap','uMercMin','uMercMax','uSite','uEl','uR0','uDr','uNgates',
   'uNaz','uMaxR','uOpacity'].forEach(n=>U[n]=gl.getUniformLocation(prog,n));
  const px = new Uint8Array(cb.stops.length*4);
  cb.stops.forEach((hex,i)=>{px[i*4]=parseInt(hex.slice(1,3),16);
    px[i*4+1]=parseInt(hex.slice(3,5),16);
    px[i*4+2]=parseInt(hex.slice(5,7),16); px[i*4+3]=255;});
  gl.activeTexture(gl.TEXTURE1);
  gl.bindTexture(gl.TEXTURE_2D, gl.createTexture());
  gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA,cb.stops.length,1,0,gl.RGBA,
                gl.UNSIGNED_BYTE,px);
  ['TEXTURE_MIN_FILTER','TEXTURE_MAG_FILTER'].forEach(p=>
    gl.texParameteri(gl.TEXTURE_2D, gl[p], gl.LINEAR));
  ['TEXTURE_WRAP_S','TEXTURE_WRAP_T'].forEach(p=>
    gl.texParameteri(gl.TEXTURE_2D, gl[p], gl.CLAMP_TO_EDGE));
  const cache = {};
  async function ensureTex(f, i){
    if (cache[i]) return;
    const im = await loadImg('data:image/png;base64,'+f.img);
    const t = gl.createTexture();
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D,t);
    gl.texImage2D(gl.TEXTURE_2D,0,gl.LUMINANCE,gl.LUMINANCE,
                  gl.UNSIGNED_BYTE,im);
    ['TEXTURE_MIN_FILTER','TEXTURE_MAG_FILTER'].forEach(p=>
      gl.texParameteri(gl.TEXTURE_2D, gl[p], gl.NEAREST));
    ['TEXTURE_WRAP_S','TEXTURE_WRAP_T'].forEach(p=>
      gl.texParameteri(gl.TEXTURE_2D, gl[p], gl.CLAMP_TO_EDGE));
    cache[i] = t;
  }
  async function preload(frames){
    for (let i=0;i<frames.length;i++) await ensureTex(frames[i], i);
  }
  async function draw(f, i, win){
    await ensureTex(f, i);
    gl.viewport(0,0,w,h);
    gl.clearColor(0,0,0,0); gl.clear(gl.COLOR_BUFFER_BIT);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D,cache[i]);
    gl.uniform1i(U.uData,0); gl.uniform1i(U.uCmap,1);
    gl.uniform2f(U.uMercMin,win.x0,win.y0);
    gl.uniform2f(U.uMercMax,win.x1,win.y1);
    gl.uniform2f(U.uSite,SITE[1]*Math.PI/180,SITE[0]*Math.PI/180);
    gl.uniform1f(U.uEl,f.el*Math.PI/180);
    gl.uniform1f(U.uR0,f.r0); gl.uniform1f(U.uDr,f.dr);
    gl.uniform1f(U.uNgates,f.ngates); gl.uniform1f(U.uNaz,f.naz);
    gl.uniform1f(U.uMaxR,f.maxr); gl.uniform1f(U.uOpacity,1.0);
    gl.enable(gl.BLEND); gl.blendFunc(gl.SRC_ALPHA,gl.ONE_MINUS_SRC_ALPHA);
    gl.drawArrays(gl.TRIANGLE_STRIP,0,4);
  }
  return {cnv, draw, preload};
}

function ellipsize(ctx, text, maxw){
  if (ctx.measureText(text).width <= maxw) return text;
  while (text.length > 1 &&
         ctx.measureText(text+'\\u2026').width > maxw)
    text = text.slice(0, -1);
  return text+'\\u2026';
}

function drawCbarTicks(ctx, cx, cy, cw, chh, cb, fs){
  // interior tick marks + labels every cb.tick, thinned to the bar width
  if (!cb.tick || cb.vmax <= cb.vmin) return;
  const span = cb.vmax-cb.vmin;
  const k0 = Math.ceil((cb.vmin+span*1e-6)/cb.tick),
        k1 = Math.floor((cb.vmax-span*1e-6)/cb.tick);
  const vals = [];
  for (let k = k0; k <= k1; k++){
    const v = k*cb.tick, x = (v-cb.vmin)/span;
    if (x > 0.03 && x < 0.97) vals.push(v);
  }
  if (!vals.length) return;
  const stride = Math.max(1, Math.ceil((vals.length*fs*2.4)/cw));
  ctx.save();
  ctx.font = "600 "+(fs*0.72)+"px 'Source Sans 3', sans-serif";
  ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
  for (let i = 0; i < vals.length; i++){
    const v = vals[i];
    const px = cx + cw*(v-cb.vmin)/span;
    ctx.strokeStyle = 'rgba(255,255,255,.9)';
    ctx.lineWidth = Math.max(1, fs*0.07);
    ctx.beginPath(); ctx.moveTo(px, cy); ctx.lineTo(px, cy+chh*0.5);
    ctx.stroke();
    if (i % stride) continue;
    const x = (v-cb.vmin)/span;
    if (x < 0.08 || x > 0.92) continue;   // keep clear of vmin/vmax labels
    ctx.fillStyle = '#fff';
    ctx.fillText(String(+v.toFixed(2)), px, cy+chh+fs*0.62);
  }
  ctx.restore();
}

const LOGO_IMG = new (window.Image)();
LOGO_IMG.src = 'data:image/png;base64,__LOGOB64__';
function rrPath(ctx, x, y, w, h, r){
  ctx.beginPath();
  ctx.moveTo(x+r, y);
  ctx.arcTo(x+w, y, x+w, y+h, r);
  ctx.arcTo(x+w, y+h, x, y+h, r);
  ctx.arcTo(x, y+h, x, y, r);
  ctx.arcTo(x, y, x+w, y, r);
  ctx.closePath();
}
function drawLogo(ctx, x, y, s){
  // the actual CliMAS logo in a rounded-rectangle badge
  if (!LOGO_IMG.complete || !LOGO_IMG.naturalWidth) return;
  ctx.save();
  rrPath(ctx, x, y, s, s, s*0.18);
  ctx.clip();
  ctx.drawImage(LOGO_IMG, x, y, s, s);
  ctx.restore();
  ctx.save();
  rrPath(ctx, x, y, s, s, s*0.18);
  ctx.lineWidth = Math.max(2, s*0.012);
  ctx.strokeStyle = 'rgba(255,255,255,.85)';
  ctx.stroke();
  ctx.restore();
}

function drawChrome(ctx, w, h, text, cb){
  // scale all chrome off the SHORT edge so portrait/landscape/4K all keep
  // the same visual proportions (no oversized text on reels)
  const s = Math.min(w, h);
  const bh = Math.max(44, Math.round(s*0.066));
  const pad = Math.round(s*0.018);
  ctx.fillStyle = '#13294B'; ctx.fillRect(0, h-bh, w, bh);
  ctx.fillStyle = '#FF5F05'; ctx.fillRect(0, h-bh, w, Math.max(2, bh*0.07));
  const fs = Math.round(bh*0.34);
  // colorbar zone reserved at right (skipped when cb is null - quad export
  // draws one mini colorbar per quadrant instead)
  const cw = Math.round(Math.min(s*0.27, w*0.35)),
        chh = Math.max(8, Math.round(bh*0.26)),
        cx = w-cw-pad, cy = h-bh*0.72;
  const maxText = (cb ? cx : w) - 2*pad;
  ctx.textBaseline = 'middle'; ctx.textAlign = 'left';
  ctx.fillStyle = '#fff';
  ctx.font = "600 "+fs+"px 'Source Sans 3', sans-serif";
  ctx.fillText(ellipsize(ctx, text, maxText), pad, h-bh*0.62);
  ctx.fillStyle = '#C8C6C7';
  ctx.font = (fs*0.8)+"px 'Source Sans 3', sans-serif";
  ctx.fillText(ellipsize(ctx,
    'CliMAS \\u00b7 Illinois \\u00b7 NEXRAD Level 2 browser', maxText),
    pad, h-bh*0.22);
  if (cb){
    const g = ctx.createLinearGradient(cx,0,cx+cw,0);
    cb.stops.forEach((c,i)=>g.addColorStop(i/(cb.stops.length-1), c));
    ctx.fillStyle = g; ctx.fillRect(cx, cy, cw, chh);
    ctx.fillStyle = '#fff'; ctx.font = (fs*0.75)+"px 'Source Sans 3'";
    ctx.textAlign = 'left';  ctx.fillText(cb.vmin, cx, cy+chh+fs*0.62);
    ctx.textAlign = 'right'; ctx.fillText(cb.vmax, cx+cw, cy+chh+fs*0.62);
    drawCbarTicks(ctx, cx, cy, cw, chh, cb, fs);
    ctx.textAlign = 'center';
    ctx.fillText(ellipsize(ctx, cb.label, cw+2*pad), cx+cw/2, cy-fs*0.55);
    ctx.textAlign = 'left';
  }
  return bh;
}

function dlBlob(blob, name){
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob); a.download = name;
  document.body.appendChild(a); a.click(); a.remove();
  setTimeout(()=>URL.revokeObjectURL(a.href), 30000);
}

async function exportQuad(kind, w, h){
  // 2x2 composite: same geographic window in all four quadrants, each
  // field with its own colormap, chip and mini colorbar; shared chrome.
  if (!panels.some(p=>p.fd.frames.length)){ exToast('No data'); return; }
  const s = Math.min(w, h);
  const bh = Math.max(44, Math.round(s*0.066));   // matches drawChrome
  const gap = Math.max(2, Math.round(s*0.004));
  const qw = Math.floor((w-gap)/2), qh = Math.floor((h-bh-gap)/2);

  const map = panels[0].map, sz = map.getSize();
  const cm = PROJ.project(map.getCenter());
  const e1 = PROJ.project(map.containerPointToLatLng([0, sz.y/2]));
  const e2 = PROJ.project(map.containerPointToLatLng([sz.x, sz.y/2]));
  const mercH = Math.abs(e2.x-e1.x) * (sz.y/sz.x);
  const mercW = mercH * (qw/qh);
  const win = {x0: cm.x-mercW/2, x1: cm.x+mercW/2,
               y0: cm.y-mercH/2, y1: cm.y+mercH/2};

  exToast('Rendering basemap\\u2026', true);
  const base = document.createElement('canvas');
  base.width = qw; base.height = qh;
  await drawTiles(base.getContext('2d'),
    'https://{s}.basemaps.cartocdn.com/dark_nolabels/{z}/{x}/{y}@2x.png',
    win, qw, ['a','b','c','d']);
  const refs = document.createElement('canvas');
  refs.width = qw; refs.height = qh;
  if (document.getElementById('ck-interstates').checked)
    await drawTiles(refs.getContext('2d'),
      'https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Transportation/MapServer/tile/{z}/{y}/{x}',
      win, qw, null);
  if (document.getElementById('ck-counties').checked)
    drawCounties(refs.getContext('2d'), win, qw);
  drawCitiesExport(refs.getContext('2d'), win, qw, qh);
  drawWarningsExport(refs.getContext('2d'), win, qw);

  const xgls = panels.map(p=>p.fd.frames.length
    ? makeExportGL(qw, qh, p.fd.cbar) : null);
  const work = document.createElement('canvas');
  work.width = w; work.height = h;
  const wctx = work.getContext('2d');
  const out = document.createElement('canvas');
  out.width = w; out.height = h;
  const octx = out.getContext('2d');
  const logoS = Math.round(s*0.13), pad = Math.round(s*0.018);

  function quadChrome(ox, oy, p){
    const qs = Math.min(qw, qh);
    const fs = Math.max(11, Math.round(qs*0.034));
    // field chip, top-left
    wctx.save();
    wctx.font = "600 "+fs+"px 'Source Sans 3', sans-serif";
    wctx.textBaseline = 'middle';
    const name = ellipsize(wctx, p.fd.name, qw*0.6);
    const tw = wctx.measureText(name).width;
    rrPath(wctx, ox+fs*0.7, oy+fs*0.7, tw+fs*1.4, fs*1.9, fs*0.5);
    wctx.fillStyle = 'rgba(19,41,75,.88)'; wctx.fill();
    wctx.fillStyle = '#fff'; wctx.textAlign = 'left';
    wctx.fillText(name, ox+fs*1.4, oy+fs*0.7+fs*0.95);
    // mini colorbar, bottom-right
    const cb = p.fd.cbar;
    const cw2 = Math.round(qw*0.30), chh = Math.max(6, Math.round(fs*0.55)),
          cx = ox+qw-cw2-fs, cy = oy+qh-chh-fs*1.7;
    const g = wctx.createLinearGradient(cx,0,cx+cw2,0);
    cb.stops.forEach((c,i)=>g.addColorStop(i/(cb.stops.length-1), c));
    rrPath(wctx, cx-fs*0.5, cy-fs*1.5, cw2+fs, chh+fs*2.65, fs*0.35);
    wctx.fillStyle = 'rgba(7,16,32,.55)'; wctx.fill();
    wctx.fillStyle = g; wctx.fillRect(cx, cy, cw2, chh);
    wctx.fillStyle = '#fff';
    wctx.font = (fs*0.8)+"px 'Source Sans 3', sans-serif";
    wctx.textAlign = 'left';  wctx.fillText(cb.vmin, cx, cy+chh+fs*0.55);
    wctx.textAlign = 'right'; wctx.fillText(cb.vmax, cx+cw2, cy+chh+fs*0.55);
    if (cb.units){ wctx.textAlign = 'center';
      wctx.fillText(cb.units, cx+cw2/2, cy-fs*0.55); }
    drawCbarTicks(wctx, cx, cy, cw2, chh, cb, fs*0.9);
    wctx.textAlign = 'left';
    wctx.restore();
  }
  function shortLabel(f){
    const parts = f.label.split('\\u2022').map(x=>x.trim());
    let t = SITE_ID+' \\u00b7 '+parts[0]+(parts[1]?' \\u00b7 '+parts[1]:'');
    if (parts[2] && parts[2].indexOf('VCP') === 0) t += ' \\u00b7 '+parts[2];
    return t;
  }
  async function compose(i){
    wctx.fillStyle = '#071020'; wctx.fillRect(0,0,w,h);
    let lab = null;
    for (let q=0; q<4; q++){
      const p = panels[q];
      const ox = (q%2)*(qw+gap), oy = Math.floor(q/2)*(qh+gap);
      wctx.drawImage(base, ox, oy);
      if (p.fd.frames.length){
        const fi = Math.min(i, p.fd.frames.length-1);
        const f = p.fd.frames[fi];
        if (!lab) lab = f;
        await xgls[q].draw(f, fi, win);
        wctx.drawImage(xgls[q].cnv, ox, oy);
      }
      wctx.drawImage(refs, ox, oy);
      if (p.fd.frames.length) quadChrome(ox, oy, p);
      else {
        const qs = Math.min(qw, qh), fs2 = Math.round(qs*0.04);
        wctx.font = "600 "+fs2+"px 'Source Sans 3', sans-serif";
        wctx.textAlign = 'center'; wctx.textBaseline = 'middle';
        wctx.lineWidth = fs2*0.25; wctx.strokeStyle = 'rgba(7,16,32,.85)';
        wctx.strokeText('Pre-polarimetric upgrade data', ox+qw/2, oy+qh/2);
        wctx.fillStyle = '#C8C6C7';
        wctx.fillText('Pre-polarimetric upgrade data', ox+qw/2, oy+qh/2);
        wctx.textAlign = 'left';
      }
    }
    drawChrome(wctx, w, h, lab ? shortLabel(lab) : SITE_ID, null);
    drawLogo(wctx, pad, h-bh-logoS-pad, logoS);
    octx.clearRect(0,0,w,h);
    octx.drawImage(work,0,0);
  }

  const act = panels.find(p=>p.fd.frames.length);
  const f0 = act.fd.frames[0];
  const stamp = f0.label.slice(0,10).replace(/-/g,'') + '_' +
                f0.label.slice(11,16).split(':').join('') + 'Z';
  const base_name = SITE_ID+'_quad_'+stamp+'_'+w+'x'+h;

  if (kind === 'still'){
    await compose(Math.max(0, idx));
    out.toBlob(b=>{ dlBlob(b, base_name+'.png');
                    exToast('Image saved'); }, 'image/png');
    return;
  }
  // prefer High-profile H.264 at a level that actually supports the output
  // size (4.0 tops out below 4K; 5.2 covers 3840x2160), then fall back
  const mime = ['video/mp4;codecs=avc1.640034','video/mp4;codecs=avc1.640033',
                'video/mp4;codecs=avc1.640028','video/mp4',
                'video/webm;codecs=vp9','video/webm']
    .find(m=>window.MediaRecorder && MediaRecorder.isTypeSupported(m));
  if (!mime){ exToast('Video recording unsupported in this browser'); return; }
  const ext = mime.indexOf('mp4')>=0 ? '.mp4' : '.webm';
  exToast('Preparing radar textures\\u2026', true);
  for (let q=0; q<4; q++)
    if (xgls[q]) await xgls[q].preload(panels[q].fd.frames);
  await compose(0);
  const stream = out.captureStream(30);
  // maximum quality: ~0.5 bits/pixel @30fps (4K ~120 Mbps, reel ~31 Mbps)
  const rec = new MediaRecorder(stream,
    {mimeType:mime, videoBitsPerSecond: Math.round(w*h*30*0.5)});
  const chunks = [];
  rec.ondataavailable = e=>{ if (e.data.size) chunks.push(e.data); };
  const done = new Promise(res=>{ rec.onstop = res; });
  rec.start(250);
  for (let i=0; i<nmax; i++){
    exToast('Recording frame '+(i+1)+'/'+nmax+'\\u2026', true);
    await compose(i);
    await new Promise(r=>setTimeout(r,
      FRAME_MS * (i === nmax-1 ? END_DWELL : 1)));   // dwell on last frame
  }
  rec.stop();
  await done;
  dlBlob(new Blob(chunks, {type:mime}), base_name+ext);
  exToast('Movie saved ('+ext.slice(1)+')');
}

async function exportZV(kind, w, h){
  // 2-panel composite. Landscape (4K) -> reflectivity LEFT, velocity RIGHT;
  // portrait (reel) -> reflectivity TOP, velocity BOTTOM. The velocity panel
  // exports whatever it currently shows (raw or dealiased), so the dealiased
  // state is preserved.
  const zp = panels.find(p=>p.name === 'Reflectivity');
  const vp = panels.find(p=>p.name === 'Radial velocity');
  if (!zp || !vp){ exToast('Z+V needs reflectivity and velocity'); return; }
  const cells = [zp, vp];
  if (!cells.some(p=>p.fd.frames.length)){ exToast('No data'); return; }
  const s = Math.min(w, h);
  const bh = Math.max(44, Math.round(s*0.066));   // matches drawChrome
  const gap = Math.max(2, Math.round(s*0.004));
  const landscape = w >= h;
  let qw, qh, pos;
  if (landscape){                       // side by side
    qw = Math.floor((w-gap)/2); qh = h - bh;
    pos = [[0,0],[qw+gap,0]];
  } else {                              // stacked
    qw = w; qh = Math.floor((h-bh-gap)/2);
    pos = [[0,0],[0,qh+gap]];
  }

  const map = zp.map, sz = map.getSize();
  const cm = PROJ.project(map.getCenter());
  const e1 = PROJ.project(map.containerPointToLatLng([0, sz.y/2]));
  const e2 = PROJ.project(map.containerPointToLatLng([sz.x, sz.y/2]));
  const mercH = Math.abs(e2.x-e1.x) * (sz.y/sz.x);
  const mercW = mercH * (qw/qh);
  const win = {x0: cm.x-mercW/2, x1: cm.x+mercW/2,
               y0: cm.y-mercH/2, y1: cm.y+mercH/2};

  exToast('Rendering basemap\\u2026', true);
  const base = document.createElement('canvas');
  base.width = qw; base.height = qh;
  await drawTiles(base.getContext('2d'),
    'https://{s}.basemaps.cartocdn.com/dark_nolabels/{z}/{x}/{y}@2x.png',
    win, qw, ['a','b','c','d']);
  const refs = document.createElement('canvas');
  refs.width = qw; refs.height = qh;
  if (document.getElementById('ck-interstates').checked)
    await drawTiles(refs.getContext('2d'),
      'https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Transportation/MapServer/tile/{z}/{y}/{x}',
      win, qw, null);
  if (document.getElementById('ck-counties').checked)
    drawCounties(refs.getContext('2d'), win, qw);
  drawCitiesExport(refs.getContext('2d'), win, qw, qh);
  drawWarningsExport(refs.getContext('2d'), win, qw);

  const xgls = cells.map(p=>p.fd.frames.length
    ? makeExportGL(qw, qh, p.fd.cbar) : null);
  const work = document.createElement('canvas');
  work.width = w; work.height = h;
  const wctx = work.getContext('2d');
  const out = document.createElement('canvas');
  out.width = w; out.height = h;
  const octx = out.getContext('2d');
  const logoS = Math.round(s*0.13), pad = Math.round(s*0.018);

  function panelChrome(ox, oy, p){
    const qs = Math.min(qw, qh);
    const fs = Math.max(11, Math.round(qs*0.034));
    wctx.save();
    wctx.font = "600 "+fs+"px 'Source Sans 3', sans-serif";
    wctx.textBaseline = 'middle';
    const name = ellipsize(wctx, p.fd.name, qw*0.6);
    const tw = wctx.measureText(name).width;
    rrPath(wctx, ox+fs*0.7, oy+fs*0.7, tw+fs*1.4, fs*1.9, fs*0.5);
    wctx.fillStyle = 'rgba(19,41,75,.88)'; wctx.fill();
    wctx.fillStyle = '#fff'; wctx.textAlign = 'left';
    wctx.fillText(name, ox+fs*1.4, oy+fs*0.7+fs*0.95);
    const cb = p.fd.cbar;
    const cw2 = Math.round(qw*0.30), chh = Math.max(6, Math.round(fs*0.55)),
          cx = ox+qw-cw2-fs, cy = oy+qh-chh-fs*1.7;
    const g = wctx.createLinearGradient(cx,0,cx+cw2,0);
    cb.stops.forEach((c,i)=>g.addColorStop(i/(cb.stops.length-1), c));
    rrPath(wctx, cx-fs*0.5, cy-fs*1.5, cw2+fs, chh+fs*2.65, fs*0.35);
    wctx.fillStyle = 'rgba(7,16,32,.55)'; wctx.fill();
    wctx.fillStyle = g; wctx.fillRect(cx, cy, cw2, chh);
    wctx.fillStyle = '#fff';
    wctx.font = (fs*0.8)+"px 'Source Sans 3', sans-serif";
    wctx.textAlign = 'left';  wctx.fillText(cb.vmin, cx, cy+chh+fs*0.55);
    wctx.textAlign = 'right'; wctx.fillText(cb.vmax, cx+cw2, cy+chh+fs*0.55);
    if (cb.units){ wctx.textAlign = 'center';
      wctx.fillText(cb.units, cx+cw2/2, cy-fs*0.55); }
    drawCbarTicks(wctx, cx, cy, cw2, chh, cb, fs*0.9);
    wctx.textAlign = 'left';
    wctx.restore();
  }
  function shortLabel(f){
    const parts = f.label.split('\\u2022').map(x=>x.trim());
    let t = SITE_ID+' \\u00b7 '+parts[0]+(parts[1]?' \\u00b7 '+parts[1]:'');
    if (parts[2] && parts[2].indexOf('VCP') === 0) t += ' \\u00b7 '+parts[2];
    return t;
  }
  async function compose(i){
    wctx.fillStyle = '#071020'; wctx.fillRect(0,0,w,h);
    let lab = null;
    for (let q=0; q<cells.length; q++){
      const p = cells[q];
      const ox = pos[q][0], oy = pos[q][1];
      wctx.drawImage(base, ox, oy);
      if (p.fd.frames.length){
        const fi = Math.min(i, p.fd.frames.length-1);
        const f = p.fd.frames[fi];
        if (!lab) lab = f;
        await xgls[q].draw(f, fi, win);
        wctx.drawImage(xgls[q].cnv, ox, oy);
      }
      wctx.drawImage(refs, ox, oy);
      if (p.fd.frames.length) panelChrome(ox, oy, p);
      else {
        const qs = Math.min(qw, qh), fs2 = Math.round(qs*0.04);
        wctx.font = "600 "+fs2+"px 'Source Sans 3', sans-serif";
        wctx.textAlign = 'center'; wctx.textBaseline = 'middle';
        wctx.lineWidth = fs2*0.25; wctx.strokeStyle = 'rgba(7,16,32,.85)';
        wctx.strokeText('No data for this hour', ox+qw/2, oy+qh/2);
        wctx.fillStyle = '#C8C6C7';
        wctx.fillText('No data for this hour', ox+qw/2, oy+qh/2);
        wctx.textAlign = 'left';
      }
    }
    drawChrome(wctx, w, h, lab ? shortLabel(lab) : SITE_ID, null);
    drawLogo(wctx, pad, h-bh-logoS-pad, logoS);
    octx.clearRect(0,0,w,h);
    octx.drawImage(work,0,0);
  }

  const act = cells.find(p=>p.fd.frames.length);
  const f0 = act.fd.frames[0];
  const stamp = f0.label.slice(0,10).replace(/-/g,'') + '_' +
                f0.label.slice(11,16).split(':').join('') + 'Z';
  const base_name = SITE_ID+'_zv_'+stamp+'_'+w+'x'+h;

  if (kind === 'still'){
    await compose(Math.max(0, idx));
    out.toBlob(b=>{ dlBlob(b, base_name+'.png');
                    exToast('Image saved'); }, 'image/png');
    return;
  }
  const mime = ['video/mp4;codecs=avc1.640034','video/mp4;codecs=avc1.640033',
                'video/mp4;codecs=avc1.640028','video/mp4',
                'video/webm;codecs=vp9','video/webm']
    .find(m=>window.MediaRecorder && MediaRecorder.isTypeSupported(m));
  if (!mime){ exToast('Video recording unsupported in this browser'); return; }
  const ext = mime.indexOf('mp4')>=0 ? '.mp4' : '.webm';
  exToast('Preparing radar textures\\u2026', true);
  for (let q=0; q<cells.length; q++)
    if (xgls[q]) await xgls[q].preload(cells[q].fd.frames);
  await compose(0);
  const stream = out.captureStream(30);
  const rec = new MediaRecorder(stream,
    {mimeType:mime, videoBitsPerSecond: Math.round(w*h*30*0.5)});
  const chunks = [];
  rec.ondataavailable = e=>{ if (e.data.size) chunks.push(e.data); };
  const done = new Promise(res=>{ rec.onstop = res; });
  rec.start(250);
  for (let i=0; i<nmax; i++){
    exToast('Recording frame '+(i+1)+'/'+nmax+'\\u2026', true);
    await compose(i);
    await new Promise(r=>setTimeout(r,
      FRAME_MS * (i === nmax-1 ? END_DWELL : 1)));   // dwell on last frame
  }
  rec.stop();
  await done;
  dlBlob(new Blob(chunks, {type:mime}), base_name+ext);
  exToast('Movie saved ('+ext.slice(1)+')');
}

async function exportMedia(kind, w, h){
  if (mode === 'quad') return exportQuad(kind, w, h);
  if (mode === 'zv') return exportZV(kind, w, h);
  const vis = activePanel(), fd = vis.fd;
  if (!fd.frames.length) { exToast('No data in this panel'); return; }
  const map = vis.map, sz = map.getSize();
  const cm = PROJ.project(map.getCenter());
  const e1 = PROJ.project(map.containerPointToLatLng([0, sz.y/2]));
  const e2 = PROJ.project(map.containerPointToLatLng([sz.x, sz.y/2]));
  const mercH = Math.abs(e2.x-e1.x) * (sz.y/sz.x);
  const mercW = mercH * (w/h);
  const win = {x0: cm.x-mercW/2, x1: cm.x+mercW/2,
               y0: cm.y-mercH/2, y1: cm.y+mercH/2};

  exToast('Rendering basemap\\u2026', true);
  const base = document.createElement('canvas');
  base.width = w; base.height = h;
  await drawTiles(base.getContext('2d'),
    'https://{s}.basemaps.cartocdn.com/dark_nolabels/{z}/{x}/{y}@2x.png',
    win, w, ['a','b','c','d']);

  const refs = document.createElement('canvas');
  refs.width = w; refs.height = h;
  if (document.getElementById('ck-interstates').checked)
    await drawTiles(refs.getContext('2d'),
      'https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Transportation/MapServer/tile/{z}/{y}/{x}',
      win, w, null);
  if (document.getElementById('ck-counties').checked)
    drawCounties(refs.getContext('2d'), win, w);
  drawCitiesExport(refs.getContext('2d'), win, w, h);
  drawWarningsExport(refs.getContext('2d'), win, w);

  const xgl = makeExportGL(w, h, fd.cbar);
  // double buffer: compose on `work`, blit complete frames to `out` —
  // the recorder only ever sees finished composites
  const work = document.createElement('canvas');
  work.width = w; work.height = h;
  const wctx = work.getContext('2d');
  const out = document.createElement('canvas');
  out.width = w; out.height = h;
  const octx = out.getContext('2d');
  const logoS = Math.round(Math.min(w,h)*0.15),
        pad = Math.round(Math.min(w,h)*0.018);

  function shortLabel(f){
    const parts = f.label.split('\\u2022').map(s=>s.trim());
    let t = SITE_ID+' \\u00b7 '+parts[0]+(parts[1]?' \\u00b7 '+parts[1]:'');
    if (parts[2] && parts[2].indexOf('VCP') === 0)
      t += ' \\u00b7 '+parts[2];
    return t;
  }
  async function compose(i){
    const fi = Math.min(i, fd.frames.length-1);
    const f = fd.frames[fi];
    await xgl.draw(f, fi, win);
    wctx.clearRect(0,0,w,h);
    wctx.drawImage(base,0,0);
    wctx.drawImage(xgl.cnv,0,0);
    wctx.drawImage(refs,0,0);
    const bh = drawChrome(wctx, w, h, shortLabel(f), fd.cbar);
    drawLogo(wctx, pad, h-bh-logoS-pad, logoS);
    octx.clearRect(0,0,w,h);
    octx.drawImage(work,0,0);
  }

  const f0 = fd.frames[0];
  const stamp = f0.label.slice(0,10).replace(/-/g,'') + '_' +
                f0.label.slice(11,16).split(':').join('') + 'Z';
  const fldShort = (SHORT[fd.name]||fd.name).toLowerCase();
  const base_name = SITE_ID+'_'+fldShort+'_'+stamp+'_'+w+'x'+h;

  if (kind === 'still'){
    await compose(Math.max(0, idx));
    out.toBlob(b=>{ dlBlob(b, base_name+'.png');
                    exToast('Image saved'); }, 'image/png');
    return;
  }
  // loop -> movie via MediaRecorder; pre-decode every radar texture first
  // prefer High-profile H.264 at a level that actually supports the output
  // size (4.0 tops out below 4K; 5.2 covers 3840x2160), then fall back
  const mime = ['video/mp4;codecs=avc1.640034','video/mp4;codecs=avc1.640033',
                'video/mp4;codecs=avc1.640028','video/mp4',
                'video/webm;codecs=vp9','video/webm']
    .find(m=>window.MediaRecorder && MediaRecorder.isTypeSupported(m));
  if (!mime){ exToast('Video recording unsupported in this browser'); return; }
  const ext = mime.indexOf('mp4')>=0 ? '.mp4' : '.webm';
  exToast('Preparing radar textures\\u2026', true);
  await xgl.preload(fd.frames);
  await compose(0);                       // first full frame before recording
  const stream = out.captureStream(30);
  // maximum quality: ~0.5 bits/pixel @30fps (4K ~120 Mbps, reel ~31 Mbps)
  const rec = new MediaRecorder(stream,
    {mimeType:mime, videoBitsPerSecond: Math.round(w*h*30*0.5)});
  const chunks = [];
  rec.ondataavailable = e=>{ if (e.data.size) chunks.push(e.data); };
  const done = new Promise(res=>{ rec.onstop = res; });
  rec.start(250);
  const n = fd.frames.length;
  for (let i=0; i<n; i++){
    exToast('Recording frame '+(i+1)+'/'+n+'\\u2026', true);
    await compose(i);
    await new Promise(r=>setTimeout(r,
      FRAME_MS * (i === n-1 ? END_DWELL : 1)));   // dwell on last frame
  }
  rec.stop();
  await done;
  dlBlob(new Blob(chunks, {type:mime}), base_name+ext);
  exToast('Movie saved ('+ext.slice(1)+')');
}

const EXMENU = document.getElementById('exmenu');
document.getElementById('export').addEventListener('click', ()=>{
  EXMENU.style.display = EXMENU.style.display==='flex' ? 'none' : 'flex';
});
EXMENU.querySelectorAll('button').forEach(b=>b.addEventListener('click', ()=>{
  EXMENU.style.display = 'none';
  const cfg = {still4k:['still',3840,2160], stillig:['still',1080,1920],
               loop4k:['loop',3840,2160],  loopig:['loop',1080,1920]}[b.dataset.k];
  exportMedia(cfg[0], cfg[1], cfg[2])
    .catch(e=>exToast('Export failed: '+e.message));
}));

// stagger texture preloads (newest-first in live mode)
panels.forEach((p,pi)=>p.fd.frames.forEach((_,i)=>
  setTimeout(()=>p.texFor(i,()=>{}),
             80*(RT ? p.fd.frames.length-1-i : i)+20*pi)));
// live view opens on the newest frame; archive starts at the top of the hour —
// unless we just rebuilt for a dealias toggle, then restore the prior frame
let _startIdx = RT ? nmax-1 : 0;
try {
  if (parent && typeof parent.__restoreIdx === 'number'){
    _startIdx = Math.min(Math.max(parent.__restoreIdx, 0), nmax-1);
    parent.__restoreIdx = undefined;
  }
} catch (e) {}
show(_startIdx);
</script></body></html>"""


# small base64 of the CliMAS logo for export stamping (from logo.png)
def _export_logo_b64():
    p = os.path.join(os.path.dirname(__file__), "logo.png")
    if not os.path.exists(p):
        return ""
    try:
        im = Image.open(p).convert("RGB").resize((512, 512), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="PNG", optimize=True)
        return base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return ""


EXPORT_LOGO_B64 = _export_logo_b64()


def _bundle_data(by_field):
    """The per-field DATA payload (textures + colorbar) the bundle renders."""
    return [dict(name=fn, cbar=colorbar_cfg(_field_cfg(fn)),
                 frames=[{k: f[k] for k in ("img", "naz", "ngates", "r0",
                                            "dr", "el", "maxr", "label")}
                         for f in frames])
            for fn, frames in by_field.items()]


def _live_sig(data):
    """Cheap content signature: changes whenever any frame is added/updated."""
    parts = []
    for d in data:
        parts.append(d["name"] + ":" + str(len(d["frames"])))
        parts.append(",".join(str(fr["label"]) for fr in d["frames"]))
    return hashlib.md5("|".join(parts).encode()).hexdigest()[:12]


def _live_paths(site):
    s = site.upper()
    return (os.path.join(VOL_CACHE_DIR, f"live_{s}.json"),
            os.path.join(VOL_CACHE_DIR, f"live_{s}.tag.json"))


def _write_live(site, by_field):
    """Write the live frame set (+ a tiny etag-only sidecar) so the in-page
    poller can merge new frames WITHOUT reloading the whole iframe."""
    try:
        data = _bundle_data(by_field)
        etag = _live_sig(data)
        ts = time_mod.time()   # bumps every compute, even when etag is unchanged
        big, tag = _live_paths(site)
        for path, payload in ((big, dict(etag=etag, ts=ts, fields=data)),
                              (tag, dict(etag=etag, ts=ts))):
            tmp = path + ".part"
            with open(tmp, "w") as f:
                json.dump(payload, f)
            os.replace(tmp, path)
        return etag
    except Exception:
        return ""


def build_bundle_page(by_field, site, slat, slon, share_base="",
                      rt_refresh_s=300):
    """One page carrying ALL fields' polar textures; the requested view
    (single field or 2×2) is substituted into __MODE__ at serve time, so
    every field/4-panel switch after the first load is instant."""
    data = _bundle_data(by_field)
    etag = _live_sig(data)
    big, tag = _live_paths(site)
    page = (QUAD_PAGE
            .replace("__DATA__", json.dumps(data))
            .replace("__SLAT__", f"{slat:.5f}")
            .replace("__SLON__", f"{slon:.5f}")
            .replace("__SITE__", site)
            .replace("__WARNURL__", WARN_URL)
            .replace("__CITIESURL__", CITIES_URL)
            .replace("__LIVEURL__", "/gradio_api/file=" + big)
            .replace("__LIVETAGURL__", "/gradio_api/file=" + tag)
            .replace("__LIVETAG0__", json.dumps(etag))
            .replace("__LOGOB64__", EXPORT_LOGO_B64)
            .replace("__RTSEC__", str(int(rt_refresh_s)))
            .replace("__SHAREBASE__", share_base))
    return (f'<iframe allow="clipboard-write" '
            f'style="width:100%;height:calc(100vh - 170px);'
            f'min-height:480px;border:0;border-radius:4px" '
            f'srcdoc="{html_mod.escape(page)}"></iframe>')


def _mode_page(tpl, field_name, view=None, deal=False, rt=False):
    """Substitute the initial view mode, optional [lat, lon, zoom] and the
    dealias flag into a cached bundle template. The template is already
    HTML-escaped (srcdoc), so escape the JSON too."""
    if field_name == QUAD:
        mode = "quad"
    elif field_name == ZVF:
        mode = "zv"
    elif field_name == DEALIAS_NAME:
        mode, deal = "Radial velocity", True
    else:
        mode = field_name
    return (tpl
            .replace("__MODE__", html_mod.escape(json.dumps(mode)))
            .replace("__VIEW__", html_mod.escape(json.dumps(view)))
            .replace("__DEAL__", "true" if deal else "false")
            .replace("__RT__", "true" if rt else "false"))


# ----------------------------------------------------------------------------- gradio callback


# rendered-page cache: default case is pre-rendered at startup, and recent
# views are served instantly (share links, repeat visits)
DEFAULT_VIEW = ("KILX", "Reflectivity", "2023", "06", "29", "18:00")
_PAGE_CACHE = collections.OrderedDict()
_PAGE_CACHE_CAP = 6   # hours; each bundle holds all four fields (~30-50 MB)
_FRAME_CACHE = collections.OrderedDict()   # hour -> raw frames (for dealias)
_FRAME_CACHE_CAP = 4
_INFLIGHT = {}
_CACHE_LOCK = threading.Lock()


def _rt_keys(site):
    """Volumes whose start time lies in the trailing 60 minutes."""
    now = dt.datetime.utcnow()
    bucket, keys = None, []
    for hdt in (now - dt.timedelta(hours=1), now):
        try:
            b, k = list_hour_keys(site, hdt.date(), hdt.hour)
        except Exception:
            b, k = None, []
        if k:
            bucket = bucket or b
            keys += k
    cutoff = now - dt.timedelta(minutes=62)
    sel = []
    for k in keys:
        m = re.search(r"(\d{8})_(\d{6})", k)
        if m:
            t = dt.datetime.strptime(m.group(1) + m.group(2),
                                     "%Y%m%d%H%M%S")
            if t >= cutoff:
                sel.append(k)
    return bucket, sorted(set(sel)), now


_RT_CACHE = {}        # site -> dict(ts, tpl, by_field, site_ll, has_deal)
_RT_TTL = 150         # seconds — refresh cadence is 300 s


def _realtime_compute(site, field_name, _p, view, want_deal=False):
    """Decode the trailing hour. Short-lived cache + in-flight dedup keep
    reloads and concurrent viewers from re-decoding live data. Dealiasing
    is supported: the cached raw decode is upgraded in place."""
    ttl = 45 if site in CHUNK_SITES else _RT_TTL
    c = _RT_CACHE.get(site)
    fresh = c and time_mod.time() - c["ts"] < ttl
    if fresh and (not want_deal or c["has_deal"]):
        return "", _mode_page(c["tpl"], field_name, view,
                              deal=want_deal, rt=True)
    key_rt = ("RT", site)
    with _CACHE_LOCK:
        evt = _INFLIGHT.get(key_rt)
        owner = evt is None
        if owner:
            _INFLIGHT[key_rt] = threading.Event()
    if not owner:
        _p(0.5, "Live hour already rendering — attaching…")
        evt.wait(600)
        c = _RT_CACHE.get(site)
        if c and (not want_deal or c["has_deal"]):
            return "", _mode_page(c["tpl"], field_name, view,
                                  deal=want_deal, rt=True)
    try:
        return _realtime_compute_inner(site, field_name, _p, view,
                                       want_deal)
    finally:
        if owner:
            with _CACHE_LOCK:
                ev = _INFLIGHT.pop(key_rt, None)
            if ev:
                ev.set()


def _realtime_chunks_inner(site, field_name, _p, view, want_deal=False):
    """Live mode on the Level 2 chunk feed: per-volume incremental decode —
    only the in-progress volume (and newly completed ones) cost CPU."""
    _p(0.02, f"Polling the live chunk feed for {site}…")
    vols = fetch_live_chunks(
        site, progress=lambda f, d: _p(0.02 + 0.28 * f, d))
    if not vols:
        raise RuntimeError("no live chunks")
    by_field = {fn: [] for fn in FIELDS}
    site_ll = None
    todo = []
    for v in vols:
        ck = (site, v["vol"])
        cached = _CHUNK_FRAMES.get(ck)
        if (cached and cached["n"] == len(v["paths"])
                and cached["complete"] == v["complete"]):
            continue
        todo.append(v)
    done = 0
    for i in range(0, len(todo), 2):
        grp = todo[i:i + 2]
        with _DECODE_SEM:
            with ProcessPoolExecutor(max_workers=N_PROC) as px:
                futs = {px.submit(
                    process_chunks, v["paths"],
                    "live vol %d%s" % (v["vol"],
                                       "" if v["complete"] else
                                       " (updating)"),
                    FIELDS): v for v in grp}
                for fut in as_completed(futs):
                    v = futs[fut]
                    done += 1
                    _p(0.32 + 0.40 * done / max(1, len(todo)),
                       f"Decoding live volume {v['vol']} "
                       f"({done}/{len(todo)})…")
                    try:
                        frames, ll = fut.result()
                    except Exception:
                        continue
                    _CHUNK_FRAMES[(site, v["vol"])] = dict(
                        n=len(v["paths"]), complete=v["complete"],
                        frames=frames, site_ll=ll)
    for v in vols:
        cached = _CHUNK_FRAMES.get((site, v["vol"]))
        if not cached:
            continue
        site_ll = site_ll or cached.get("site_ll")
        for fn, fr in cached["frames"].items():
            by_field[fn].extend(fr)
    if want_deal:
        comp = [v for v in vols if v["complete"]]
        todo_d = [v for v in comp if (site, v["vol"]) not in _CHUNK_DEAL]
        dn = 0
        for i in range(0, len(todo_d), 2):
            grp = todo_d[i:i + 2]
            with _DECODE_SEM:
                with ProcessPoolExecutor(max_workers=N_PROC) as px:
                    futs = {px.submit(
                        dealias_volume_file,
                        _concat_chunks(v["paths"], os.path.join(
                            CHUNK_DIR, site, str(v["vol"]), "volume.bin")),
                        "live vol %d" % v["vol"]): v for v in grp}
                    for fut in as_completed(futs):
                        v = futs[fut]
                        dn += 1
                        _p(0.74 + 0.18 * dn / max(1, len(todo_d)),
                           f"Dealiasing live volume {v['vol']} "
                           f"({dn}/{len(todo_d)}, {_DEALIAS_ENGINE})…")
                        try:
                            _CHUNK_DEAL[(site, v["vol"])] = fut.result()
                        except Exception:
                            _CHUNK_DEAL[(site, v["vol"])] = []
        dframes = []
        for v in comp:
            dframes.extend(_CHUNK_DEAL.get((site, v["vol"]), []))
        dframes.sort(key=lambda f: f["time"])
        by_field[DEALIAS_NAME] = dframes[:MAX_FRAMES]
    for fn in by_field:
        by_field[fn].sort(key=lambda f: f["time"])
        by_field[fn] = by_field[fn][:MAX_FRAMES]
    # bound the per-volume caches
    for cache, cap in ((_CHUNK_FRAMES, 60), (_CHUNK_DEAL, 40)):
        while len(cache) > cap:
            cache.pop(next(iter(cache)))
    _prune_chunk_cache()
    return _rt_finish(site, field_name, _p, view, want_deal,
                      by_field, site_ll)


def _realtime_compute_inner(site, field_name, _p, view, want_deal=False):
    if site in CHUNK_SITES:
        try:
            return _realtime_chunks_inner(site, field_name, _p, view,
                                          want_deal)
        except Exception:
            pass   # fall back to the archive-bucket live path below
    _p(0.02, f"Listing the last 60 minutes for {site}…")
    bucket, keys, now = _rt_keys(site)
    if not keys:
        return _msg(f"No Level 2 volumes from {site} in the last hour — "
                    f"the radar may be down or data is still in transit.")
    n = len(keys)
    c = _RT_CACHE.get(site)
    if (c and time_mod.time() - c["ts"] < _RT_TTL and want_deal
            and not c["has_deal"]):
        # fresh raw decode exists — only the dealias pass is needed
        by_field = dict(c["by_field"])
        site_ll = c["site_ll"]
    else:
        with ThreadPoolExecutor(max_workers=6) as ex:
            futs = [ex.submit(_safe_download, bucket, k, VOL_CACHE_DIR)
                    for k in keys]
            for i, _ in enumerate(as_completed(futs)):
                _p(0.04 + 0.16 * (i + 1) / n,
                   f"Fetching live volumes… {i + 1}/{n}")
        by_field, site_ll = _decode_keys(bucket, keys, _p, 0.22, 0.50,
                                         "Decoding live volumes")
        _prune_vol_cache()
    if want_deal:
        dframes, done = [], 0
        for i in range(0, n, 2):
            chunk = keys[i:i + 2]
            with _DECODE_SEM:
                with ProcessPoolExecutor(max_workers=N_PROC) as px:
                    pfuts = [px.submit(dealias_volume, bucket, k,
                                       VOL_CACHE_DIR) for k in chunk]
                    for fut in as_completed(pfuts):
                        done += 1
                        _p(0.72 + 0.22 * done / n,
                           f"Dealiasing live volumes {done}/{n} "
                           f"({_DEALIAS_ENGINE})…")
                        dframes.extend(fut.result())
        dframes.sort(key=lambda f: f["time"])
        by_field[DEALIAS_NAME] = dframes[:MAX_FRAMES]
    return _rt_finish(site, field_name, _p, view, want_deal,
                      by_field, site_ll)


def _rt_finish(site, field_name, _p, view, want_deal, by_field, site_ll):
    if site_ll is not None and abs(site_ll[0]) < 0.1 and abs(site_ll[1]) < 0.1:
        site_ll = None
    if site_ll is None:
        try:
            from pyart.io.nexrad_common import NEXRAD_LOCATIONS
            loc = NEXRAD_LOCATIONS.get(site)
            if loc:
                site_ll = (loc["lat"], loc["lon"])
        except Exception:
            pass
    if site_ll is None or not any(by_field.values()):
        return _msg(f"Could not decode any live data from {site}.")
    _p(0.96, "Packing live bundle…")
    rtsec = 120 if site in CHUNK_SITES else 300
    tpl = build_bundle_page(by_field, site, site_ll[0], site_ll[1],
                            f"?site={site}&rt=1", rt_refresh_s=rtsec)
    _write_live(site, by_field)   # refresh the in-place poll file
    _RT_CACHE[site] = dict(ts=time_mod.time(), tpl=tpl, by_field=by_field,
                           site_ll=site_ll,
                           has_deal=DEALIAS_NAME in by_field)
    for k in [k for k in _RT_CACHE if k != site][4:]:
        _RT_CACHE.pop(k, None)
    return "", _mode_page(tpl, field_name, view, deal=want_deal, rt=True)


def browse(site, field_name, year, month, day, hour, progress=None,
           view=None, want_deal=False, realtime=False):
    def _p(frac, desc):
        if progress is not None:
            try:
                progress(frac, desc=desc)
            except Exception:
                pass

    try:
        m = re.match(r"\s*([A-Za-z]{4})\b", site or "")
        if not m:
            return _msg("Pick a NEXRAD site (e.g. KTLX, KILX, PHWA).")
        site = m.group(1).upper()
        if realtime:
            return _realtime_compute(site, field_name, _p, view, want_deal)
        try:
            date = dt.date(int(year), int(month), int(day))
        except ValueError:
            return _msg("That calendar date doesn't exist — check day/month.")
        hr = int(str(hour)[:2])
        if field_name not in FIELDS and field_name not in (QUAD, ZVF,
                                                           DEALIAS_NAME):
            field_name = "Reflectivity"

        # ---- per-hour bundle cache / in-flight dedup ---------------------
        key_h = _hour_key(site, date, hr)
        with _CACHE_LOCK:
            tpl = _PAGE_CACHE.get(key_h)
            if tpl is not None:
                _PAGE_CACHE.move_to_end(key_h)
        if tpl is not None:
            if ((field_name == DEALIAS_NAME or want_deal)
                    and _DEAL_MARKER not in tpl):
                tpl = _dealias_compute(site, date, hr, _p)
            return "", _mode_page(tpl, field_name, view, want_deal)
        with _CACHE_LOCK:
            evt = _INFLIGHT.get(key_h)
            owner = evt is None
            if owner:
                _INFLIGHT[key_h] = threading.Event()
        if not owner:
            _p(0.5, "This hour is already rendering — attaching to it…")
            evt.wait(900)
            with _CACHE_LOCK:
                tpl = _PAGE_CACHE.get(key_h)
            if tpl is not None:
                if field_name == DEALIAS_NAME and _DEAL_MARKER not in tpl:
                    tpl = _dealias_compute(site, date, hr, _p)
                return "", _mode_page(tpl, field_name, view)
            # fall through and compute ourselves if the other run failed

        try:
            err, tpl = _browse_compute(site, field_name, date, hr, _p)
            if err is not None:
                return err
            if ((field_name == DEALIAS_NAME or want_deal)
                    and _DEAL_MARKER not in tpl):
                tpl = _dealias_compute(site, date, hr, _p)
            return "", _mode_page(tpl, field_name, view, want_deal)
        finally:
            if owner:
                with _CACHE_LOCK:
                    ev = _INFLIGHT.pop(key_h, None)
                if ev:
                    ev.set()
    except Exception:
        return _msg("Unexpected error:\n```\n"
                    + traceback.format_exc()[-1500:] + "\n```")


# marker proving the dealiased field is inside a bundle template (must not
# collide with the JS SHORT-name literal, hence the json "name" form)
_DEAL_MARKER = html_mod.escape(json.dumps("name") + ": "
                               + json.dumps(DEALIAS_NAME))


# Only one decode batch may use the CPUs at a time, and work is chunked so
# concurrent requests (boot pre-render vs. a live user) interleave fairly
# instead of thrashing two process pools on two cores.
_DECODE_SEM = threading.Semaphore(1)


def _decode_keys(bucket, keys, _p, base, span, label):
    """Decode volumes (all fields) in semaphore-guarded chunks of 2."""
    by_field = {fn: [] for fn in FIELDS}
    site_ll = None
    n, done = len(keys), 0
    for i in range(0, n, 2):
        chunk = keys[i:i + 2]
        with _DECODE_SEM:
            with ProcessPoolExecutor(max_workers=N_PROC) as px:
                futs = [px.submit(process_volume, bucket, k, FIELDS,
                                  VOL_CACHE_DIR) for k in chunk]
                for fut in as_completed(futs):
                    done += 1
                    _p(base + span * done / n, f"{label} {done}/{n}…")
                    volframes, ll = fut.result()
                    for fn, fr in volframes.items():
                        by_field[fn].extend(fr)
                    site_ll = site_ll or ll
    for fn in by_field:
        by_field[fn].sort(key=lambda f: f["time"])
        by_field[fn] = by_field[fn][:MAX_FRAMES]
    return by_field, site_ll


def _hour_key(site, date, hr):
    return (site, f"{date.year}", f"{date.month:02d}",
            f"{date.day:02d}", f"{hr:02d}")


def _share_base(site, date, hr):
    qs = (f"?site={site}&year={date.year}&month={date.month:02d}"
          f"&day={date.day:02d}&hour={hr:02d}")
    host = os.environ.get("SPACE_HOST", "")
    return f"https://{host}/{qs}" if host else qs


def _browse_compute(site, field_name, date, hr, _p):
    """Decode ALL fields in one pass over the hour's volumes; build and
    cache the bundle template. Returns (error_tuple_or_None, tpl_or_None)."""
    try:
        _p(0.02, f"Listing Level 2 volumes for {site} {date} {hr:02d}Z…")
        bucket, keys = list_hour_keys(site, date, hr)
        if not keys:
            return _msg(
                f"No Level 2 volumes found for {site} on {date} "
                f"{hr:02d}:00–{hr:02d}:59 UTC. Check the site ID and date "
                f"(dual-pol fields require 2011+ for most sites)."
            ), None

        n = len(keys)
        with ThreadPoolExecutor(max_workers=6) as ex:
            futs = [ex.submit(_safe_download, bucket, k, VOL_CACHE_DIR)
                    for k in keys]
            for i, _ in enumerate(as_completed(futs)):
                _p(0.04 + 0.20 * (i + 1) / n,
                   f"Downloading volumes from AWS… {i + 1}/{n} "
                   f"(cached volumes skip this)")
        by_field, site_ll = _decode_keys(
            bucket, keys, _p, 0.26, 0.66,
            "Decoding volumes (all four fields)")
        _prune_vol_cache()

        # legacy (Message 1 era, pre-~2008) volumes carry no site coords and
        # decode as 0°N 0°E — fall back to the WSR-88D station table
        if site_ll is not None and abs(site_ll[0]) < 0.1 and abs(site_ll[1]) < 0.1:
            site_ll = None
        if site_ll is None:
            try:
                from pyart.io.nexrad_common import NEXRAD_LOCATIONS
                loc = NEXRAD_LOCATIONS.get(site)
                if loc:
                    site_ll = (loc["lat"], loc["lon"])
            except Exception:
                pass

        if site_ll is None or not any(by_field.values()):
            return _msg(
                f"Volumes were found for {site} in that hour, but no 0.5° "
                f"sweep data could be decoded. Pre-dual-pol data (before "
                f"~2011-2013 depending on site) has no ZDR/CC."
            ), None

        _p(0.94, "Packing the all-field WebGL bundle (textures for every "
                 "field load once)…")
        slat, slon = site_ll
        # keep empty fields in the bundle: their panels show an explanatory
        # message (e.g. dual-pol fields on pre-upgrade data)
        tpl = build_bundle_page(by_field, site, slat, slon,
                                _share_base(site, date, hr))
        with _CACHE_LOCK:
            _PAGE_CACHE[_hour_key(site, date, hr)] = tpl
            while len(_PAGE_CACHE) > _PAGE_CACHE_CAP:
                _PAGE_CACHE.popitem(last=False)
            _FRAME_CACHE[_hour_key(site, date, hr)] = dict(
                by_field=by_field, slat=slat, slon=slon)
            while len(_FRAME_CACHE) > _FRAME_CACHE_CAP:
                _FRAME_CACHE.popitem(last=False)
        return None, tpl
    except Exception:
        return _msg("Unexpected error:\n```\n"
                    + traceback.format_exc()[-1500:] + "\n```"), None


def _dealias_compute(site, date, hr, _p):
    """Region-based dealiasing for the hour; merges the new field into the
    cached bundle and returns the updated template."""
    key_h = _hour_key(site, date, hr)
    with _CACHE_LOCK:
        fc = _FRAME_CACHE.get(key_h)
    if fc is None:  # server restarted since the hour was decoded
        err, _tpl = _browse_compute(site, "Reflectivity", date, hr, _p)
        if err is not None:
            raise RuntimeError(err[0])
        with _CACHE_LOCK:
            fc = _FRAME_CACHE.get(key_h)
    bucket, keys = list_hour_keys(site, date, hr)
    n = len(keys)
    dframes = []
    done = 0
    for i in range(0, n, 2):
        chunk = keys[i:i + 2]
        with _DECODE_SEM:
            with ProcessPoolExecutor(max_workers=N_PROC) as px:
                pfuts = [px.submit(dealias_volume, bucket, k, VOL_CACHE_DIR)
                         for k in chunk]
                for fut in as_completed(pfuts):
                    done += 1
                    _p(0.05 + 0.85 * done / n,
                       f"Region-based dealiasing {done}/{n} volumes "
                       f"({_DEALIAS_ENGINE})…")
                    dframes.extend(fut.result())
    dframes.sort(key=lambda f: f["time"])
    dframes = dframes[:MAX_FRAMES]
    _p(0.95, "Rebuilding bundle with dealiased velocity…")
    by_field = dict(fc["by_field"])
    by_field[DEALIAS_NAME] = dframes
    tpl = build_bundle_page(by_field, site, fc["slat"], fc["slon"],
                            _share_base(site, date, hr))
    with _CACHE_LOCK:
        _PAGE_CACHE[key_h] = tpl
        _FRAME_CACHE[key_h] = dict(by_field=by_field,
                                   slat=fc["slat"], slon=fc["slon"])
    return tpl


def _msg(text):
    return text, ""


def prepare_archive(site, year, month, day, hour):
    """Bundle the hour's raw Level 2 volumes into SITEyyyymmdd_HHUTC.zip
    (files are pulled from the volume cache; anything missing is fetched).
    The volumes are internally compressed already, so members are stored."""
    m = re.match(r"\s*([A-Za-z]{4})\b", site or "")
    if not m:
        raise gr.Error("Pick a site first.")
    site = m.group(1).upper()
    date = dt.date(int(year), int(month), int(day))
    hr = int(str(hour)[:2])
    bucket, keys = list_hour_keys(site, date, hr)
    if not keys:
        raise gr.Error("No Level 2 volumes for this selection.")
    with ThreadPoolExecutor(max_workers=6) as ex:
        list(ex.map(lambda k: _safe_download(bucket, k, VOL_CACHE_DIR), keys))
    out = os.path.join(VOL_CACHE_DIR, f"{site}{date:%Y%m%d}_{hr:02d}UTC.zip")
    if not os.path.exists(out):
        tmp = out + ".part"
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED) as zf:
            for k in keys:
                p = os.path.join(VOL_CACHE_DIR, os.path.basename(k))
                if os.path.exists(p):
                    zf.write(p, arcname=os.path.basename(k))
        os.replace(tmp, out)
    return out


# ----------------------------------------------------------------------------- ui

ILLINI = dict(orange="#FF5F05", blue="#13294B", blue2="#1D3866", blue3="#25457F",
              storm_light="#C8C6C7", severe="#C84113")

ILLINI_THEME = gr.themes.Default(
    primary_hue=gr.themes.Color(
        c50="#FFF1E8", c100="#FFE1CC", c200="#FFC39A", c300="#FFA268",
        c400="#FF8136", c500="#FF5F05", c600="#E25504", c700="#C84113",
        c800="#9E3610", c900="#7A2A0C", c950="#571E08"),
    font=[gr.themes.GoogleFont("Source Sans 3"), "system-ui", "sans-serif"],
).set(
    body_background_fill=ILLINI["blue"],
    body_background_fill_dark=ILLINI["blue"],
    body_text_color="#FFFFFF", body_text_color_dark="#FFFFFF",
    block_background_fill=ILLINI["blue2"],
    block_background_fill_dark=ILLINI["blue2"],
    border_color_primary=ILLINI["blue3"],
    border_color_primary_dark=ILLINI["blue3"],
    input_background_fill=ILLINI["blue"],
    input_background_fill_dark=ILLINI["blue"],
    button_primary_background_fill=ILLINI["orange"],
    button_primary_background_fill_hover="#E25504",
    button_primary_text_color="#FFFFFF",
)

ILLINI_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@600;700;800&family=Source+Sans+3:wght@400;600;700&display=swap');
.gradio-container { background: #13294B !important; max-width: 100% !important;
  padding: 0 !important;
  font-family: 'Source Sans 3','Source Sans Pro',system-ui,Arial,sans-serif !important; }
/* the app wrapper (<main class="fillable app">) is centred at 1280px with 32px
   padding by default — make the content span the full window with 2px edges */
.gradio-container > main, .gradio-container .fillable, .gradio-container main.app {
  max-width: 100% !important; margin: 0 !important; padding: 2px !important; }
/* collapse all inter-element spacing so the map gets the window */
.gradio-container .gap, .gradio-container .wrap,
.gradio-container .contain { gap: 2px !important; }
.gradio-container .html-container { padding: 0 !important; margin: 0 !important; }
.gradio-container .block { margin: 0 !important; }
/* the map fills edge-to-edge */
#map-html, #map-html .html-container, #map-html .block { padding: 0 !important; }
.gradio-container h1 { font-family: 'Montserrat','Arial Black',sans-serif !important;
  font-weight: 800 !important; text-transform: uppercase; letter-spacing: .06em;
  font-size: 17px !important; margin: 0 !important;
  color: #fff !important; border-bottom: 3px solid #FF5F05;
  padding-bottom: 2px; display: inline-block; }
.gradio-container .prose a { color: #FF8136 !important; }
.gradio-container .prose, .gradio-container .prose p { color: #C8C6C7 !important;
  font-size: 13px !important; }
.gradio-container .prose p { margin: 2px 0 !important; }
.gradio-container .block { padding: 2px 6px !important; }
.gradio-container .form { gap: 2px !important; }
.gradio-container .gap, .gradio-container .gradio-row { gap: 3px !important; }
/* compact one-row control strip */
#ctrl-row { flex-wrap: nowrap !important; }
.gradio-container .block > label,
.gradio-container span[data-testid="block-info"] {
  font-size: 11px !important; margin-bottom: 1px !important; }
.gradio-container input, .gradio-container .gradio-dropdown input {
  font-size: 13px !important; }
.gradio-container .gradio-dropdown { min-height: 0 !important; }
#mode-sw .wrap { flex-direction: row !important; gap: 4px !important;
  flex-wrap: nowrap !important; }
#mode-sw label { padding: 0 11px !important; font-size: 13px !important;
  height: 38px !important; display: flex !important;
  align-items: center !important; box-sizing: border-box !important; }
#ctrl-row button { font-size: 13px !important; padding: 6px 8px !important; }
/* visually grey the time dropdowns when disabled (Live mode) — Gradio sets
   the input disabled but doesn't dim it, so the lock wasn't obvious */
#ctrl-row .block:has(input:disabled) { opacity: .45 !important; }
#ctrl-row .block:has(input:disabled) input { cursor: not-allowed !important; }
@media (max-width: 900px) { #ctrl-row { flex-wrap: wrap !important; } }
footer { display: none !important; }
#dax-trigger, #rt-refresh, #rt-force { display: none !important; }
@media (max-width: 700px) {
  .gradio-container { padding: 8px 10px 4px !important; }
  .gradio-container h1 { font-size: 15px !important; }
  .gradio-container .prose, .gradio-container .prose p {
    font-size: 11px !important; }
}
"""

# CliMAS logo: use logo.png if present in the repo, else an inline SVG replica
if os.path.exists(os.path.join(os.path.dirname(__file__), "logo.png")):
    _b64 = base64.b64encode(
        open(os.path.join(os.path.dirname(__file__), "logo.png"), "rb").read()
    ).decode()
    LOGO_HTML = (f'<img src="data:image/png;base64,{_b64}" alt="CliMAS" '
                 f'style="width:46px;height:46px;border-radius:10px;flex:none"/>')
else:
    LOGO_HTML = """
<svg viewBox="0 0 400 400" width="46" height="46"
     style="border-radius:10px;flex:none" aria-label="CliMAS">
  <circle cx="200" cy="200" r="200" fill="#13294B"/>
  <path d="M150 70 h100 v42 h-26 v76 h26 v42 h-100 v-42 h26 v-76 h-26 z"
        fill="#FF5F05" stroke="#fff" stroke-width="10"/>
  <g fill="#fff" text-anchor="middle"
     font-family="'Source Sans 3','Source Sans Pro',sans-serif"
     font-weight="600" font-size="28">
    <text x="200" y="268">Climate,</text>
    <text x="200" y="298">Meteorology &amp;</text>
    <text x="200" y="328">Atmospheric</text>
    <text x="200" y="358">Sciences</text>
  </g>
</svg>"""

HEADER_HTML = f"""
<div style="display:flex;align-items:center;gap:10px">
  {LOGO_HTML}
  <div>
    <h1>NEXRAD Level 2 — 0.5° sweep browser</h1>
    <p style="color:#C8C6C7;font-size:12px;margin:1px 0">
      Brought to you by the
      <a href="https://climas.illinois.edu" target="_blank"
         style="color:#FF8136">Department of Climate, Meteorology &amp;
         Atmospheric Sciences</a>, University of Illinois Urbana–Champaign</p>
  </div>
</div>"""

OG_HEAD = f"""
<meta property="og:title" content="NEXRAD level 2 browser"/>
<meta property="og:description" content="Gate-native WebGL browsing of archived WSR-88D 0.5° scans (SAILS-aware) from the AWS NEXRAD Level 2 archive."/>
<meta property="og:image" content="{THUMB_URL}"/>
<meta property="og:type" content="website"/>
<meta name="twitter:card" content="summary_large_image"/>
<meta name="twitter:title" content="NEXRAD level 2 browser"/>
<meta name="twitter:image" content="{THUMB_URL}"/>
<script>
// show only elapsed time in progress timers (strip Gradio's "/eta" part)
addEventListener('DOMContentLoaded', function () {{
  var rxFull = /^\\s*\\d+(?:\\.\\d+)?\\/\\d+(?:\\.\\d+)?s\\s*$/;
  function fixEl(el) {{
    if (!el || !el.childNodes || !rxFull.test(el.textContent)) return;
    var seen = false;
    el.childNodes.forEach(function (n) {{
      if (n.nodeType !== 3) return;
      var v = n.nodeValue;
      if (!seen) {{
        var m = v.match(/^(\\s*\\d+(?:\\.\\d+)?)\\/\\d+(?:\\.\\d+)?(s\\s*)?$/);
        if (m) {{ n.nodeValue = m[1] + (m[2] || ''); seen = true; return; }}
        var i = v.indexOf('/');
        if (i >= 0) {{ n.nodeValue = v.slice(0, i); seen = true; }}
      }} else if (/^\\d+(?:\\.\\d+)?$/.test(v.trim())) {{
        n.nodeValue = '';
      }}
    }});
  }}
  new MutationObserver(function (ms) {{
    ms.forEach(function (m) {{
      if (m.type === 'characterData') fixEl(m.target.parentElement);
      if (m.addedNodes) m.addedNodes.forEach(function (an) {{
        if (an.nodeType === 1 && an.textContent.length < 24) fixEl(an);
      }});
    }});
  }}).observe(document.body, {{subtree: true, characterData: true,
                               childList: true}});
}});
</script>
<script>
// Embedded on huggingface.co the Space iframe is auto-resized (via
// iframeResizer) from the app's reported content height. The map iframe
// is sized with calc(100vh - 205px), so content height is always
// ~viewport + footer: the host grows the iframe, 100vh grows, the map
// grows again - a one-way ratchet that any reflow re-triggers. Break the
// loop by pinning the map to a fixed pixel height whenever we are
// framed. NOTE: must run immediately - this script can be evaluated
// after DOMContentLoaded has already fired.
(function () {{
  if (window.self === window.top) return;   // direct visit: keep vh sizing
  var pinH = null;
  function pin() {{
    var ifs = document.querySelectorAll('.html-container iframe');
    for (var i = 0; i < ifs.length; i++) {{
      var f = ifs[i];
      if (f.dataset.hpin) continue;
      f.dataset.hpin = '1';
      if (pinH === null) {{
        // leave room for the compact header + control row + footer so the
        // map fills the granted viewport with minimal dead space
        pinH = Math.max(480, window.innerHeight - 170);
      }}
      f.style.height = pinH + 'px';
    }}
  }}
  function arm() {{
    pin();
    new MutationObserver(pin).observe(document.body,
                                      {{subtree: true, childList: true}});
  }}
  if (document.readyState === 'loading') {{
    addEventListener('DOMContentLoaded', arm);
  }} else {{
    arm();
  }}
}})();
</script>
"""

gr.set_static_paths(paths=[VOL_CACHE_DIR])   # serves warnings.json

with gr.Blocks(title="NEXRAD Level 2 — 0.5° browser", head=OG_HEAD,
               theme=ILLINI_THEME, css=ILLINI_CSS) as demo:
    gr.HTML(HEADER_HTML)
    with gr.Row(elem_id="ctrl-row"):
        mode_sw = gr.Radio(["Archive", "Live"], value="Archive",
                           label="Mode", scale=1, min_width=172,
                           elem_id="mode-sw")
        site_tb = gr.Dropdown(SITE_CHOICES, value="KILX", label="Site",
                              allow_custom_value=True, scale=2, min_width=150)
        field_dd = gr.Dropdown(list(FIELDS) + [QUAD, ZVF], value="Reflectivity",
                               label="Field", scale=2, min_width=140)
        year_dd = gr.Dropdown(YEARS, value="2023", label="Year (UTC)",
                              scale=1, min_width=84)
        month_dd = gr.Dropdown(MONTHS, value="06", label="Month (UTC)",
                               scale=1, min_width=72)
        day_dd = gr.Dropdown(DAYS, value="29", label="Day (UTC)",
                             scale=1, min_width=72)
        hour_dd = gr.Dropdown(HOURS, value="18:00", label="Hour (UTC)",
                              scale=1, min_width=84)
        with gr.Column(scale=1, min_width=150):
            go = gr.Button("Load hour", variant="primary")
            dl = gr.DownloadButton("Download raw (.zip)",
                                   interactive=False, size="sm")
            # hidden trigger: the in-map "Dealias V" checkbox clicks this
            # via parent.document when the hour hasn't been dealiased yet
            dax = gr.Button("Dealias velocity", elem_id="dax-trigger",
                            size="sm")
            # hidden: the live page clicks this every 5 minutes
            rtr = gr.Button("Refresh live", elem_id="rt-refresh", size="sm")
            # hidden: the in-map refresh button clicks this to force a live
            # re-decode now (picks up new scans) without reloading the iframe
            frr = gr.Button("Force live", elem_id="rt-force", size="sm")
    status = gr.Markdown()
    map_html = gr.HTML(elem_id="map-html")

    shared = gr.State(False)
    view_st = gr.State(None)
    deal_st = gr.State(False)   # live dealiasing on for this session

    def browse_h(mode, site, field, year, month, day, hour,
                 progress=gr.Progress()):
        rt = (mode == "Live")
        info, page = browse(site, field, year, month, day, hour,
                            progress=progress, realtime=rt)
        return info, page, gr.DownloadButton(
            interactive=bool(page) and not rt)

    go.click(browse_h,
             [mode_sw, site_tb, field_dd, year_dd, month_dd, day_dd, hour_dd],
             [status, map_html, dl], show_progress_on=map_html)

    def set_mode(mode, site, field, year, month, day, hour,
                 progress=gr.Progress()):
        rt = (mode == "Live")
        info, page = browse(site, field, year, month, day, hour,
                            progress=progress, realtime=rt)
        dd = [gr.Dropdown(interactive=not rt) for _ in range(4)]
        return (*dd, info, page,
                gr.DownloadButton(interactive=bool(page) and not rt),
                False)   # reset live-dealias preference on mode change

    mode_sw.change(set_mode,
                   [mode_sw, site_tb, field_dd, year_dd, month_dd, day_dd,
                    hour_dd],
                   [year_dd, month_dd, day_dd, hour_dd, status, map_html, dl,
                    deal_st],
                   show_progress_on=map_html)

    def rt_refresh(site, field, progress=gr.Progress()):
        info, page = browse(site, field, "", "", "", "",
                            progress=progress, realtime=True)
        return info, page

    rtr.click(rt_refresh, [site_tb, field_dd], [status, map_html],
              show_progress_on=map_html)
    dl.click(prepare_archive,
             [site_tb, year_dd, month_dd, day_dd, hour_dd], dl)

    def dealias_h(mode, site, field, year, month, day, hour,
                  progress=gr.Progress()):
        if mode == "Live":
            info, page = browse(site, "Radial velocity", year, month, day,
                                hour, progress=progress, realtime=True,
                                want_deal=True)
        else:
            info, page = browse(site, DEALIAS_NAME, year, month, day, hour,
                                progress=progress)
        return info, page, (mode == "Live")

    dax.click(dealias_h,
              [mode_sw, site_tb, field_dd, year_dd, month_dd, day_dd,
               hour_dd],
              [status, map_html, deal_st], show_progress_on=map_html)

    # server-side live refresh: a Timer tick every 2 minutes replaces the
    # in-page hidden-button click (programmatic DOM clicks proved unreliable
    # across browsers). Archive sessions skip at zero cost.
    rt_timer = gr.Timer(120)

    def rt_tick(mode, site, field, deal, progress=gr.Progress()):
        # Only refresh the live frame file; the in-page poller merges the new
        # frames in place when ready. Crucially this has NO Gradio outputs, so
        # the display is never marked "generating"/greyed during an autoupdate.
        if mode == "Live":
            try:
                browse(site, "Radial velocity" if deal else field,
                       "", "", "", "", progress=progress,
                       realtime=True, want_deal=deal)
            except Exception:
                pass

    rt_timer.tick(rt_tick, [mode_sw, site_tb, field_dd, deal_st],
                  None, show_progress="hidden")

    def force_refresh(mode, site, field, deal, progress=gr.Progress()):
        # user hit the in-map refresh: drop the short-lived RT cache so we
        # re-check the feed for new scans now and rewrite live_<site>.json.
        # NO Gradio outputs -> the display is never greyed; the in-page poller
        # (with its own spinning icon) merges the result when ready.
        if mode == "Live":
            try:
                _RT_CACHE.pop(site, None)
                browse(site, "Radial velocity" if deal else field,
                       "", "", "", "", progress=progress,
                       realtime=True, want_deal=deal)
            except Exception:
                pass

    frr.click(force_refresh, [mode_sw, site_tb, field_dd, deal_st], None,
              show_progress="hidden")
    gr.Markdown(
        "Level 2 decoding by [xradar](https://github.com/swnesbitt/xradar) "
        "(openradar; S. Nesbitt fork) "
        "· radar processing by [Py-ART](https://arm-doe.github.io/pyart/) "
        "(Helmus & Collis 2016, [doi:10.5334/jors.119](https://doi.org/10.5334/jors.119)) "
        "· velocity dealiasing by the Rust region-based dealiaser "
        "[region-dealias](https://github.com/swnesbitt/region-dealias) "
        "· colormaps from [cmweather](https://github.com/openradar/cmweather) "
        "and [CMasher](https://cmasher.readthedocs.io) "
        "· data from the [NOAA NEXRAD Level II archive on AWS](https://registry.opendata.aws/noaa-nexrad/) "
        "· [Source code on GitHub](https://github.com/swnesbitt/nexrad-level2-browser)."
    )

    def init_values(request: gr.Request):
        """Fast: restore dropdown values (and map view) from URL params."""
        q = dict(request.query_params) if request else {}
        deal = q.get("dealias") == "1"
        rt = q.get("rt") == "1"
        view = None
        try:
            if "lat" in q and "lon" in q and "zoom" in q:
                view = [float(q["lat"]), float(q["lon"]),
                        max(3, min(14, int(float(q["zoom"]))))]
        except (TypeError, ValueError):
            view = None
        site = q.get("site", "KILX").upper()[:4]
        field = q.get("field", "Reflectivity")
        if field not in FIELDS and field not in (QUAD, ZVF, DEALIAS_NAME):
            field = "Reflectivity"
        year = q.get("year", "2023")
        month = q.get("month", "06").zfill(2)
        day = q.get("day", "29").zfill(2)
        hour = q.get("hour", "18")[:2].zfill(2) + ":00"
        year = year if year in YEARS else "2023"
        month = month if month in MONTHS else "06"
        day = day if day in DAYS else "29"
        hour = hour if hour in HOURS else "18:00"
        arch = not rt
        return (site, field,
                gr.Dropdown(value=year, interactive=arch),
                gr.Dropdown(value=month, interactive=arch),
                gr.Dropdown(value=day, interactive=arch),
                gr.Dropdown(value=hour, interactive=arch),
                ("site" in q or "year" in q), view, deal,
                "Live" if rt else "Archive")

    def maybe_browse(mode, view, deal, site, field, year, month, day,
                     hour, progress=gr.Progress()):
        """Slow: auto-load on every visit — the default case on a plain
        visit, the shared view (incl. map center/zoom) when params exist."""
        rt = (mode == "Live")
        info, page = browse(site, field, year, month, day, hour,
                            progress=progress, view=view, want_deal=deal,
                            realtime=rt)
        return info, page, gr.DownloadButton(
            interactive=bool(page) and not rt)

    demo.load(
        init_values, None,
        [site_tb, field_dd, year_dd, month_dd, day_dd, hour_dd, shared,
         view_st, deal_st, mode_sw],
    ).then(
        maybe_browse,
        [mode_sw, view_st, deal_st, site_tb, field_dd, year_dd, month_dd,
         day_dd, hour_dd],
        [status, map_html, dl], show_progress_on=map_html,
    )

# pre-render the default case at startup so first visitors get it instantly
threading.Thread(target=lambda: browse(*DEFAULT_VIEW), daemon=True).start()
# keep live warnings fresh for real-time mode
threading.Thread(target=_warn_poller, daemon=True).start()

if __name__ == "__main__":
    # ssr_mode=False -> Python serves the OG-patched index.html, so social
    # link previews show our title/thumbnail instead of Gradio defaults
    demo.launch(ssr_mode=False)
