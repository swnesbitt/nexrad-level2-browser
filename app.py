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
import traceback
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import (ProcessPoolExecutor, ThreadPoolExecutor,
                                as_completed)

import matplotlib

matplotlib.use("Agg")

import cmweather  # noqa: F401  (registers ChaseSpectral & friends)
import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import pyart
import xradar as xd
from PIL import Image
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
        pyart="velocity", cmap="balance", vmin=-40, vmax=40,
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

ELEV_MAX = 0.75          # deg — treat sweeps below this as the 0.5° split cut
MAX_FRAMES = 60          # safety cap on rendered sweeps
N_PROC = max(1, min(2, os.cpu_count() or 1))   # decode workers (CPU-bound)

# persistent raw-volume cache (survives across requests; LRU-pruned)
VOL_CACHE_DIR = os.path.join(tempfile.gettempdir(), "nexrad_vol_cache")
os.makedirs(VOL_CACHE_DIR, exist_ok=True)
VOL_CACHE_BYTES = 2 << 30  # 2 GiB


def _prune_vol_cache():
    try:
        files = [(os.path.getmtime(p), os.path.getsize(p), p)
                 for p in (os.path.join(VOL_CACHE_DIR, f)
                           for f in os.listdir(VOL_CACHE_DIR))]
        files.sort(reverse=True)
        total = 0
        for mt, sz, p in files:
            total += sz
            if total > VOL_CACHE_BYTES:
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

    try:  # censor gates with no usable return (same-sweep reflectivity)
        z = radar.get_field(sweep, "reflectivity")
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
                tick=field_cfg["tick"], label=f"{field_cfg['label']}{unit}")


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
    if "DBZH" in ds:  # censor gates with no usable return
        nodata = nodata | _noreturn_mask(ds["DBZH"].values)
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


def _process_xradar(path, key, cfgs):
    """Read once with xradar (swnesbitt fork); render every requested field.
    Returns ({field_name: [frames]}, site_ll)."""
    dtree = xd.io.open_nexradlevel2_datatree(_gunzip(path))
    root = dtree.ds
    site_ll = (float(root["latitude"].values), float(root["longitude"].values))
    vol = os.path.basename(key)
    vcp = str(root.attrs.get("scan_name", "") or "")
    dyn = str(root.attrs.get("dynamic_scan_type", "") or "")
    if dyn and dyn.lower() not in ("none", "false", ""):
        vcp = f"{vcp} ({dyn})" if vcp else dyn

    sweeps = []
    for name in sorted((c for c in dtree.children if c.startswith("sweep_")),
                       key=lambda n: int(n.split("_")[1])):
        ds = dtree[name].ds
        el = float(ds["sweep_fixed_angle"].values)
        if el >= ELEV_MAX:
            continue
        has_v = "VRADH" in ds and bool(np.isfinite(ds["VRADH"].values).any())
        sweeps.append(dict(name=name, ds=ds, el=el, has_v=has_v))

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

QUAD_PAGE = """<!DOCTYPE html>
<html><head>
<meta charset="utf-8"/>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
 @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@600;700&family=Source+Sans+3:wght@400;600;700&display=swap');
 :root{--io:#FF5F05;--ib:#13294B;--ib2:#1D3866;--ib3:#25457F;--sl:#C8C6C7}
 html,body{margin:0;height:100%;background:var(--ib)}
 #grid{position:absolute;inset:0 0 86px 0;display:grid;gap:4px;
       grid-template-columns:1fr 1fr;grid-template-rows:1fr 1fr;
       background:#0c1d38}
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
 #bar{position:absolute;left:0;right:0;bottom:0;height:86px;
      background:var(--ib);border-top:4px solid var(--io);color:#fff;
      font:13px 'Source Sans 3',system-ui,sans-serif;display:flex;
      flex-direction:column;justify-content:center;gap:6px;
      padding:6px 14px;box-sizing:border-box}
 #row1{display:flex;align-items:center;gap:10px}
 #slider{flex:1;accent-color:var(--io)}
 #op{accent-color:var(--io);width:90px}
 #label{font-variant-numeric:tabular-nums;white-space:nowrap;overflow:hidden;
        text-overflow:ellipsis;color:var(--sl)}
 button{background:var(--io);color:#fff;border:0;border-radius:4px;
        padding:4px 12px;cursor:pointer;font-size:13px;
        font-family:'Montserrat',sans-serif;font-weight:700}
 button:hover{background:#E25504}
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
<div id="bar">
 <div id="row1">
   <button id="play">&#9654;</button>
   <span id="modes"></span>
   <input id="slider" type="range" min="0" max="0" value="0" step="1"/>
   <label class="ck"><input type="checkbox" id="ck-counties" checked/><span>Counties</span></label>
   <label class="ck"><input type="checkbox" id="ck-interstates" checked/><span>Highways</span></label>
   <span style="opacity:.7">opacity</span>
   <input id="op" type="range" min="10" max="100" value="100"/>
 </div>
 <div id="label"></div>
</div>
<script>
const DATA = __DATA__;
const SITE = [__SLAT__, __SLON__];
const SHARE_BASE = "__SHAREBASE__";
const QUADF = "All fields (4-panel)";
const INIT_VIEW = __VIEW__;   // [lat, lon, zoom] from a share link, or null
let mode = __MODE__;
if (mode !== 'quad' && !DATA.some(fd => fd.name === mode))
  mode = DATA[0].name;
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
  const chip = document.createElement('div'); chip.className = 'chip';
  chip.textContent = fd.name; wrap.appendChild(chip);
  if (!fd.frames.length) {
    const nd = document.createElement('div'); nd.className = 'nodata';
    nd.textContent = (fd.name.indexOf('reflectivity') > 0 ||
                      fd.name.indexOf('Correlation') === 0)
      ? 'Pre-polarimetric upgrade data' : 'No data for this hour';
    wrap.appendChild(nd);
  }
  // mini colorbar
  const cb = fd.cbar;
  const cwrap = document.createElement('div'); cwrap.className = 'pcbar';
  const cl = document.createElement('div'); cl.className = 'pcl';
  cl.textContent = cb.label;
  const ccv = document.createElement('canvas'); ccv.width = 10; ccv.height = 180;
  const ct = document.createElement('div'); ct.className = 'pct';
  cwrap.appendChild(cl); cwrap.appendChild(ccv); cwrap.appendChild(ct);
  wrap.appendChild(cwrap);
  const cctx = ccv.getContext('2d');
  const g = cctx.createLinearGradient(0, ccv.height, 0, 0);
  cb.stops.forEach((c,i)=>g.addColorStop(i/(cb.stops.length-1), c));
  cctx.fillStyle = g; cctx.fillRect(0,0,ccv.width,ccv.height);
  const t0 = Math.ceil(cb.vmin/cb.tick)*cb.tick;
  for (let v=t0; v<=cb.vmax+1e-9; v+=cb.tick){
    const val = +v.toFixed(6);
    const d = document.createElement('div');
    d.style.top = (100-(val-cb.vmin)/(cb.vmax-cb.vmin)*100)+'%';
    d.textContent = ''+(cb.tick<1?val.toFixed(1):Math.round(val));
    ct.appendChild(d);
  }
  // map (zoom control on all; CSS shows it only on the first visible panel)
  const map = L.map(mdiv, {center:SITE, zoom:7, zoomControl:true,
                           attributionControl:first});
  // radar canvas pane sits between basemap tiles and all vector panes
  map.createPane('radarpane');
  map.getPane('radarpane').style.zIndex = 250;
  map.getPane('radarpane').style.pointerEvents = 'none';
  map.getPane('radarpane').appendChild(cv);
  // reference overlays (counties / roads) draw above everything
  map.createPane('refpane');
  map.getPane('refpane').style.zIndex = 650;
  map.getPane('refpane').style.pointerEvents = 'none';
  L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png',
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
  const texCache = new Array(fd.frames.length).fill(null);
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
    img.src = 'data:image/png;base64,' + fd.frames[i].img;
  }
  let raf = 0;
  function draw(){
    raf = 0;
    const pi = Math.min(idx, fd.frames.length-1);
    if (pi < 0) return;
    const f = fd.frames[pi];
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
  return {map, fd, ring, requestDraw, texFor, wrap};
}

DATA.forEach((fd,i)=>panels.push(makePanel(fd, i===0)));

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
  'Differential reflectivity':'ZDR','Correlation coefficient':'CC'};
const modesEl = document.getElementById('modes');
function mkBtn(txt, m){
  const b = document.createElement('button');
  b.textContent = txt; b.dataset.m = m;
  b.onclick = ()=>setMode(m);
  modesEl.appendChild(b);
}
DATA.forEach(fd=>mkBtn(SHORT[fd.name]||fd.name, fd.name));
if (DATA.length > 1) mkBtn('2\\u00d72', 'quad');
function setMode(m){
  mode = m;
  const grid = document.getElementById('grid');
  let zcDone = false;
  panels.forEach(p=>{
    const vis = (m === 'quad') || (p.fd.name === m);
    p.wrap.style.display = vis ? 'block' : 'none';
    p.wrap.classList.toggle('solo', vis && m !== 'quad');
    const zc = vis && !zcDone;
    p.wrap.classList.toggle('zc', zc);
    if (zc) zcDone = true;
  });
  grid.style.gridTemplateColumns = m === 'quad' ? '1fr 1fr' : '1fr';
  grid.style.gridTemplateRows    = m === 'quad' ? '1fr 1fr' : '1fr';
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
  const mr = vis.fd.frames.length ? vis.fd.frames[0].maxr : 3e5;
  const dLat = mr/111320, dLon = mr/(111320*Math.cos(SITE[0]*Math.PI/180));
  vis.map.fitBounds(
    [[SITE[0]-dLat,SITE[1]-dLon],[SITE[0]+dLat,SITE[1]+dLon]],
    {padding:[4,4]});
}, 90);

// controls
const nmax = Math.max.apply(null, DATA.map(fd=>fd.frames.length));
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
playBtn.addEventListener('click', ()=>{
  playing=!playing;
  playBtn.innerHTML = playing?'&#10074;&#10074;':'&#9654;';
  if(playing) timer=setInterval(()=>show((idx+1)%nmax),450);
  else clearInterval(timer);
});
document.addEventListener('keydown', e=>{
  if(e.key==='ArrowRight') show(Math.min(idx+1,nmax-1));
  if(e.key==='ArrowLeft') show(Math.max(idx-1,0));
});
document.getElementById('share').addEventListener('click', ()=>{
  const vis = panels.find(p=>p.wrap.style.display !== 'none') || panels[0];
  const c = vis.map.getCenter(), z = vis.map.getZoom();
  const url = SHARE_BASE + '&field=' +
    encodeURIComponent(mode === 'quad' ? QUADF : mode) +
    '&lat=' + c.lat.toFixed(4) + '&lon=' + c.lng.toFixed(4) + '&zoom=' + z;
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
let countyLayers = null, countyLoading = false;
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

// stagger texture preloads
panels.forEach((p,pi)=>p.fd.frames.forEach((_,i)=>
  setTimeout(()=>p.texFor(i,()=>{}), 80*i+20*pi)));
show(0);
</script></body></html>"""


def build_bundle_page(by_field, site, slat, slon, share_base=""):
    """One page carrying ALL fields' polar textures; the requested view
    (single field or 2×2) is substituted into __MODE__ at serve time, so
    every field/4-panel switch after the first load is instant."""
    data = [dict(name=fn, cbar=colorbar_cfg(FIELDS[fn]),
                 frames=[{k: f[k] for k in ("img", "naz", "ngates", "r0",
                                            "dr", "el", "maxr", "label")}
                         for f in frames])
            for fn, frames in by_field.items()]
    page = (QUAD_PAGE
            .replace("__DATA__", json.dumps(data))
            .replace("__SLAT__", f"{slat:.5f}")
            .replace("__SLON__", f"{slon:.5f}")
            .replace("__SITE__", site)
            .replace("__SHAREBASE__", share_base))
    return (f'<iframe allow="clipboard-write" '
            f'style="width:100%;height:calc(100vh - 245px);'
            f'min-height:480px;border:0;border-radius:4px" '
            f'srcdoc="{html_mod.escape(page)}"></iframe>')


def _mode_page(tpl, field_name, view=None):
    """Substitute the initial view mode and optional [lat, lon, zoom] into a
    cached bundle template. The template is already HTML-escaped (srcdoc),
    so escape the JSON too."""
    mode = "quad" if field_name == QUAD else field_name
    return (tpl
            .replace("__MODE__", html_mod.escape(json.dumps(mode)))
            .replace("__VIEW__", html_mod.escape(json.dumps(view))))


# ----------------------------------------------------------------------------- gradio callback


# rendered-page cache: default case is pre-rendered at startup, and recent
# views are served instantly (share links, repeat visits)
DEFAULT_VIEW = ("KILX", "Reflectivity", "2023", "06", "29", "18:00")
_PAGE_CACHE = collections.OrderedDict()
_PAGE_CACHE_CAP = 6   # hours; each bundle holds all four fields (~30-50 MB)
_INFLIGHT = {}
_CACHE_LOCK = threading.Lock()


def browse(site, field_name, year, month, day, hour, progress=None,
           view=None):
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
        try:
            date = dt.date(int(year), int(month), int(day))
        except ValueError:
            return _msg("That calendar date doesn't exist — check day/month.")
        hr = int(str(hour)[:2])
        if field_name not in FIELDS and field_name != QUAD:
            field_name = "Reflectivity"

        # ---- per-hour bundle cache / in-flight dedup ---------------------
        key_h = _hour_key(site, date, hr)
        with _CACHE_LOCK:
            tpl = _PAGE_CACHE.get(key_h)
            if tpl is not None:
                _PAGE_CACHE.move_to_end(key_h)
        if tpl is not None:
            return "", _mode_page(tpl, field_name, view)
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
                return "", _mode_page(tpl, field_name, view)
            # fall through and compute ourselves if the other run failed

        try:
            return _browse_compute(site, field_name, date, hr, _p, view)
        finally:
            if owner:
                with _CACHE_LOCK:
                    ev = _INFLIGHT.pop(key_h, None)
                if ev:
                    ev.set()
    except Exception:
        return _msg("Unexpected error:\n```\n"
                    + traceback.format_exc()[-1500:] + "\n```")


def _hour_key(site, date, hr):
    return (site, f"{date.year}", f"{date.month:02d}",
            f"{date.day:02d}", f"{hr:02d}")


def _share_base(site, date, hr):
    qs = (f"?site={site}&year={date.year}&month={date.month:02d}"
          f"&day={date.day:02d}&hour={hr:02d}")
    host = os.environ.get("SPACE_HOST", "")
    return f"https://{host}/{qs}" if host else qs


def _browse_compute(site, field_name, date, hr, _p, view=None):
    """Decode ALL fields in one pass over the hour's volumes; build and
    cache the 4 single-field pages plus the 4-panel page; return the one
    that was requested."""
    try:
        _p(0.02, f"Listing Level 2 volumes for {site} {date} {hr:02d}Z…")
        bucket, keys = list_hour_keys(site, date, hr)
        if not keys:
            return _msg(
                f"No Level 2 volumes found for {site} on {date} "
                f"{hr:02d}:00–{hr:02d}:59 UTC. Check the site ID and date "
                f"(dual-pol fields require 2011+ for most sites)."
            )

        by_field = {fn: [] for fn in FIELDS}
        site_ll = None
        n = len(keys)
        with ThreadPoolExecutor(max_workers=6) as ex:
            futs = [ex.submit(_safe_download, bucket, k, VOL_CACHE_DIR)
                    for k in keys]
            for i, _ in enumerate(as_completed(futs)):
                _p(0.04 + 0.20 * (i + 1) / n,
                   f"Downloading volumes from AWS… {i + 1}/{n} "
                   f"(cached volumes skip this)")
        with ProcessPoolExecutor(max_workers=N_PROC) as px:
            pfuts = [px.submit(process_volume, bucket, k, FIELDS,
                               VOL_CACHE_DIR) for k in keys]
            done = 0
            for fut in as_completed(pfuts):
                done += 1
                _p(0.26 + 0.66 * done / n,
                   f"Decoding volumes {done}/{n} on {N_PROC} cores — "
                   f"all four fields, regridding to polar textures")
                volframes, ll = fut.result()
                for fn, fr in volframes.items():
                    by_field[fn].extend(fr)
                site_ll = site_ll or ll
        _prune_vol_cache()
        for fn in by_field:
            by_field[fn].sort(key=lambda f: f["time"])
            by_field[fn] = by_field[fn][:MAX_FRAMES]

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
            )

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
        return "", _mode_page(tpl, field_name, view)
    except Exception:
        return _msg("Unexpected error:\n```\n" + traceback.format_exc()[-1500:] + "\n```")


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
  padding: 10px 18px 4px !important;
  font-family: 'Source Sans 3','Source Sans Pro',system-ui,Arial,sans-serif !important; }
.gradio-container h1 { font-family: 'Montserrat','Arial Black',sans-serif !important;
  font-weight: 800 !important; text-transform: uppercase; letter-spacing: .06em;
  font-size: 20px !important; margin: 0 !important;
  color: #fff !important; border-bottom: 3px solid #FF5F05;
  padding-bottom: 4px; display: inline-block; }
.gradio-container .prose a { color: #FF8136 !important; }
.gradio-container .prose, .gradio-container .prose p { color: #C8C6C7 !important;
  font-size: 13px !important; }
.gradio-container .prose p { margin: 2px 0 !important; }
.gradio-container .block { padding: 4px 10px !important; }
.gradio-container .form { gap: 4px !important; }
.gradio-container .gap, .gradio-container .gradio-row { gap: 6px !important; }
footer { display: none !important; }
"""

# CliMAS logo: use logo.png if present in the repo, else an inline SVG replica
if os.path.exists(os.path.join(os.path.dirname(__file__), "logo.png")):
    _b64 = base64.b64encode(
        open(os.path.join(os.path.dirname(__file__), "logo.png"), "rb").read()
    ).decode()
    LOGO_HTML = (f'<img src="data:image/png;base64,{_b64}" alt="CliMAS" '
                 f'style="width:64px;height:64px;border-radius:50%;flex:none"/>')
else:
    LOGO_HTML = """
<svg viewBox="0 0 400 400" width="64" height="64"
     style="border-radius:50%;flex:none" aria-label="CliMAS">
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
<div style="display:flex;align-items:center;gap:14px">
  {LOGO_HTML}
  <div>
    <h1>NEXRAD Level 2 — 0.5° sweep browser</h1>
    <p style="color:#C8C6C7;font-size:13px;margin:2px 0">
      Browse one hour of archived WSR-88D base scans (SAILS-aware) from the
      <a href="https://registry.opendata.aws/noaa-nexrad/" target="_blank"
         style="color:#FF8136">AWS Open Data archive</a>.</p>
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
"""

with gr.Blocks(title="NEXRAD Level 2 — 0.5° browser", head=OG_HEAD,
               theme=ILLINI_THEME, css=ILLINI_CSS) as demo:
    gr.HTML(HEADER_HTML)
    with gr.Row():
        site_tb = gr.Dropdown(SITE_CHOICES, value="KILX", label="Site",
                              allow_custom_value=True, scale=2)
        field_dd = gr.Dropdown(list(FIELDS) + [QUAD], value="Reflectivity",
                               label="Field", scale=2)
        year_dd = gr.Dropdown(YEARS, value="2023", label="Year (UTC)", scale=1)
        month_dd = gr.Dropdown(MONTHS, value="06", label="Month (UTC)", scale=1)
        day_dd = gr.Dropdown(DAYS, value="29", label="Day (UTC)", scale=1)
        hour_dd = gr.Dropdown(HOURS, value="18:00", label="Hour (UTC)", scale=1)
        with gr.Column(scale=1, min_width=170):
            go = gr.Button("Load hour", variant="primary")
            dl = gr.DownloadButton("Download raw (.zip)",
                                   interactive=False, size="sm")
    status = gr.Markdown()
    map_html = gr.HTML()
    def browse_h(site, field, year, month, day, hour,
                 progress=gr.Progress()):
        info, page = browse(site, field, year, month, day, hour,
                            progress=progress)
        return info, page, gr.DownloadButton(interactive=bool(page))

    go.click(browse_h, [site_tb, field_dd, year_dd, month_dd, day_dd, hour_dd],
             [status, map_html, dl], show_progress_on=map_html)
    dl.click(prepare_archive,
             [site_tb, year_dd, month_dd, day_dd, hour_dd], dl)
    gr.Markdown(
        "Level 2 decoding by [xradar](https://github.com/swnesbitt/xradar) "
        "(openradar; S. Nesbitt fork) "
        "· radar processing by [Py-ART](https://arm-doe.github.io/pyart/) "
        "(Helmus & Collis 2016, [doi:10.5334/jors.119](https://doi.org/10.5334/jors.119)) "
        "· colormaps from [cmweather](https://github.com/openradar/cmweather) "
        "· data from the [NOAA NEXRAD Level II archive on AWS](https://registry.opendata.aws/noaa-nexrad/) "
        "· [Source code on GitHub](https://github.com/swnesbitt/nexrad-level2-browser)."
    )

    shared = gr.State(False)
    view_st = gr.State(None)

    def init_values(request: gr.Request):
        """Fast: restore dropdown values (and map view) from URL params."""
        q = dict(request.query_params) if request else {}
        view = None
        try:
            if "lat" in q and "lon" in q and "zoom" in q:
                view = [float(q["lat"]), float(q["lon"]),
                        max(3, min(14, int(float(q["zoom"]))))]
        except (TypeError, ValueError):
            view = None
        site = q.get("site", "KILX").upper()[:4]
        field = q.get("field", "Reflectivity")
        if field not in FIELDS and field != QUAD:
            field = "Reflectivity"
        year = q.get("year", "2023")
        month = q.get("month", "06").zfill(2)
        day = q.get("day", "29").zfill(2)
        hour = q.get("hour", "18")[:2].zfill(2) + ":00"
        year = year if year in YEARS else "2023"
        month = month if month in MONTHS else "06"
        day = day if day in DAYS else "29"
        hour = hour if hour in HOURS else "18:00"
        return (site, field, year, month, day, hour,
                ("site" in q or "year" in q), view)

    def maybe_browse(is_shared, view, site, field, year, month, day, hour,
                     progress=gr.Progress()):
        """Slow: auto-load on every visit — the default case on a plain
        visit, the shared view (incl. map center/zoom) when params exist."""
        info, page = browse(site, field, year, month, day, hour,
                            progress=progress, view=view)
        return info, page, gr.DownloadButton(interactive=bool(page))

    demo.load(
        init_values, None,
        [site_tb, field_dd, year_dd, month_dd, day_dd, hour_dd, shared,
         view_st],
    ).then(
        maybe_browse,
        [shared, view_st, site_tb, field_dd, year_dd, month_dd, day_dd,
         hour_dd],
        [status, map_html, dl], show_progress_on=map_html,
    )

# pre-render the default case at startup so first visitors get it instantly
threading.Thread(target=lambda: browse(*DEFAULT_VIEW), daemon=True).start()

if __name__ == "__main__":
    # ssr_mode=False -> Python serves the OG-patched index.html, so social
    # link previews show our title/thumbnail instead of Gradio defaults
    demo.launch(ssr_mode=False)
