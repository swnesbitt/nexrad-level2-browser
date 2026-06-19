# NEXRAD Level 2 browser — local macOS app

Run the whole thing on your Mac: the Gradio backend starts on `127.0.0.1` and
opens in a native window (WKWebView). All **processing** (decode, dealias,
texture packing) happens locally — which means full CPU/RAM, warm caches, a
persistent live poller, and none of the Hugging Face rebuild/startup delays.

> **"Local" ≠ offline.** The app still downloads radar **data** over the
> internet: Level 2 volumes from AWS, the live chunk feed, IEM warnings, and the
> CARTO/Esri basemap tiles. True offline would need pre-cached volumes + an
> offline basemap (not included).

## Run it now (no packaging)

From this `desktop/` folder, in a Python 3.11 environment:

```bash
pip install -r requirements.txt
python launcher.py
```

A window titled "NEXRAD Level 2 — 0.5° browser" opens with the full app.
First launch pre-renders the default case (KILX derecho), same as the Space.

(`launcher.py` imports the existing `../app.py`, launches Gradio on a free local
port, calls `app.start_background()` for the pre-render + warnings/live pollers,
then opens the webview.)

## Build the installer (free, unsigned — Apple Silicon)

No paid Apple Developer ID required. On an **Apple Silicon Mac** with an
**arm64 Python 3.11**:

```bash
cd desktop
bash build_app.sh
```

That makes a clean build venv, generates the block-I app icon from `../logo.png`,
runs py2app, ad-hoc signs the bundle (so it launches on Apple Silicon), and
packages a drag-to-Applications disk image:

- `dist/NEXRAD Level 2.app`
- `NEXRAD-Level-2.dmg`  ← give this to your students

**Distribute:** share `NEXRAD-Level-2.dmg`. Because it isn't notarized, the
first launch shows an "unidentified developer" warning — `FIRST_OPEN.txt` (also
copied into the .dmg as "How to open (read me first).txt") walks students
through the one-time **right-click → Open** / **System Settings → Open Anyway**
step. After that it opens like any app.

**Expect to iterate the first build.** Bundling pyart/scipy/numpy/matplotlib/
netCDF4 reliably usually means adding to `packages`/`includes` in `setup.py` and
chasing missing dylibs/data files. To debug, run the built binary from a
terminal so you see the import error:

```bash
"dist/NEXRAD Level 2.app/Contents/MacOS/NEXRAD Level 2"
```

If py2app keeps fighting you, **Briefcase (BeeWare)** is often friendlier for
scientific Python + a webview (it can also produce a `.dmg`).

A truly warning-free install needs notarization, which requires a paid Developer
ID ($99/yr) or your university's Apple Developer account — see the options note
at the end.

## Keeping the HF Space and the Mac app in sync

They share a single source of truth, so an update lands in both:

- **App code & UI:** all in `../app.py`. The Space runs it as `__main__`; the
  Mac launcher `import app` and calls `app.demo.launch(...)` + `app.start_background()`.
  Edit `app.py` once → it applies to both. The launcher relies on only that
  two-symbol interface (`demo`, `start_background`) and checks for it at startup.
- **Dependencies:** defined once in `../requirements.txt` (used by the Space).
  `desktop/requirements.txt` does `-r ../requirements.txt` and adds only the
  webview wrapper — so versions can't drift.
- **Dealiaser:** both use the published `region-dealias` wheel, with the same
  pyart fallback.

**Update workflow:** make the change in `app.py` (and `requirements.txt` if a
dep changed), commit/push → the Space redeploys automatically; on the Mac,
`git pull` then `python launcher.py` (or rebuild the `.app`). Nothing app-level
is duplicated between the two, so they stay identical by construction.

The only build-specific files are in this `desktop/` folder
(`launcher.py`, `requirements.txt`, `setup.py`) — they wrap the app, they don't
fork it.

## Known hurdles / notes

- **Python 3.11.** Matches the `region-dealias` wheels (cp310/311/312) and the
  Space. If `region-dealias` can't install for your interpreter, the app falls
  back to pyart automatically (slower, identical output).
- **Apple Silicon (arm64).** All wheels ship arm64. A universal2 (Intel+ARM)
  build needs Intel wheels too.
- **App size / first launch.** The scientific stack makes the bundle large
  (~hundreds of MB to ~1 GB) and the first launch slower.
- **Code signing / notarization.** This build is intentionally **unsigned/free**
  (ad-hoc signed only). Distribution options without paying:
  - *Free (this build):* share the `.dmg`; users do the one-time "Open Anyway"
    step in `FIRST_OPEN.txt`. Fully functional.
  - *Zero-install alternative:* point students at the Hugging Face Space URL (a
    browser, no install) — uses the Space's compute, not local.
  - *Warning-free, still free to you:* if your university has an Apple Developer
    membership, IT can sign + notarize under it.
  - *Paid:* an Apple Developer ID ($99/yr) lets you notarize yourself.
- **multiprocessing.** The decode pool uses macOS 'spawn', which re-imports
  `app` in each worker. `app.start_background()` (not module import) starts the
  daemons so workers don't relaunch them; keep the `__main__` guard +
  `freeze_support()` in `launcher.py`.
- **Dock icon.** Add a block-I `AppIcon.icns` and uncomment `iconfile` in
  `setup.py`.
