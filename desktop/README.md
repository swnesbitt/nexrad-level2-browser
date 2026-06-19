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

## Build a standalone .app

`setup.py` is a **py2app** starting point:

```bash
pip install py2app
python setup.py py2app          # -> dist/NEXRAD Level 2.app
```

Expect to iterate — bundling pyart/scipy/numpy/matplotlib/netCDF4 reliably
usually means adding to `packages`/`includes` and fixing missing dylibs/data
files. If it's painful, try **Briefcase (BeeWare)** instead, which is generally
friendlier for scientific Python + a webview.

## Known hurdles / notes

- **Python 3.11.** Matches the `region-dealias` wheels (cp310/311/312) and the
  Space. If `region-dealias` can't install for your interpreter, the app falls
  back to pyart automatically (slower, identical output).
- **Apple Silicon (arm64).** All wheels ship arm64. A universal2 (Intel+ARM)
  build needs Intel wheels too.
- **App size / first launch.** The scientific stack makes the bundle large
  (~hundreds of MB to ~1 GB) and the first launch slower.
- **Code signing / notarization.** For personal use you can skip it (right-click
  → Open the first time). To distribute it, sign with an Apple Developer ID and
  notarize, or Gatekeeper will block it.
- **multiprocessing.** The decode pool uses macOS 'spawn', which re-imports
  `app` in each worker. `app.start_background()` (not module import) starts the
  daemons so workers don't relaunch them; keep the `__main__` guard +
  `freeze_support()` in `launcher.py`.
- **Dock icon.** Add a block-I `AppIcon.icns` and uncomment `iconfile` in
  `setup.py`.
