"""
py2app build config for the standalone, UNSIGNED macOS app (Apple Silicon).

Don't run this by hand — use ./build_app.sh, which makes a clean arm64 venv,
generates the icon, runs py2app, ad-hoc signs, and packages the .dmg. (If you
do run it directly: `pip install py2app && python setup.py py2app`.)

Heads-up: bundling the scientific stack (pyart / scipy / numpy / matplotlib /
netCDF4 / HDF5) with py2app is fiddly — expect to iterate on `packages`,
`includes`, and `excludes`, and to chase missing dylibs/data files. If py2app
fights you, Briefcase (BeeWare) is often smoother for this kind of app.

Apple Silicon only: build with an arm64 Python; the app inherits that arch.
A universal2 (Intel + ARM) build would need Intel wheels too.
"""

from setuptools import setup

APP = ["launcher.py"]

# app.py and its assets live one level up; copy them into the bundle root so
# `import app` / `from sites import SITES` resolve, and logo/cities load.
DATA_FILES = [
    ("", ["../app.py", "../sites.py", "../logo.png", "../cities.json"]),
]

OPTIONS = {
    "argv_emulation": False,
    "packages": [
        "gradio", "gradio_client", "safehttpx", "groovy",
        "pyart", "xradar", "scipy", "numpy", "matplotlib",
        "cmweather", "cmasher", "shapefile", "PIL", "region_dealias",
        "webview", "netCDF4", "cftime",
    ],
    "includes": ["app", "sites"],
    "plist": {
        "CFBundleName": "NEXRAD Level 2",
        "CFBundleDisplayName": "NEXRAD Level 2 — 0.5° browser",
        "CFBundleIdentifier": "edu.illinois.climas.nexrad-l2",
        "CFBundleShortVersionString": "0.1.0",
        "NSHighResolutionCapable": True,
        # data fetches use HTTPS; allow them explicitly to be safe
        "NSAppTransportSecurity": {"NSAllowsArbitraryLoads": True},
    },
    "iconfile": "AppIcon.icns",     # generated from ../logo.png by build_app.sh
}

setup(
    app=APP,
    data_files=DATA_FILES,
    options={"py2app": OPTIONS},
    setup_requires=["py2app"],
)
