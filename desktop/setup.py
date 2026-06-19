"""
py2app build config — a STARTING POINT for a standalone .app.

    pip install py2app
    python setup.py py2app           # -> dist/NEXRAD Level 2.app

Heads-up: bundling the scientific stack (pyart / scipy / numpy / matplotlib /
netCDF4 / HDF5) with py2app is fiddly — expect to iterate on `packages`,
`includes`, and `excludes`, and to chase missing dylibs/data files. If py2app
fights you, Briefcase (BeeWare) is often smoother for this kind of app.

This builds on Apple Silicon (arm64); region-dealias and the scientific wheels
all ship arm64. A universal2 build needs Intel wheels too.
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
    # "iconfile": "AppIcon.icns",   # drop in a block-I .icns for the dock icon
}

setup(
    app=APP,
    data_files=DATA_FILES,
    options={"py2app": OPTIONS},
    setup_requires=["py2app"],
)
