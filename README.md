---
title: NEXRAD level 2 browser
emoji: 📡
colorFrom: red
colorTo: indigo
sdk: gradio
sdk_version: 5.34.2
python_version: "3.11"
app_file: app.py
pinned: false
license: mit
thumbnail: https://huggingface.co/spaces/snesbitt/nexrad-level2-browser/resolve/main/thumbnail.png
---

# NEXRAD Level 2 — 0.5° sweep browser

Interactive map browser for historical NEXRAD Level 2 base (0.5°) scans,
including SAILS / MESO-SAILS re-inserted sweeps.

- Pick a WSR-88D site, field, and UTC hour; the app pulls every Level 2
  volume for that hour from the [AWS Open Data NEXRAD archive](https://registry.opendata.aws/noaa-nexrad/)
  (NOAA bucket, Unidata mirror fallback).
- Every sweep with fixed angle < 0.75° is rendered with
  [Py-ART](https://arm-doe.github.io/pyart/) to a georeferenced transparent
  overlay (web mercator) — so SAILS cuts appear as extra time steps.
- Frames are shown on a Leaflet map with a time slider, play/pause, and
  opacity control. Reflectivity uses the **ChaseSpectral** colormap
  (via [cmweather](https://cmweather.readthedocs.io/)).

Fields: reflectivity, radial velocity, differential reflectivity,
correlation coefficient (dual-pol fields require ~2011+ data).
