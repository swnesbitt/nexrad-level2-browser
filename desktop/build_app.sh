#!/usr/bin/env bash
#
# Build a standalone, UNSIGNED macOS app + .dmg (Apple Silicon only).
# No paid Apple Developer ID needed — the app is ad-hoc signed so it launches;
# users do a one-time "Open Anyway" (see FIRST_OPEN.txt).
#
# Run ON an Apple Silicon Mac, with an arm64 Python 3.11:
#     cd desktop
#     bash build_app.sh
#
# Outputs:  dist/NEXRAD Level 2.app   and   NEXRAD-Level-2.dmg
#
set -euo pipefail
cd "$(dirname "$0")"

APPNAME="NEXRAD Level 2"

# 0) require Apple Silicon / arm64 Python ------------------------------------
ARCH="$(python3 -c 'import platform; print(platform.machine())')"
if [ "$ARCH" != "arm64" ]; then
  echo "ERROR: this build targets Apple Silicon only, but python3 is '$ARCH'." >&2
  echo "Use an arm64 Python 3.11 (e.g. from python.org or 'arch -arm64 brew')." >&2
  exit 1
fi

# 1) clean build venv with the app deps + py2app -----------------------------
rm -rf .build-venv build dist
python3 -m venv .build-venv
source .build-venv/bin/activate
python -m pip install --upgrade pip wheel
pip install -r requirements.txt          # -> inherits ../requirements.txt
pip install py2app

# 2) app icon (block-I) from the CliMAS logo --------------------------------
ICONSET="$(mktemp -d)/AppIcon.iconset"; mkdir -p "$ICONSET"
for s in 16 32 64 128 256 512; do
  sips -z "$s" "$s"        ../logo.png --out "$ICONSET/icon_${s}x${s}.png"   >/dev/null
  d=$((s*2)); sips -z "$d" "$d" ../logo.png --out "$ICONSET/icon_${s}x${s}@2x.png" >/dev/null
done
iconutil -c icns "$ICONSET" -o AppIcon.icns

# 3) build the .app (py2app ad-hoc signs it for Apple Silicon) ---------------
python setup.py py2app

# 4) ad-hoc re-sign the whole bundle (belt-and-suspenders) ------------------
codesign --force --deep --sign - "dist/$APPNAME.app" || \
  echo "warn: codesign step failed; the app may need right-click->Open."

# 5) package a drag-to-Applications .dmg with the read-me --------------------
STAGE="$(mktemp -d)/dmg"; mkdir -p "$STAGE"
cp -R "dist/$APPNAME.app" "$STAGE/"
ln -s /Applications "$STAGE/Applications"
cp FIRST_OPEN.txt "$STAGE/How to open (read me first).txt"
rm -f "NEXRAD-Level-2.dmg"
hdiutil create -volname "$APPNAME" -srcfolder "$STAGE" -ov -format UDZO \
  "NEXRAD-Level-2.dmg"

echo
echo "Done."
echo "  app : dist/$APPNAME.app"
echo "  dmg : NEXRAD-Level-2.dmg   (share this)"
echo "First-launch instructions for users are in FIRST_OPEN.txt / the .dmg."
