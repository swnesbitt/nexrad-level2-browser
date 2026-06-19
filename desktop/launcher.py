"""
Local desktop launcher for the NEXRAD Level 2 browser.

Starts the existing Gradio app on 127.0.0.1 and opens it in a native macOS
window (WKWebView, via pywebview). All radar processing runs locally on this
machine; data is still fetched over the internet (AWS Level 2 archive, the live
chunk feed, IEM warnings, and basemap tiles) unless you add offline caching.

Run it directly:

    pip install -r requirements.txt
    python launcher.py

Notes
-----
* The app's background daemons (startup pre-render + warnings/live pollers) are
  started explicitly via ``app.start_background()`` — they no longer run on
  import, which keeps ProcessPoolExecutor workers safe under macOS 'spawn'.
* ``multiprocessing.freeze_support()`` + the ``__main__`` guard are required so
  spawned decode workers don't re-run this launcher.
"""

import multiprocessing
import os
import socket
import sys
import time
import urllib.request

# project root (the folder containing app.py / sites.py / logo.png / cities.json)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _free_port():
    s = socket.socket()
    try:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]
    finally:
        s.close()


def _wait_until_up(url, timeout=90.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(url, timeout=1)
            return True
        except Exception:
            time.sleep(0.5)
    return False


def main():
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)

    import app          # builds `demo`; daemons stay dormant until we start them
    import webview      # macOS WKWebView wrapper

    port = _free_port()
    app.demo.launch(
        server_name="127.0.0.1",
        server_port=port,
        ssr_mode=False,
        show_api=False,
        inbrowser=False,
        prevent_thread_lock=True,   # return control so we can open the window
        quiet=True,
    )
    app.start_background()          # pre-render default view + warnings/live pollers

    url = f"http://127.0.0.1:{port}/"
    _wait_until_up(url)

    webview.create_window(
        "NEXRAD Level 2 — 0.5° browser",
        url,
        width=1440,
        height=900,
        min_size=(960, 640),
    )
    webview.start()                 # blocks on the main thread until window closes
    os._exit(0)                     # tear down the server + daemon threads


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
