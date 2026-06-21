# Determining the latest NEXRAD Level 2 chunks (hand-off brief)

How the NEXRAD Level 2 browser finds the newest real-time radar data. Written so
another agent can reimplement it. No AWS credentials needed — everything is
anonymous HTTPS GETs against a public S3 bucket.

## Source

Real-time feed: the **`unidata-nexrad-level2-chunks`** S3 bucket (NSF Unidata,
`us-east-1`, public). It carries near-real-time Level 2 for **all ~160 WSR-88D
sites**, published as small "chunks" as each volume scan is still being made —
much lower latency than the assembled-volume archive buckets
(`noaa-nexrad-level2`, `unidata-nexrad-level2`).

Base URL: `https://unidata-nexrad-level2-chunks.s3.amazonaws.com`

## Key layout

```
<SITE>/<VOL>/<YYYYMMDD>-<HHMMSS>-<SEQ>-<TYPE>
        e.g.  KILX/412/20260621-181233-007-I
```

- `SITE`  — 4-letter ICAO (e.g. `KILX`).
- `VOL`   — the radar's **volume-scan counter, an integer that rotates 1→999→1**.
  This is *not* a timestamp; it's a wrapping sequence number.
- chunk filename = `(\d{8})-(\d{6})-(\d{3})-([SIE])` → date, time, sequence, type.
- `TYPE`:  **S** = start (first chunk of the volume), **I** = intermediate,
  **E** = end (last chunk). **An `E` chunk means the volume is complete.**

Concatenating one volume's chunks in sequence order (`S + I… + E`) reconstructs a
valid `AR2V` Level 2 archive file. A volume with no `E` yet is the in-progress
scan — that's the bleeding edge.

## Algorithm: find the latest volume(s)

1. **List the volume folders for the site** with a delimited ListObjectsV2 (one
   cheap call, no full key enumeration):
   `?list-type=2&prefix=<SITE>/&delimiter=/` → the `CommonPrefixes` are the
   `<SITE>/<VOL>/` folders. Parse out the integer `VOL`s.

2. **Order newest→oldest, accounting for the 1–999 wrap.** Plain numeric sort is
   wrong right after the counter rolls over (e.g. `998, 999, 1, 2`). Fix: if the
   set spans the wrap (`max - min > 800`), add 1000 to the low numbers
   (`< 500`) to form an "effective" key, then sort descending by that. So `1, 2`
   rank above `998, 999`.

3. **Probe only the newest few** (the app caps at ~18). For each, list its
   chunks (`?list-type=2&prefix=<SITE>/<VOL>/`), parse + sort by `SEQ`, take the
   first chunk's `YYYYMMDD-HHMMSS` as the volume **start time**, and set
   `complete = any chunk is type E`.

4. **Trailing-window filter.** Keep volumes whose start time is within the last
   `window_min` (default 65 min), newest→oldest; stop once you're well past the
   window (everything older is too).

5. **The "latest" = highest wrap-adjusted `VOL`.** The newest *data* is the
   highest-`SEQ` chunk in that volume; if it has no `E`, the radar is mid-scan
   and more chunks will appear.

## Efficiency / correctness notes (the parts that bite)

- **Cache completed volumes.** A volume with an `E` never changes — cache its
  metadata and never re-list it. Only re-probe the newest and any still-open
  (no-`E`) volumes. This is what lets you poll every ~minute cheaply: one folder
  listing + a re-list of just the growing volume.
- **The wrap is the #1 bug source.** Without the `>800` span check you'll serve
  stale data for ~an hour after each rollover.
- **Incomplete volumes are usable.** Decode the partial `S+I…` set and drop the
  incomplete trailing sweep(s); you get the freshest low-level scan without
  waiting for `E`.
- **Download chunks in parallel** (thread pool) into a local cache keyed by
  `SITE/VOL/`; prune dirs older than ~2 h.
- **Polling cadence:** re-run step 1 every ~60 s. Client-tab timers freeze when a
  tab is backgrounded, so do the polling server-side if you need it continuous.
- **Latency:** chunks appear within tens of seconds of the radar; an assembled
  archive volume in the other buckets lags by minutes.

## Minimal pseudocode

```
def latest_volumes(site, window_min=65):
    vols = list_common_prefixes(f"{site}/", delimiter="/")      # -> [int VOL]
    span = max(vols) - min(vols)
    key  = lambda n: n + 1000 if (span > 800 and n < 500) else n
    for vol in sorted(vols, key=key, reverse=True)[:18]:
        keys = list_keys(f"{site}/{vol}/")                      # chunk objects
        chunks = sorted(parse(k) for k in keys)                 # by SEQ
        start  = datetime_of(chunks[0])
        if age_minutes(start) <= window_min:
            yield dict(vol=vol, start=start,
                       complete=any(c.type == "E" for c in chunks),
                       keys=[c.key for c in chunks])            # S..I..E order
```

Reference implementation: `_s3_page`, `_chunk_list_vol`, `_chunk_recent_vols`,
`_sync_chunks`, `fetch_live_chunks` in `app.py` of
`github.com/swnesbitt/nexrad-level2-browser`.
