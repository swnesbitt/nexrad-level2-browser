# Plan: a Rust port of Py‑ART `dealias_region_based`

Goal: reimplement Py‑ART's region‑based velocity dealiasing in Rust so it runs
faster, lives in its own repository, and drops into the NEXRAD Level‑2 browser
in place of `pyart.correct.dealias_region_based` — producing **numerically
identical** results to the current Python path.

---

## 1. Objective and constraints

- **Identical output**, not "close." Because the algorithm's output is
  `velocity + (integer folds) × (2·Nyquist)`, identical fold integers give
  *bit‑exact* corrected velocities on every unmasked gate. The bar is exact
  equality, verified against Py‑ART on real volumes.
- **Faster.** The hot path today is pure‑Python object‑array bookkeeping (the
  `_EdgeTracker` network reduction). That is where Rust wins.
- **Standalone repo**, reusable beyond this app, with a pip‑installable wheel.
- **Drop‑in** for the Space with a clean fallback to Py‑ART.

### What we actually have to match (big simplification)

The app calls:

```python
pyart.correct.dealias_region_based(sub, vel_field="velocity", keep_original=False)
```

Every other argument is default, and crucially `ref_vel_field=None`. That means
the **entire reference‑anchoring branch is never executed**: no
`scipy.optimize.fmin_l_bfgs_b`, no `_cost_function`, no `_gradient`, no second
`_find_regions`. So v1 of the Rust port can target exactly the path the app
uses and be identical for the app, while the (rarely used) sounding‑anchored
branch is a clearly scoped Phase‑2 add‑on.

The path we must reproduce, per sweep:

1. `_parse_nyquist_vel` → per‑sweep Nyquist (from the radar).
2. gatefilter = exclude masked + invalid velocity gates → boolean mask.
3. `nyquist_interval = 2 · nyquist`.
4. `_find_sweep_interval_splits` → interval limits via `np.linspace`
   (`interval_splits=3`, plus extra bins if data exceeds ±Nyquist).
5. `_find_regions` → connected‑component labels (loops the velocity bins,
   `scipy.ndimage.label` per bin, offsets the label numbers).
6. `_edge_sum_and_count` → `_fast_edge_finder` (Cython) enumerates region
   adjacencies; then dedup via `lexsort` + `add.reduceat`.
7. `_RegionTracker` + `_EdgeTracker` + `_combine_regions` loop → the dynamic
   network reduction that assigns a fold count to each region.
8. `centered` global offset so mean folds ≈ 0.
9. `scorr += nwrap × nyquist_interval`; mask filtered gates.

---

## 2. Architecture decision

**Recommendation: a standalone Rust crate with PyO3/maturin Python bindings,
published as a wheel.** Not a pyart fork, not folded into the xradar fork.

| Option | Pros | Cons | Verdict |
|---|---|---|---|
| **Fork Py‑ART**, swap the function for Rust | true drop‑in; users keep `pyart.correct.…` | must track a large upstream; adds a Rust toolchain to pyart's build; you now own a heavyweight fork for one function; painful to rebase | ✗ overkill |
| **Merge into your `xradar` fork** | one fewer repo; you already maintain it | xradar is an *I/O / xarray* library — dealiasing is out of scope; bloats a fork whose value is the LDM/signed‑int IO fixes; forces a Rust build dep onto a pure‑Python package; muddies future upstream PRs | ✗ wrong home |
| **Standalone crate + PyO3 wheel** | single responsibility; reusable; clean CI + wheels; trivial to validate against pyart; app depends on it as one pip line; can be optimized independently (rayon) | one more repo; must build/publish wheels for the target platform | ✓ **recommended** |

The dealiaser only needs plain arrays (a 2‑D velocity sweep, a mask, a Nyquist
value, a wrap‑around flag). It does **not** need pyart's `Radar` internals.
That clean boundary is exactly what makes a standalone library the right shape.

---

## 3. Repository design

Proposed name: **`region-dealias`** (crate) / **`region_dealias`** (Python pkg).

```
region-dealias/
├── Cargo.toml                 # workspace
├── crates/
│   ├── region-dealias-core/   # pure Rust, no Python — the algorithm
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── label.rs        # scipy.ndimage.label replica (4-conn, raster order)
│   │   │   ├── intervals.rs    # _find_sweep_interval_splits + linspace
│   │   │   ├── regions.rs      # _find_regions
│   │   │   ├── edges.rs        # _fast_edge_finder + dedup (lexsort/reduceat)
│   │   │   ├── network.rs      # _RegionTracker, _EdgeTracker, _combine_regions
│   │   │   ├── sweep.rs        # one-sweep driver (steps 3–9)
│   │   │   └── numpy_compat.rs # banker's rounding, f32/f64 helpers
│   │   └── tests/              # golden vectors vs pyart
│   └── region-dealias-py/      # PyO3 bindings (maturin)
│       └── src/lib.rs
├── python/region_dealias/
│   ├── __init__.py            # low-level array API + radar-compat wrapper
│   └── _fallback.py           # import pyart.correct.dealias_region_based
├── tests/                     # pytest: parity harness vs pyart
├── benches/                   # criterion benchmarks
├── .github/workflows/
│   ├── ci.yml                 # build + parity tests (installs pyart)
│   └── wheels.yml             # maturin/cibuildwheel manylinux + macOS/arm
└── pyproject.toml             # maturin backend
```

### Public API (two layers)

Low‑level, array‑in / array‑out (what core exposes, what tests hit):

```python
folds = region_dealias.sweep_folds(
    vel: np.ndarray[float32, (nrays, ngates)],
    mask: np.ndarray[bool],          # True = excluded
    nyquist: float,
    rays_wrap_around: bool,
    interval_splits: int = 3,
    skip_between_rays: int = 100,
    skip_along_ray: int = 100,
    centered: bool = True,
) -> np.ndarray[int32]               # per-gate fold count
```

High‑level, **drop‑in** mirror of pyart's signature so the app barely changes:

```python
from region_dealias import dealias_region_based   # same name, same kwargs
corr = dealias_region_based(sub, vel_field="velocity", keep_original=False)
# returns the same {'data', '_FillValue', 'valid_min', 'valid_max'} dict
```

The wrapper pulls `vel`, the per‑sweep Nyquist, `scan_type`, and the
masked/invalid gates out of the `Radar` object (replicating pyart's
`_parse_*` helpers) and calls the Rust core per sweep.

---

## 4. Port scope and the correctness‑critical details

These are the spots where a naïve rewrite silently diverges. Each must be a
deliberate, tested replica.

1. **`scipy.ndimage.label` replica.** Default 2‑D connectivity is the
   4‑neighbour cross; labels are assigned in **C‑order raster scan of first
   encounter**. Region *numbers* flow into region sizes, edge node indices, and
   ultimately the `argmax`/merge order — so the numbering must match exactly,
   not just the partition. Implement a two‑pass (or union‑find) labeler that
   reproduces scipy's first‑encounter ordering.

2. **`_fast_edge_finder` enumeration.** Port the exact scan: for each non‑zero
   gate in raster order, probe left/right/top/bottom, skipping up to
   `max_gap_x`/`max_gap_y` masked gates, with ray wrap‑around on the x‑axis.
   Edge tuples are collected in this order. Velocities are stored as **float64**
   here even though the field is cast to float32 for probing.

3. **Edge dedup.** `lexsort((index1, index2))` then `add.reduceat` to sum
   `vel1`, `vel2`, `count` per unique `(i, j)`. The resulting order feeds
   `_EdgeTracker`, and the init does `if i < j: continue` (keeps one direction).
   Replicate the sort key and the reduceat summation order.

4. **Dtype chain (must match step by step).** field → `float32`; collector
   velocities → `float64`; per‑edge `sum_diff = (vel − nvel)/nyquist_interval`
   stored as **float32** and accumulated in float32 across merges; interval
   limits are float64 from `linspace`; `float32 < float64` comparisons upcast to
   float64. Mirror each.

5. **NumPy rounding = banker's rounding.** `int(np.round(diff))` and the
   `centered` `int(round(total_folds/gates))` use round‑half‑to‑**even**. Rust's
   `f64::round` is half‑away‑from‑zero — wrong. Provide a `round_half_even`.

6. **`argmax` tie‑break.** `np.argmax(weight)` returns the **first** max. The
   network reduction's merge order depends on it, so the Rust pop must return
   the lowest‑index max (not an arbitrary heap top). (A real priority queue is a
   later optimization — see §7 — and must preserve this tiebreak.)

7. **`_combine_regions` merge rules.** Size‑tie goes to the `else` branch
   (base = node2, `rdiff = −rdiff`); `_reverse_edge_direction`, `_combine_edges`,
   and the `_common_finder` reuse logic must be ported faithfully.

8. **`centered` offset and interval‑split edge cases** (velocities beyond
   ±Nyquist add bins; all‑masked sweeps skip; `nfeatures < 2` skip).

Phase‑2 (only if/when the app ever uses `ref_vel_field`): port the
`fmin_l_bfgs_b` anchoring. This needs an L‑BFGS‑B with identical convergence —
hard to match bit‑for‑bit; likely keep that branch in Python/scipy and call
Rust only for the core. Flagged as out of scope for v1.

---

## 5. Bit‑identical validation (the heart of the project)

Identity is a *testable property*, so we make it CI‑enforced:

- **Golden corpus.** A set of real NEXRAD volumes spanning: multiple VCPs
  (212/215/12/121/MPDA), a range of Nyquists, SAILS/MRLE split cuts, legacy
  pre‑2008 MSG‑1, clear‑air vs widespread aliasing, and the app's known cases
  (KILX 2023‑06‑29 derecho; KMLB Frances 2004). Store small extracted sweep
  arrays (`vel`, `mask`, `nyquist`, `wrap`) as `.npz` fixtures so tests don't
  need network or pyart's IO.
- **Parity harness (pytest).** For each fixture, run pyart's
  `dealias_region_based` and `region_dealias` on identical inputs and assert:
  - `np.array_equal(folds_rust, folds_pyart)` (the integer fold field), and
  - bit‑exact equality of corrected velocities on unmasked gates.
- **CI gate.** `ci.yml` installs `arm-pyart` + `scipy`, builds the Rust wheel,
  runs the harness; a single mismatch fails the build. Pin the pyart version
  the parity is certified against (record it in the README) and re‑run on bumps.
- **Property/fuzz tests.** Random synthetic sweeps (random regions, Nyquists,
  wrap on/off) to surface ordering/rounding divergences the corpus misses.
- **Intermediate‑stage tests.** Don't only compare final output — compare the
  *labels*, the *edge list*, and the *unwrap numbers* against instrumented
  pyart, so a divergence is localized to one module instead of "the answer is
  off."

---

## 6. Integrating into the Space

- **Distribute as a prebuilt wheel.** The HF Space installs from
  `requirements.txt` and shouldn't need a Rust toolchain at build time. Build
  **manylinux2014 x86_64, CPython 3.11** wheels (the Space pins
  `python_version: 3.11`, free CPU is x86_64 linux) via `maturin` +
  `cibuildwheel`/`maturin-action` on GitHub Actions; also build **macOS arm64**
  for local dev. Prefer **abi3** (`cp311‑abi3`) so one wheel covers 3.11+.
- **Publish to PyPI** → add `region-dealias>=x.y` to `requirements.txt`. (Avoid
  `pip install git+…`: that forces a source build on HF and may lack `cargo`.)
  Alternative if PyPI isn't wanted: host the wheel as a GitHub Release asset and
  reference its URL in `requirements.txt`.
- **App change is tiny and safe.** Replace the two call sites
  (`dealias_volume` and `dealias_volume_file`) with:

  ```python
  try:
      from region_dealias import dealias_region_based
  except Exception:
      from pyart.correct import dealias_region_based   # fallback, identical
  ```

  Because the wrapper mirrors pyart's signature and output dict, nothing else
  changes. Keep the fallback permanently so a wheel/ABI problem degrades to
  today's behavior instead of breaking live/dealias mode.
- **Parallelism already exists upstream.** The app dealiases volumes in a
  `ProcessPoolExecutor`. The Rust lib can additionally parallelize **across
  sweeps** with `rayon` (each sweep is independent), but keep a single‑thread
  mode for the parity tests so ordering is deterministic.

---

## 7. Performance approach and expected gains

- **Where Python is slow today:** `_find_regions`/`_fast_edge_finder` are
  already C/Cython, but `_EdgeTracker` + `_combine_regions` is **pure Python over
  `object`‑dtype arrays and Python lists**, doing an `np.argmax` over all edges
  on every pop — roughly O(E²) in the worst case and the dominant cost on busy
  (heavily aliased) sweeps. This is the prize.
- **v1 (identical):** port everything straight, keep `argmax`‑first‑tie
  semantics. Even a literal translation removes the Python‑object overhead and
  should give a large speedup on the edge‑tracker loop while staying exact.
- **v2 (optional, still identical):** replace the linear `argmax` with a
  max‑heap that breaks ties by lowest edge index (reproducing `argmax`), the
  priority‑queue improvement pyart itself notes as a TODO; add `rayon` across
  sweeps. Gate every optimization behind the parity harness so "faster" can
  never mean "different."
- **Benchmark** with `criterion` on the golden corpus and report end‑to‑end
  per‑volume wall‑clock vs pyart on the Space's 2‑vCPU profile (the metric that
  actually matters for the live dealias path).

---

## 8. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Subtle numeric divergence (rounding, dtype, ordering) | per‑stage parity tests + CI gate; replicate dtypes/rounding deliberately (§4) |
| `scipy.ndimage.label` ordering mismatch | dedicated label‑parity test before anything downstream |
| pyart changes the algorithm upstream | pin & document the certified pyart version; re‑run parity on bumps |
| HF build can't compile Rust | ship prebuilt manylinux wheel; never source‑build on HF |
| ABI/wheel mismatch on the Space | abi3 wheel; permanent pyart fallback import |
| `ref_vel_field` path needed later | scoped Phase‑2; likely keep scipy L‑BFGS‑B in Python |

---

## 9. Roadmap and rough effort

1. **Scaffold** the repo, crate workspace, maturin/PyO3, CI skeleton. *(~0.5 day)*
2. **Fixture + harness** first: extract golden sweeps, wire the pyart parity
   test (red before any port). *(~1 day)*
3. **Port core**, module by module, each green against per‑stage fixtures:
   label → intervals → regions → edges → network → sweep driver. *(~3–5 days)*
4. **Radar‑compat wrapper** + full‑volume parity on the corpus. *(~1 day)*
5. **Wheels + PyPI**, integrate into the Space behind the fallback, benchmark.
   *(~1 day)*
6. **(Optional) v2** priority queue + rayon, re‑verified identical. *(~1–2 days)*

Total for an identical, deployed v1: roughly **1–1.5 weeks** of focused work,
front‑loaded on the validation harness so "identical" is provable at every step.

---

## 10. Decisions (confirmed)

- **Distribution:** publish to **PyPI** under your account; `requirements.txt`
  pins `region-dealias>=x.y`.
- **Targets:** build wheels for **linux‑x86_64** (the Space) **and macOS‑arm64**
  (local dev) from day one — cp311 abi3.
- **Sounding‑anchored (`ref_vel_field`) path:** **not developed** — formally out
  of scope for the Rust library. The app doesn't use it; if it's ever needed it
  stays in Python/scipy and calls Rust only for the core.
- **Name:** `region-dealias` (crate) / `region_dealias` (package) — confirmed.
