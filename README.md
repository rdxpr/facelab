# FaceLab

A local **benchmarking and hyperparameter-tuning harness for classroom
attendance**: enroll a roster, run several face detectors and recognizers over
the same classroom frames, and compare them on calibrated attendance F1,
size-stratified recall, and latency.

FaceLab runs the face pipeline locally and has no dependency on any backend —
the engine is vendored into `facelab/`, so this repository clones, builds, and
runs on its own.

```
Enroll (per recognizer)            Scan (detector + recognizer)        Benchmark
─────────────────────────          ────────────────────────────        ─────────
reference photo → embeddings  →    frames → detect → align → embed  →  cached artifacts
stored per recognizer id           → match → present/review/unknown    → calibrate τ, score grid
```

## Highlights

- **Detectors and recognizers are independent.** Pick any detector (SCRFD,
  RetinaFace, YOLOv8-Face, YuNet) with any recognizer (ArcFace R50/R100,
  MobileFaceNet, AdaFace, EdgeFace, SphereFace, Raccoon-L). No more "one pack per
  run".
- **SAHI for back rows.** Optional slicing-aided hyper-inference tiles the upper
  band or the whole frame so small back-row faces survive detection.
- **Calibrated thresholds.** Similarity scores are not comparable across
  backbones, so benchmarks sweep the accept threshold on labelled data and report
  the operating point that maximises F1 (τ\*), the lowest threshold with zero
  false attendances, and the lowest with precision ≥ 0.98.
- **Stratified recall.** Detection and identification recall are split into
  large / medium / small face buckets so you can see the back-row cost directly.
- **Persistent inference cache.** Every scan stores detections, aligned crops,
  quality, pose, and raw vectors per `(image, detector, params, recognizer)`. Once
  cached, threshold/margin/fusion sweeps run in pure NumPy — the UI recolors a
  scan live as you drag the sliders, with no re-inference.
- **Ground-truth datasets.** Build reusable labelled datasets; sweeps never
  re-upload images.
- **Storage modes.** Local development persists everything under `./.data`. Any
  non-`dev` `FACELAB_ENV` switches to ephemeral storage that is wiped on start.

## Quick start (local)

Requires Python 3.11+ and, optionally, [uv](https://docs.astral.sh/uv/).

```bash
# 1. Install dependencies (uv is fastest)
uv sync                       # or: python -m venv .venv && .venv/bin/pip install -r requirements.txt

# 2. Make sure a detector + recognizer exist (skips if already present)
.venv/bin/python scripts/fetch_models.py                       # buffalo_l (SCRFD-10G + ArcFace R50)
.venv/bin/python scripts/fetch_models.py --yunet               # + OpenCV YuNet (Apache-2.0, 230 KB)
.venv/bin/python scripts/fetch_models.py --all                 # every pack + YuNet + UniFace weights

# 3. Run
./run.sh                      # http://127.0.0.1:8010
```

`./run.sh` prefers the project's `.venv`, then `uv run`, then plain `python3`.

> **Why a single model by default?** The default fetch pulls only `buffalo_l`,
> and a model is only selectable once its weights are on disk (or, for UniFace,
> once `uniface` is installed). Run `fetch_models.py --all` to populate the whole
> benchmark matrix — it is a large download (hundreds of MB). `--list` shows what
> is available.

### Optional: UniFace models

`uniface` unlocks the AdaFace / EdgeFace / SphereFace recognizers plus the
eDifFIQA quality and 6D head-pose predictors. Everything else works without it.

```bash
uv pip install --python .venv/bin/python "uniface[cpu]"   # or [gpu] on CUDA
.venv/bin/python scripts/fetch_models.py --uniface all
```

Without `uniface`, the registry marks those models unavailable and the quality
score falls back to the pre-L2 embedding norm.

## Using the UI

The interface has three tabs.

### Attendance

1. **Students** — add a name and one or more single-face photos. Each photo is
   embedded once per available recognizer, so switching backbones never requires
   re-enrolling.
2. **Classroom scan** — choose a detector, a recognizer, a detection resolution,
   and optional SAHI tiling. Tune the detection threshold, NMS IoU, accept and
   review thresholds, the top1–top2 margin gate, and the assignment strategy.
3. **Results** — the roster splits into *present* / *review* / *absent*; frames
   show green (present), amber (review), and red (unknown) boxes. Click any box
   for the **Face Inspector**: the aligned crop, size bucket, face height,
   detector score, quality, head pose, and top-3 candidates with scores and
   margins.

Moving any matching slider (accept, review, margin, assignment, fusion, …)
re-scores the loaded scan from cache instantly — no re-inference.

### Datasets

Build a labelled dataset from classroom frames: mark the students who are
actually present, add notes, and save. Datasets are the ground truth benchmarks
calibrate against and are reused across sweeps.

### Benchmark

Select datasets, a grid of detectors × recognizers × resolutions × tiling modes,
and the operating parameters, then run. The **Model matrix** reports, per
combination:

| Column | Meaning |
| ------ | ------- |
| `F1` / `Prec` / `Rec` | Attendance precision/recall/F1 at the chosen threshold |
| `τ*` | Threshold that maximises F1 |
| `L/M/S recall` | Identification recall for large / medium / small faces |
| `det p50` / `total p50` | Detection and total inference latency (ms) |

Click a row for the threshold curve (precision/recall/F1 vs τ), the zero-false-
positive and precision≥0.98 operating points, and per-bucket recall.

## How it works

1. **Detect** — the chosen detector runs on the frame (optionally across SAHI
   tiles), merged with NMS.
2. **Align + embed** — 5-point landmarks drive a 112×112 aligned crop; the
   recognizer returns a raw vector, plus optional flip-averaged and low-res-
   augmented variants.
3. **Match** — `facelab/matching.py` builds a face×student similarity matrix from
   per-student templates and turns it into `present` / `needs_review` /
   `unmatched` decisions with an optional margin gate and Hungarian/greedy
   assignment.
4. **Fuse** — a burst of frames is merged by k-of-n voting, a max-with-margin
   gate, or quality-weighted embedding clustering.
5. **Score** — benchmarks calibrate the threshold and compute attendance plus
   stratified detection/identification recall from cached embeddings.

### Result JSON shape

```jsonc
{
  "scanId": "20261113-183907-671f5b",
  "createdAt": "2026-11-13T18:39:07+00:00",
  "detectorId": "insightface:scrfd_10g_bnkps",
  "recognizerId": "insightface:w600k_r50",
  "summary": {
    "frames": 2, "facesDetected": 11, "matchedCount": 3, "unknown": 4,
    "acceptThreshold": 0.42, "reviewThreshold": 0.34, "minMargin": 0.04,
    "detSize": 1280, "tilingMode": "none", "variant": "flip_avg",
    "bestIndex": 0, "elapsedMs": 8430,
    "timings": { "detect_ms": 210.4, "align_embed_ms": 2400.1, "match_ms": 11.2 }
  },
  "people": [ { "personId": 1, "number": 1, "name": "…", "status": "present",
                "confidence": 0.62, "framesHit": 2, "templates": 2 } ],
  "frames": [ { "order": 0, "width": 1280, "height": 960,
                "imageUrl": "/media/scans/…/frame-0.jpg",
                "annotatedUrl": "/media/scans/…/frame-0-annotated.jpg",
                "cacheKey": "…",
                "faces": [ { "box": [x1, y1, x2, y2], "status": "present",
                             "personId": 1, "confidence": 0.62, "margin": 0.31,
                             "sizeBucket": "medium", "quality": 0.78,
                             "cropUrl": "/media/crops/…_det0.jpg",
                             "topCandidates": [ {"studentId": 1, "score": 0.62} ] } ] } ],
  "warnings": []
}
```

## API

| Method | Path | Purpose |
| ------ | ---- | ------- |
| `GET`  | `/api/health` | Liveness + installed model names |
| `GET`  | `/api/state` | Roster, totals, settings, registry |
| `GET`  | `/api/registry` | Detector/recognizer availability, license, metrics |
| `GET`  | `/api/models` | Installed packs with template/people counts |
| `GET`  | `/api/model?detector=&recognizer=` | Load (or check) a model pair |
| `GET`  | `/api/cache` | Inference cache size |
| `POST` | `/api/people` | Enroll a student (`name`, `recognizer`, `images[]`) |
| `POST` | `/api/people/{id}/photos` | Add photos (`recognizer`, `images[]`) |
| `POST` | `/api/students/re-embed` | Backfill embeddings for other recognizers |
| `PATCH`| `/api/people/{id}` | Rename a student |
| `DELETE`| `/api/people/{id}` · `/api/people/{id}/photos/{photoId}` | Remove a student / photo |
| `POST` | `/api/scan` | Scan frames (detector, recognizer, thresholds, `images[]`) |
| `GET`  | `/api/results` | List saved runs |
| `GET`  | `/api/results/{scanId}` | Full result JSON |
| `POST` | `/api/results/{scanId}/sweep` | Re-score from cache with new parameters |
| `DELETE`| `/api/results/{scanId}` | Delete a run and its frames |
| `GET`/`POST`/`DELETE` | `/api/datasets[/{id}]` | List, build, delete ground-truth datasets |
| `POST` | `/api/benchmark/run` | Run a model grid over datasets |
| `POST` | `/api/benchmark/sweep` | Re-score datasets (cached) with new parameters |
| `GET`  | `/api/benchmarks[/{runId}]` | List / fetch benchmark runs |
| `POST` | `/api/reset` | Clear roster, scans, datasets, and results |

Interactive API docs are at `/docs` (FastAPI). Set `FACELAB_API_TOKEN` to
require `Authorization: Bearer <token>` (or `X-API-Token`) on mutating routes.

## Configuration

Every setting is optional and uses a `FACELAB_` prefix; see `.env.example`. The
most useful ones:

| Variable | Default | Notes |
| -------- | ------- | ----- |
| `FACELAB_DEFAULT_DETECTOR` | `insightface:scrfd_10g_bnkps` | Default detector id |
| `FACELAB_DEFAULT_RECOGNIZER` | `insightface:w600k_r50` | Default recognizer id |
| `FACELAB_FACE_SCAN_DET_SIZE` | `1280` | Scan detector input size |
| `FACELAB_ACCEPT_THRESHOLD` | `0.42` | Present if score ≥ accept and margin passes |
| `FACELAB_REVIEW_THRESHOLD` | `0.34` | Flag for review if score ≥ review |
| `FACELAB_MIN_MARGIN` | `0.04` | Minimum top1−top2 margin for "present" |
| `FACELAB_TILING_MODE` | `none` | `none`, `sahi_upper_band`, `sahi_full` |
| `FACELAB_ASSIGNMENT_MODE` | `hungarian` | `hungarian`, `greedy`, `margin` |
| `FACELAB_MULTI_FRAME_FUSION` | `k_of_n_voting` | Burst fusion strategy |
| `FACELAB_BENCHMARK_DET_SIZES` | `640,1024,1280` | Sweep resolutions |
| `FACELAB_ONNX_PROVIDERS` | auto | `CUDAExecutionProvider,CPUExecutionProvider` |
| `FACELAB_API_TOKEN` | *(empty)* | Require a bearer token on writes |
| `FACELAB_CORS_ORIGINS` | localhost | Comma-separated allowlist |

## Running on a server

`FACELAB_ENV` decides whether uploads survive a restart:

| `FACELAB_ENV` | Data location | Behaviour |
| ------------- | ------------- | --------- |
| `dev` (default) | `./.data` | Roster, photos, frames, datasets, results persist |
| anything else | `FACELAB_EPHEMERAL_DIR` (default `<temp>/facelab`) | wiped on every start |

```bash
FACELAB_ENV=prod FACELAB_EPHEMERAL_DATA=false ./run.sh   # persist despite prod
FACELAB_ENV=dev  FACELAB_EPHEMERAL_DATA=true  ./run.sh   # wipe despite dev
```

> **Privacy.** Without `FACELAB_API_TOKEN` there is no authentication: anyone who
> can reach the port can enroll faces, view results, or call `/api/reset`. Keep it
> on localhost or a private network. InsightFace pretrained models and several
> UniFace weights are licensed for non-commercial research use.

### Docker

```bash
docker build -t facelab .
docker run --rm -p 8010:8010 facelab                              # ephemeral
docker run --rm -p 8010:8010 -v facelab-data:/data \
  -e FACELAB_EPHEMERAL_DATA=false -e FACELAB_DATA_DIR=/data facelab
```

The image sets `FACELAB_ENV=prod`, `FACELAB_HOST=0.0.0.0`, and `PORT=8010`.

## Project layout

```
facelab/
  config.py         FACELAB_* settings + detector/recognizer registry
  face.py           detector/recognizer engine, SAHI, analyze_frame, annotation
  matching.py       pure NumPy/SciPy matching, fusion, attendance metrics
  benchmark.py      calibration, stratified recall, latency percentiles
  datasets.py       persistent ground-truth dataset store
  cache.py          inference cache keys + JSON artifacts
server.py           FastAPI app: roster, scans, datasets, benchmarks
static/             Zero-build UI (HTML/CSS/JS)
scripts/
  fetch_models.py   Download model packs + UniFace weights, write a SHA-256 manifest
models/<pack>/      ONNX weights (git-ignored, fetched on demand)
.data/              Roster, frames, datasets, cache, results (git-ignored, dev data)
tests/              pytest suites for the matching and benchmark engines
Dockerfile          Container image (libGL for OpenCV, prod/ephemeral defaults)
```

## Tests

```bash
.venv/bin/python -m pytest tests -q
```

`tests/test_matching.py` covers similarity, assignment, margin gating, and burst
fusion; `tests/test_benchmark.py` covers galleries, calibration, stratified
recall, aggregation, and latency.

## Notes

- **Model licensing.** The InsightFace *library* is MIT, but the pretrained
  *models* are non-commercial; check the `license` field on `/api/registry`.
- FaceLab vendors the face pipeline instead of importing any backend, so projects
  evolve independently. Keep behavior in sync deliberately, not by import.
