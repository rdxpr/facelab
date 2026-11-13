# FaceLab

A standalone harness for **testing face-detection / recognition models** and
**storing each run's results** so the parent Attendyrc project can reuse them.

FaceLab runs the [InsightFace](https://github.com/deepinsight/insightface)
pipeline locally, lets you enroll reference photos under a chosen model pack,
scan classroom frames, and saves every scan as structured JSON. It has no
dependency on the Attendyrc backend — the face engine is vendored into
`facelab/`, so this repository clones, builds, and runs on its own.

```
Students (enroll per model)        Classroom scan (per model)
──────────────────────────         ──────────────────────────
reference photo → embedding   →    frames → faces → match against
                                    that model's templates → result JSON
```

## Highlights

- **Multiple models.** Every model pack in `models/<name>/` is selectable.
  Reference photos enroll under a model; scans only match templates for the
  selected model, so results from different models never mix.
- **Structured results.** Each scan is written under the data directory
  (`./.data/results/<scanId>.json` in dev) and served over the API, alongside
  the original and annotated frames.
- **Offline-friendly models.** Weights are too large for git (GitHub rejects
  files over 100 MB), so `scripts/fetch_models.py` downloads them on demand and
  the local copy under `models/` is reused.
- **Storage modes.** Local development persists the roster, photos, and results
  in `./.data`. When `FACELAB_ENV` is anything other than `dev`, uploads and
  results go to an ephemeral directory that is wiped on every start, so nothing
  about real people is retained on a server.

## Quick start (local)

Requires Python 3.11+ and, optionally, [uv](https://docs.astral.sh/uv/).

```bash
# 1. Install dependencies (uv is fastest)
uv sync                     # or: python -m venv .venv && .venv/bin/pip install -r requirements.txt

# 2. Make sure a model pack exists (skips if models/buffalo_l is already there)
.venv/bin/python scripts/fetch_models.py            # defaults to buffalo_l

# 3. Run
./run.sh                    # http://127.0.0.1:8010
```

`./run.sh` prefers the project's `.venv`, then `uv run`, then plain `python3`,
so it works no matter how you set the environment up.

### Using more than one model

Drop each pack into its own folder (the folder name is the model name the UI
shows):

```
models/
├── buffalo_l/      # det_10g.onnx, w600k_r50.onnx, ...
├── buffalo_s/
└── antelopev2/
```

Fetch known packs directly:

```bash
.venv/bin/python scripts/fetch_models.py --model buffalo_s --model buffalo_sc
.venv/bin/python scripts/fetch_models.py --list          # show known packs
```

The folder only needs the ONNX files FaceLab uses for detection and
recognition. By default `FACELAB_ALLOWED_MODULES=detection,recognition`, so
extra ONNX files in a pack are ignored.

## How it works

1. **Enroll.** Pick the active model in the top bar, then add a student with one
   or more single-face photos. Each photo is detected and turned into an
   embedding stored with the model name it came from.
2. **Scan.** Upload one or more classroom frames. Detection runs at the chosen
   detector size, faces are matched against the active model's templates with a
   greedy best-score assignment, and the busiest frame is annotated.
3. **Reuse.** The run is saved under the data directory (`./.data/results/...`
   in dev) and listed under *Saved runs*. Reload any run, open its JSON, or
   delete it.

Legacy rosters (from before models existed) are treated as belonging to the
default model pack.

### Result JSON shape

```jsonc
{
  "scanId": "20261113-170942-d719bd",
  "createdAt": "2026-11-13T11:39:42+00:00",
  "model": "buffalo_l",
  "summary": {
    "model": "buffalo_l", "threshold": 0.5, "detSize": 960,
    "frames": 2, "facesDetected": 33, "matchedCount": 3, "unknown": 4,
    "peopleCount": 12, "templateCount": 18, "bestIndex": 0, "elapsedMs": 8430
  },
  "people": [ { "personId": 1, "number": 1, "name": "…", "status": "Present",
                "confidence": 0.62, "templates": 2 } ],
  "frames": [ { "order": 0, "sourceName": "frame.jpg", "width": 1280,
                "height": 960, "imageUrl": "/media/scans/…/frame-0.jpg",
                "annotatedUrl": "/media/scans/…/frame-0-annotated.jpg",
                "faces": [ { "box": [x1, y1, x2, y2], "detScore": 0.9,
                             "matched": true, "personId": 1, "name": "…",
                             "confidence": 0.62 } ] } ],
  "warnings": []
}
```

## API

| Method | Path | Purpose |
| ------ | ---- | ------- |
| `GET`  | `/api/health` | Liveness + installed model names |
| `GET`  | `/api/state` | Roster, totals, per-model template usage, settings |
| `GET`  | `/api/models` | Installed packs with size, template/people counts |
| `GET`  | `/api/model?name=<pack>` | Load (or check) a model pack |
| `POST` | `/api/people` | Enroll a student (`name`, `model`, `images[]`) |
| `POST` | `/api/people/{id}/photos` | Add photos to a student (`model`, `images[]`) |
| `PATCH`| `/api/people/{id}` | Rename a student |
| `DELETE`| `/api/people/{id}` | Remove a student |
| `DELETE`| `/api/people/{id}/photos/{photoId}` | Remove one photo |
| `POST` | `/api/scan` | Scan frames (`model`, `threshold`, `detSize`, `images[]`) |
| `GET`  | `/api/results` | List saved runs (optionally `?model=<pack>`) |
| `GET`  | `/api/results/{scanId}` | Full result JSON |
| `DELETE`| `/api/results/{scanId}` | Delete a run and its frames |
| `POST` | `/api/reset` | Clear roster, scans, and results |

The interactive API docs are at `/docs` (FastAPI).

## Configuration

Every setting is optional and uses a `FACELAB_` prefix; see `.env.example`. The
most useful ones:

| Variable | Default | Notes |
| -------- | ------- | ----- |
| `FACELAB_MODELS_DIR` | `./models` | One subfolder per model pack |
| `FACELAB_ENV` | `dev` | Any non-`dev` value enables ephemeral storage |
| `FACELAB_EPHEMERAL_DATA` | auto | Force ephemeral (`true`) or persistent (`false`) |
| `FACELAB_EPHEMERAL_DIR` | system temp `/facelab` | Where ephemeral data lives |
| `FACELAB_DATA_DIR` | `./.data` | Persistent roster/frames/results (dev mode) |
| `FACELAB_DEFAULT_MODEL` | `buffalo_l` | Used when a request omits `model` |
| `FACELAB_ALLOWED_MODULES` | `detection,recognition` | Empty loads every model in a pack |
| `FACELAB_FACE_DET_SIZE` | `640` | Enrollment detector size |
| `FACELAB_FACE_SCAN_DET_SIZE` | `960` | Scan detector size |
| `FACELAB_FACE_SIMILARITY_THRESHOLD` | `0.5` | Cosine-match threshold |
| `FACELAB_RESULTS_TO_KEEP` | `50` | Saved run JSONs retained |
| `FACELAB_CORS_ORIGINS` | `*` | Comma-separated allowlist for the parent app |

## Running on a server

`FACELAB_ENV` decides whether uploads survive a restart:

| `FACELAB_ENV` | Data location | Behaviour |
| ------------- | ------------- | --------- |
| `dev` (default) | `./.data` | Roster, photos, frames, and results persist |
| anything else (e.g. `prod`) | `FACELAB_EPHEMERAL_DIR` (default `<system temp>/facelab`) | Directory is wiped on every start |

So a server started with `FACELAB_ENV=prod` keeps **no reference photos, scans,
or results** across restarts — only the model weights under `./models` remain.
Override the behaviour explicitly if you want something different:

```bash
FACELAB_ENV=prod FACELAB_EPHEMERAL_DATA=false ./run.sh   # persist despite prod
FACELAB_ENV=dev  FACELAB_EPHEMERAL_DATA=true  ./run.sh   # wipe despite dev
```

Set `FACELAB_HOST=0.0.0.0` (and optionally `PORT`) when running behind a reverse
proxy or in a container. `scripts/start.sh` does this for you.

> **Privacy.** There is no authentication: anyone who can reach the port can
> enroll faces, view results, or call `/api/reset`. Keep it on localhost or a
> private network, or put an auth proxy in front of it. The InsightFace
> pretrained models are licensed for non-commercial research use.

### Docker

The image sets `FACELAB_ENV=prod`, `FACELAB_HOST=0.0.0.0`, and `PORT=8010`, so
it is ephemeral by default.

```bash
docker build -t facelab .

# Ephemeral: photos and results vanish when the container restarts.
docker run --rm -p 8010:8010 facelab

# Persistent: opt out of ephemeral storage and mount a volume for the data.
docker run --rm -p 8010:8010 \
  -e FACELAB_EPHEMERAL_DATA=false -e FACELAB_DATA_DIR=/data \
  -v facelab-data:/data facelab

# Keep downloaded model weights between container restarts (~326 MB).
docker run --rm -p 8010:8010 -v facelab-models:/app/models facelab
```

## Project layout

```
facelab/            Vendored face engine + settings (no backend imports)
  config.py         FACELAB_* settings, paths
  face.py           detection, matching, annotation, per-model analyzer cache
server.py           FastAPI app: roster, scans, results, models
static/             Zero-build UI (HTML/CSS/JS)
scripts/
  fetch_models.py   Download model packs from the InsightFace model zoo
  start.sh          Container entrypoint (fetch model, then serve)
models/<pack>/      ONNX weights (git-ignored, fetched on demand)
.data/              Roster, frames, results (git-ignored, dev runtime data)
Dockerfile          Container image (libGL for OpenCV, prod/ephemeral defaults)
```

## Notes

- **Model licensing.** The InsightFace *library* is MIT, but the pretrained
  *models* are for non-commercial research use. Check their terms before any
  commercial deployment.
- FaceLab intentionally vendors the face pipeline instead of importing the
  Attendyrc backend, so the two projects evolve independently. If matching
  behavior must stay identical, keep `facelab/face.py` in sync with
  `backend/app/ml/face.py`.
