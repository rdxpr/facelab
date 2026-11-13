---
title: FaceLab
sdk: docker
app_port: 7860
---

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
- **Structured results.** Each scan is written to
  `.data/results/<scanId>.json` and served over the API, alongside the original
  and annotated frames.
- **Offline-friendly models.** Weights are too large for git (GitHub rejects
  files over 100 MB), so `scripts/fetch_models.py` downloads them on demand and
  the local copy under `models/` is reused.
- **Deployable.** Ships with a `Dockerfile`. The hosted image keeps everything
  in `/tmp` (ephemeral) so photos and results are never persisted, while local
  development keeps persistent storage.

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
3. **Reuse.** The run is saved to `.data/results/<scanId>.json` and listed under
   *Saved runs*. Reload any run, open its JSON, or delete it.

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
| `FACELAB_DATA_DIR` | `./.data` | Roster, frames, and results |
| `FACELAB_DEFAULT_MODEL` | `buffalo_l` | Used when a request omits `model` |
| `FACELAB_ALLOWED_MODULES` | `detection,recognition` | Empty loads every model in a pack |
| `FACELAB_FACE_DET_SIZE` | `640` | Enrollment detector size |
| `FACELAB_FACE_SCAN_DET_SIZE` | `960` | Scan detector size |
| `FACELAB_FACE_SIMILARITY_THRESHOLD` | `0.5` | Cosine-match threshold |
| `FACELAB_RESULTS_TO_KEEP` | `50` | Saved run JSONs retained |
| `FACELAB_CORS_ORIGINS` | `*` | Comma-separated allowlist for the parent app |

## Hosting on Hugging Face Spaces

This repo is ready to run as a **Docker** Space. Create one at
[huggingface.co/new-space](https://huggingface.co/new-space) with the **Docker**
SDK, then push this repo to the Space's git remote or upload the files — the
image builds automatically. Spaces reads its config from the YAML front matter
at the top of this README (`sdk: docker`, `app_port: 7860`).

### Hosted is ephemeral, local is persistent

The container image stores everything under `/tmp`:

- `FACELAB_DATA_DIR=/tmp/data` — roster, reference photos, frames, results
- `FACELAB_MODELS_DIR=/tmp/models` — downloaded ONNX packs

`/tmp` is wiped on every restart, so on a hosted Space **reference photos,
scans, and results are never persisted** and each cold start begins from a clean
slate. The model pack is re-downloaded from GitHub on the first boot (~1 minute).

Local development is unaffected: `./run.sh` uses the persistent `./models` and
`./.data` directories.

> **Privacy.** Public Spaces have no authentication: anyone with the URL can
> enroll faces, view results, or call `/api/reset`. Do not upload real people's
> faces to a public Space — make the Space private or add auth first. The
> InsightFace pretrained models are licensed for non-commercial research use.

The free CPU tier (2 vCPU, 16 GB RAM, 50 GB disk) runs `buffalo_l` at 960 px
comfortably. The server binds `0.0.0.0` and listens on `7860` (the image sets
`PORT=7860`).

### Run the container locally

```bash
docker build -t facelab .

# Ephemeral, like a Space:
docker run --rm -p 7860:7860 facelab

# Persistent: override the storage paths and mount volumes.
docker run --rm -p 7860:7860 \
  -e FACELAB_DATA_DIR=/data -e FACELAB_MODELS_DIR=/models \
  -v facelab-data:/data -v facelab-models:/models facelab
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
.data/              Roster, frames, results (git-ignored, local runtime data)
Dockerfile          Container image (libGL for OpenCV, ephemeral /tmp storage)
```

## Notes

- **Model licensing.** The InsightFace *library* is MIT, but the pretrained
  *models* are for non-commercial research use. Check their terms before any
  commercial deployment.
- FaceLab intentionally vendors the face pipeline instead of importing the
  Attendyrc backend, so the two projects evolve independently. If matching
  behavior must stay identical, keep `facelab/face.py` in sync with
  `backend/app/ml/face.py`.
