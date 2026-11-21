# FaceLab — Architecture

> A complete orientation for engineers/agents working on this repository.
> Last updated from the codebase at commit `96301d7` (2026-11-14).

FaceLab is a **local, standalone benchmarking and hyperparameter-tuning harness
for classroom face attendance**. It splits face analysis into an independent
**detector** and **recognizer** registry, caches raw inference per
`(image, detector, detector params, recognizer)`, and lets you re-score cached
embeddings instantly while thresholds/fusion settings change. Labelled datasets
drive automatic threshold calibration and size-stratified metrics.

It has **no backend dependency**: the whole pipeline is vendored under
`facelab/`, and `server.py` exposes it plus a zero-build vanilla-JS UI.

---

## 1. Repository map

```
facelab/                  Core engine (pure Python + ONNX; importable without the server)
  config.py               pydantic-settings Settings, model registries, paths, storage modes
  face.py                 Detectors, recognizers, SAHI tiling, alignment, embeddings, annotation
  matching.py             Pure NumPy/SciPy similarity, assignment, multi-frame fusion, metrics
  benchmark.py            Galleries, calibration curves, operating points, stratified recall, latency
  cache.py                Inference-cache keys, JSON artifacts, LRU pruning
  datasets.py             Persistent ground-truth dataset store
server.py                 FastAPI app: state, enrollment, scans, sweeps, datasets, benchmarks
static/                   Zero-build UI: index.html, app.js, styles.css
scripts/
  fetch_models.py         Download InsightFace packs + YuNet + UniFace weights; write SHA-256 manifest
  crop_faces.py           Standalone batch face-crop helper (padded crops for enrollment)
  start.sh                Container entrypoint: fetch model pack, then run server
tests/                    pytest suites (no ONNX required)
models/                   ONNX weights per pack (git-ignored; fetched on demand)
.data/                    Runtime state: roster, photos, scans, results, datasets, cache, crops (git-ignored)
run.sh                    Local launcher (prefers .venv)
Dockerfile                prod/ephemeral container image
.env.example              Every FACELAB_* setting
README.md, architecture.md
```

Ignored task/audit guides live at the repo root (`*audit.py`, `*guide.md`,
`IMPROVEMENT_GUIDE.md`, `scripts/incoming/`, `scripts/face-crops/`).

---

## 2. Runtime & deployment

### Local
- Python **3.11+**; dependencies pinned in `pyproject.toml` / `requirements.txt`.
- `uv sync` (or venv + pip), then `scripts/fetch_models.py`, then `./run.sh`.
- `run.sh` execs `.venv/bin/python server.py`, else `uv run python server.py`, else `python3`.
- Default bind: `127.0.0.1:8010` (or `FACELAB_HOST`/`HOST`, `FACELAB_PORT`/`PORT`;
  defaults to `0.0.0.0` when a port env var is set). CLI: `--host --port --log-level`.

### Docker
- `Dockerfile` uses `python:3.11-slim`, installs `libgl1`/`libglib2.0-0` (OpenCV),
  sets `FACELAB_ENV=prod`, `FACELAB_HOST=0.0.0.0`, `PORT=8010`, `HOME=/tmp`.
- `CMD ["sh", "/app/scripts/start.sh"]`. `start.sh` fetches the default pack
  (`FACELAB_FETCH_MODELS=1|all|0`) then starts the server.
- Healthcheck: `GET /api/health` (start period 120 s for first model fetch).
- `.dockerignore` excludes `.data`, `models`, `.venv`, caches, `.env`.

### Storage modes (`config.Settings`)
- `env="dev"` (default) ⇒ **persistent** `./.data`.
- Any other `env` ⇒ **ephemeral**: `data_dir` becomes `ephemeral_dir`
  (`<temp>/facelab`) and is **wiped at process start**.
- `ephemeral_data` (bool|None) overrides the env-derived behaviour.
- `prepare_dirs()` runs in `server.py` before mounts and `_load_state()` (not at
  `facelab.config` import). In ephemeral mode it refuses to wipe unsafe targets
  (filesystem root, home, temp root, repo root/parent) and `rmtree`s the data dir.
- Consequence: never keep user source photos in `.data` if the server may run in
  non-dev mode (the crop helper uses `scripts/incoming/` instead).

---

## 3. Configuration

`facelab/config.py` defines module-level registries and a `Settings` class
(`pydantic-settings`, prefix `FACELAB_`, `.env` aware, `extra="ignore"`).

### 3.1 Settings fields (defaults)

| Field | Default | Meaning |
|---|---|---|
| `app_name` | `FaceLab` | Display name |
| `env` | `dev` | Storage mode switch |
| `models_dir` | `<repo>/models` | Resolved to absolute |
| `data_dir` | `<repo>/.data` | Resolved to absolute |
| `ephemeral_data` | `None` | Force ephemeral on/off |
| `ephemeral_dir` | `<temp>/facelab` | Ephemeral data root |
| `default_model` | `buffalo_l` | Legacy pack name |
| `default_detector` | `insightface:scrfd_10g_bnkps` | Default detector id |
| `default_recognizer` | `insightface:w600k_r50` | Default recognizer id |
| `allowed_modules` | `detection,recognition` | Legacy `FaceAnalysis` modules |
| `onnx_providers` | `""` | Explicit provider order; auto if empty |
| `face_det_size` | `640` | Enrollment detector size |
| `face_scan_det_size` | `1280` | Scan/benchmark detector size |
| `face_scan_max_frames` | `12` | Upload cap per scan/dataset |
| `face_similarity_threshold` | `0.42` | Legacy single threshold |
| `accept_threshold` | `0.42` | Present/accept band |
| `review_threshold` | `0.34` | Review band |
| `min_margin` | `0.04` | Top1−top2 gate |
| `assignment_mode` | `hungarian` | `hungarian`/`greedy`/`margin` |
| `template_strategy` | `top_k_mean` | `top_k_mean`/`max`/`mean`/`prototype` |
| `template_top_k` | `2` | k for top-k mean |
| `tiling_mode` | `none` | `none`/`sahi_upper_band`/`sahi_full` |
| `multi_frame_fusion` | `k_of_n_voting` | Fusion mode |
| `fusion_k` | `2` | k-of-n votes |
| `min_face_quality` | `0.0` | Quality floor |
| `quality_predictor` | `uniface:ediffiqa_s` | Optional UniFace quality model |
| `head_pose_predictor` | `uniface:headpose_6d` | Optional UniFace pose model |
| `benchmark_det_sizes` | `640,1024,1280` | Grid sizes |
| `benchmark_max_rows` | `32` | Grid cap (combos) |
| `benchmark_max_frames` | `60` | Frames per dataset in a run |
| `benchmarks_to_keep` | `10` | Saved benchmark runs (+preview dirs) |
| `benchmark_preview_max_width` | `1600` | Downscale for preview JPEG |
| `max_photos_per_upload` | `8` | Enrollment upload cap |
| `max_photos_per_person` | `12` | Photos per student |
| `results_to_keep` | `50` | Saved result JSONs |
| `scans_to_keep` | `50` | `scans/<id>/` frame folders |
| `datasets_to_keep` | `25` | Dataset folders |
| `cache_entries_to_keep` | `2000` | Inference-cache JSON artifacts |
| `cors_origins` | localhost:8010 | Comma-separated allowlist |
| `api_token` | `""` | Bearer/X-API-Token guard when set |

Derived properties: `ephemeral`, `allowed_module_list`, `cors_origin_list`,
`benchmark_det_size_list` (filters 320–2048), `provider_list` (configured,
else CUDA→CoreML→CPU filtered by `ort.get_available_providers()`), and path
properties (`faces_dir`, `scans_dir`, `results_dir`, `crops_dir`, `cache_dir`,
`datasets_dir`, `benchmarks_dir`, `students_file`, `legacy_people_file`,
`models_manifest_file`). `ensure_dirs()` creates the data tree;
`prepare_dirs()` additionally handles the ephemeral wipe.

### 3.2 Model registries

- `KNOWN_DETECTORS: dict[id, entry]`
  - Common fields: `id`, `label`, `library` (`insightface`|`uniface`|`opencv`),
    `kind="detector"`, `landmarks_5pt`, `license`, `role`, optional `wider_hard`,
    optional `input_sizes` (e.g. YOLOv8-Face `[640]`).
  - Weight location: `file_candidates` (searched in `models/` and
    `models/<pack>/`) or `uniface_weights` (UniFace cache).
  - Entries: SCRFD-10G/2.5G/34G, SCRFD-10G-WO (Raccoon-L), UniFace RetinaFace R50
    and YOLOv8n-Face, OpenCV YuNet.
- `KNOWN_RECOGNIZERS: dict[id, entry]`
  - `library`, `embedding_dim`, `file_candidates`/`uniface_weights`, `license`,
    `role`, optional `pack`.
  - Entries: ArcFace R50 (`w600k_r50`), Glint360K R100 (`glintr100`),
    MobileFaceNet (`w600k_mbf`), AdaFace IR-101, EdgeFace-S γ0.5, SphereFace-20,
    custom FaceLiVTv2-S ONNX.
- `KNOWN_PACKS: dict[name, {required, license}]` — used by `fetch_models.py` and
  the packs endpoint. `required` lists only the detector+recognizer FaceLab needs.
- Helpers: `detector_registry()`, `recognizer_registry()` (copies),
  `default_detector_for_pack`, `default_recognizer_for_pack`,
  `pack_for_detector/recognizer`.

> Adding a detector/recognizer = add a registry entry + ensure its weights are
> resolvable + optionally a `KNOWN_PACKS` entry and `fetch_models.py` support.
> The UI and API pick it up automatically from `/api/registry`.

---

## 4. Domain model

### 4.1 People / students
Internally `server._people: dict[int, person]` where `person` is:

```python
{
  "id": int, "number": int, "name": str, "createdAt": iso,
  "photos": [
    {
      "id": str, "filename": "faces/p<number>/<id>.jpg", "sha256": str,
      "quality_score": float, "quality_note": str, "head_pose": dict|None,
      "createdAt": iso,
      "embeddings": {
        "<recognizer_id>": {
          "raw": np.ndarray(512,), "flip_avg": np.ndarray|None,
          "lowres_aug": np.ndarray|None, "norm_magnitude": float
        }
      }
    }
  ]
}
```

Persisted as `.data/students.json`, version 3:

```json
{ "version": 3, "nextNumber": 1,
  "students": [{ "id": 1, "number": 1, "name": "...", "createdAt": "...",
    "photos": [{ "id": "...", "filename": "faces/p1/....jpg", "sha256": "",
      "qualityScore": 0.8, "qualityNote": "...", "headPose": null,
      "createdAt": "...",
      "embeddings": { "insightface:w600k_r50": {
        "raw": [...], "flip_avg": [...], "lowres_aug": [...],
        "lowres_aug_version": 2, "norm_magnitude": 12.3 } } }] }] }
```

`lowres_aug_version` marks templates built with the current flip-averaged
degraded definition; older clean/degraded blends are dropped on load and
regenerated by re-embedding.

Legacy migration: `.data/people.json` (v1/v2, base64 float32 embeddings keyed by
model pack) → mapped to `default_recognizer_for_pack(pack)` and re-saved.
`_load_state()` runs at import; on any error it renames the file to
`.broken-<epoch>.json` and starts empty.

`_public_person` shape: `{id, number, name, createdAt, templates, recognizers:
{rid: count}, photos: [{id, url, quality, qualityScore, headPose, createdAt,
recognizers: [rid...]}]}`.
`_public_state`: `{people, students (alias), nextNumber, totals: {people,
templates}, modelUsage, settings, registry}`.

### 4.2 Gallery / templates
`benchmark.build_gallery(students, recognizer_id, variant, ensemble=False)`
collects L2-normalized vectors per student from `photo.embeddings[recognizer][variant]`
(falling back to `raw`; `ensemble=True` stacks all variants). Result:
`dict[student_id, np.ndarray(n_templates, dim)]`.

### 4.3 Detections
`face.Detection` = `{bbox: (x1,y1,x2,y2) float32, score: float, kps: (5,2)|None}`
with `.height`. `analyze_frame` expands each detection to a cache payload:

```python
{det_index, bbox:[x1,y1,x2,y2], face_height_px, size_bucket,
 det_score, landmarks_5pt, landmarks_crop, aligned_crop_path,
 embedding, embedding_flip_avg, embedding_lowres_aug, embedding_norm_mag,
 quality_score, quality_source, head_pose}
```

### 4.4 Decisions
`matching.assign_single_frame` returns per-face:

```python
{det_index, matched_student_id, status: "present"|"needs_review"|"unmatched",
 score, margin, top_candidates: [{student_id, score}, ... up to 3]}
```

`fused["students"][sid]` (fusion output): `{status: "present"|"needs_review"|
"absent", score, frames_hit, quality}`. `fused["per_frame"]` is the per-frame
decision list (cluster mode rewrites status/id/score/margin/top_candidates).

### 4.5 Scans / results
Saved to `.data/results/<scanId>.json` and frames to `.data/scans/<scanId>/`.
See §9.4 for the JSON shape.

### 4.6 Datasets
`.data/datasets/<datasetId>/frames/*` + `manifest.json`:

```json
{ "dataset_id": "...", "datasetId": "...", "name": "...", "notes": "...",
  "createdAt": "...",
  "frames": [{ "frame_id": "frame_01", "filename": "frames/frame_01.jpg",
               "original_name": "...", "sha256": "...", "width": 1280, "height": 960 }],
  "ground_truth": {
    "present_student_ids": [1, 2],
    "expected_total_faces": 10,
    "annotated_boxes": [{ "frame_id": "frame_01", "bbox": [x1,y1,x2,y2],
      "student_id": 1|null, "occlusion": "none", "row_bucket": "large|medium|small" }]
  } }
```

`list_datasets()` returns `{dataset_id, datasetId, name, createdAt, frames,
present, annotatedBoxes, thumbUrl}`. Dataset ids must match
`^[A-Za-z0-9_-]{1,80}$` (path-traversal guard). Failed creation cleans up.

### 4.7 Benchmarks
`.data/benchmarks/<runId>.json` (rows, curves, frames metadata) plus
`.data/benchmarks/<runId>/<NN-slug>.jpg` (one preview per combo). See §9.6.

---

## 5. Face engine (`facelab/face.py`)

### 5.1 Locks and caches
- `_ENGINE_LOCK = threading.RLock()` guards all model caches.
- `_ANALYZERS` legacy `FaceAnalysis` packs (one per pack name).
- `_DETECTORS` = **`OrderedDict`** keyed `(detector_id, (w,h), round(det_thresh,4),
  round(nms_thresh,4))`; **max 6** (`_MAX_DETECTORS`), LRU eviction +
  `move_to_end` on hit.
- `_RECOGNIZERS` keyed `(recognizer_id, tuple(provider_list))`.
- `_QUALITY` / `_POSE` caches; missing weights return `None` without negative
  caching, so weights fetched later take effect without a restart.
- ONNX-backed model construction uses `_with_cpu_fallback`: when accelerator
  session initialization fails (including GPU-memory exhaustion) it first calls
  `_evict_idle_models`, which drops every cached session except the one being
  built and then `gc.collect()`s so ONNX Runtime releases the idle sessions'
  CUDA arenas; the accelerator initialization is then retried. Only if the new
  model still does not fit is it rebuilt with CPU-only providers. A providers
  list with no CPU entry is never silently downgraded.
- `unload_other_models(detector_id, recognizer_id)` applies the same eviction at
  model-selection time (`GET /api/model`), so choosing a model keeps only the
  active pair warm on the GPU.

### 5.2 Availability
- File-backed: `resolve_model_file(entry)` searches `models/<name>` then
  `models/<pack>/<name>` for each `file_candidates`.
- UniFace: requires the package **and** a cached weight file under
  `~/.uniface/models/<value>.*` (via `uniface.model_store.get_cache_dir`).
- `registry_payload()` returns `{detectors: [...entry + available + reason],
  recognizers: [...], providers}`.

### 5.3 Detectors
`BaseDetector.detect(image_bgr) -> list[Detection]`; `warm_up()` runs a blank frame.
- `InsightFaceDetector`: loads `ort.InferenceSession` + `SCRFD`, calls
  `prepare(ctx_id=0, input_size, det_thresh, nms_thresh)`, pins thresholds on the
  model, requires `use_kps`. `detect()` calls `self.model.detect(image, input_size=...)`
  (**no** `det_thresh` kwarg; the threshold lives on `self.model.det_thresh`).
  When the primary pass finds nothing or a face taller than 50% of the frame
  (a close-up overflows SCRFD's anchor scales), it re-runs at 128 px and merges
  with `merge_detections`, which keeps primary geometry unless the small pass
  recovers a much larger box that contains a clipped fragment.
- `UnifaceDetector`: picks a class by weight-name hints (RetinaFace/YOLOv8/SCRFD/
  CenterFace), passes `confidence_threshold`/`nms_threshold`/`providers` and an
  `input_size` chosen by inspecting the class's default type (`int` vs tuple).
  Some models are fixed-size (YOLOv8 = 640) and raise otherwise; the registry's
  `input_sizes` prevents that.
- `OpenCVYuNetDetector`: `cv2.FaceDetectorYN_create`; `detect()` fits the frame
  into `det_size` (long side) with `INTER_AREA`, calls `setInputSize` with the
  scaled dimensions, then maps boxes/keypoints back to source coordinates. This
  makes the requested `detSize` (and every benchmark grid row) effective.

### 5.4 Recognizers
`BaseRecognizer.embed(crops) -> np.ndarray(n, dim)`.
- `InsightFaceRecognizer`: `ArcFaceONNX.get_feat(list(crops))`.
- `UnifaceRecognizer`: per-crop `model.get_embedding(crop, None)` stacked.
- Batch note: ArcFace ONNX declares a static output batch `{1,512}`; feeding N
  crops triggers a benign onnxruntime "actual shape {N,512}" warning — results
  are correct.

### 5.5 Imaging helpers
- `image_to_bgr`: accepts ndarray (RGB→BGR), file-like, bytes, PIL image;
  **EXIF-transposes** through PIL (`ImageOps.exif_transpose`) for bytes/file-like.
- `align_face(image, kps, size=112)`: `face_align.estimate_norm` + `warpAffine`.
  When the warp reduces the source by more than 2× (`scale < 0.5`, i.e. faces
  taller than ~224 px), it Gaussian-blurs the face ROI with
  `sigma = 0.5*(1/scale - 1)` before the warp; plain bilinear sampling aliases
  high-frequency detail into the 112 crop. The ROI transform is numerically
  identical to blurring the whole frame.
- `_project_landmarks_to_crop`: affine-maps 5-pt landmarks into the 112 crop for
  the inspector overlay.
- `lowres_augment(crop, tiny=24, size=112)`: downsample→bicubic upsample.
- `embed_variants(recognizer, crop, flip_aug, lowres_aug)`: the single definition
  of `raw` (clean), `flip_avg` (clean mirror average) and `lowres_aug`
  (flip-averaged *degraded* crop). Enrolment uses all three; classroom probes
  never compute the degraded variant because `select_embedding` scores the clean
  vector.

### 5.6 SAHI tiling
- `generate_tiles(h, w, tile_size=640, overlap=0.25, upper_band_only)`: stride =
  `tile_size*(1-overlap)`; upper band covers the top 70% of the image; returns
  clamped, de-duplicated `(x0,y0,x1,y1)` boxes.
- `detect_with_sahi(image, detect_fn, tile_size, overlap, iou_threshold,
  tiling_mode)`: `none` → whole frame; otherwise detect on whole frame + every
  tile (offset boxes/landmarks) and merge with `_merge_tiled_detections`.
- `_merge_tiled_detections(boxes, scores, landmarks, truncated, iou_threshold)`:
  ranks detections clipped by an *internal* tile border below complete ones, and
  drops a clipped box when ≥90% of it lies inside a complete detection (heavy
  clipping can push IoU below the NMS threshold). Score order is preserved among
  detections of equal completeness.
- `nms_boxes_and_landmarks(...)` remains available for plain NMS.

### 5.7 Enrolment (`embed_reference`)
Detects with `detect_with_sahi(tiling_mode="none")`, keeps detections with
keypoints, selects the **dominant face** (`select_dominant_face`: largest bbox,
ties by score), aligns one 112×112 crop, then embeds it with **every** requested
recognizer via `embed_variants`. Per recognizer it stores `raw`, optional
`flip_avg` (clean mirror average), optional `lowres_aug` (flip-averaged
**degraded** template for matching low-resolution classroom probes), and
`norm_magnitude`. Quality uses the UniFace predictor when available; otherwise
`quality_score` is `None` / `quality_source="unavailable"` (there is no
recognizer-independent norm fallback). Pose is optional. Returns
`{face_height_px, quality_score, quality_source, head_pose, embeddings}`.

> The padded crop helper (`scripts/crop_faces.py`) exists because this function
> needs a natural photo with margin; the experimental tight aligned crop is not
> always sufficient. The dominant-face rule means a portrait with a stray
> background face still enrols the subject.

### 5.8 Full-frame analysis (`analyze_frame`)
1. Resolve detector (`get_detector`) and recognizer (`get_recognizer`).
2. Detect (SAHI per `tiling_mode`), then for each detection compute bbox, height,
   size bucket, and (if keypoints) align a 112×112 crop + project landmark overlay;
   optionally write `<crops_dir>/<prefix>_det<i>.jpg`.
3. Embed all crops at once, plus the mirror pass when flip is enabled. The
   degraded pass is **not** run: `lowres_aug` is gallery-side only. Embeddings
   are indexed by **crop index** (`crop_index`), not detection index, so a
   detection without keypoints cannot shift vectors.
4. Quality/pose per payload when enabled. Without a quality model,
   `quality_score` stays `None` and fusion treats it as unknown (not 0), so a
   positive `min_quality` does not reject every face.
5. Return `{detector_id, recognizer_id, det_params, timings_ms, detections}` with
   per-detection embeddings as raw (un-normalized) arrays.

### 5.9 Warm-up & annotation
- `warm_up(detector_id, recognizer_id, det_size, tiling_mode, det_thresh,
  nms_thresh)` runs one throwaway `analyze_frame` (blank 480×640 image), then
  explicitly warms the recognizer because the blank frame has no face crop, and
  returns `{warmup_ms}`. `GET /api/model` evicts every other cached model and
  runs this under `_model_lock` before reporting the pair ready. Initialization/
  inference failures are surfaced as `FaceEngineUnavailable` rather than making
  an unusable model look ready.
- `annotate_classroom_photo(image_bgr, matches, label_unknown=False)`: draws a box
  per match (green present / amber review / red unknown), with a filled label tag.
  Present → `Name (score)`, review → `? Name (score)`, unknown → `"Unknown"` only
  when `label_unknown`, otherwise **box only**. Returns RGB.

### 5.10 Legacy pack API
`available_models()`, `model_exists()`, `get_face_analyzer()` remain for pack
listing and the legacy `/api/model?name=` path (`_ANALYZERS`). They are not used
by scans.

---

## 6. Matching engine (`facelab/matching.py`)

Pure NumPy/SciPy; no I/O or ONNX, so every rule is unit-testable.

### 6.1 Similarity
- `size_bucket(px)`: large ≥55, medium ≥30, else small.
- `l2_normalize`: zero-safe row normalization.
- `flip_averaged(a, b)`: `normalize(normalize(a)+normalize(b))`.
- `compute_student_similarity(face, templates, strategy, top_k)`: `max`, `mean`,
  `top_k_mean` (mean of best k), `prototype` (similarity to normalized mean).
  Unknown strategy → `ValueError`.
- `build_similarity_matrix(faces, gallery, strategy, top_k)` →
  `(matrix[n_faces, n_students], student_ids)`.

### 6.2 Single-frame assignment (`assign_single_frame`)
Defaults: accept `0.42`, review `0.34`, margin `0.04`, `hungarian`. Always
returns one decision per detected face; an empty gallery yields all
`unmatched` decisions instead of an empty list (callers used to drop the faces).
- `hungarian`: mask `< review` to `-1`, add a bonus of 2 for pairs `≥ accept`,
  then `linear_sum_assignment(-weights)`. Maximising the number of accepted
  matches before raw similarity prevents losing one present match in exchange
  for two review-band matches.
- `greedy`: iterate edges by the same weighted score, one face/student each.
- `margin`: each face independently takes its top-1 if ≥ review.
- Per face: top-3 candidates, `margin = best - second`, assigned score; status is
  `present` if assigned ≥ accept **and** (`margin ≥ min_margin` when it is the top
  choice), else `needs_review`; otherwise `unmatched`.

### 6.3 Multi-frame fusion (`fuse_multi_frame`)
Builds per-frame decisions + similarity matrices, then per mode:
- `k_of_n_voting`: counts frames where the assigned decision clears
  `review_threshold` **and** `quality ≥ min_quality`, plus `present` hits.
  `required = max(1, min(k, num_frames))`; present if
  `hits ≥ required and present_hits ≥ 1`; else review if `hits ≥ 1`.
  (The old single-frame fallback was removed.)
- `max_with_margin_gate`: per detection, skip below `min_quality`; a detection
  that is not `present` or fails the margin gate is remembered as `review_sids`;
  otherwise its gallery similarity updates the student's score/quality/frames.
  Final: present if `frames_hit > 0` and score ≥ accept, else review if
  `frames_hit or sid in review_sids`.
- `quality_weighted_embedding_cluster`: `_cluster_tracks` greedily groups
  detections passing `min_quality` across frames by cosine ≥ 0.55 **and**
  normalized-box IoU ≥ 0.3, at most one face per frame per track; track centroid
  uses an unnormalized quality-weighted sum (unmeasured `None` quality weighted
  at `1.0`) and a running-mean box. Tracks are matched once; their
  status/id/score/margin/top_candidates are copied back onto every member
  detection for UI/annotation consistency.

### 6.4 Metrics
`attendance_metrics(predicted_ids, truth_ids)` → `{tp, fp, fn, precision,
recall, f1}` (rounded 4dp).

---

## 7. Benchmark engine (`facelab/benchmark.py`)

### 7.1 Galleries & embedding selection
- `build_gallery` (§4.2).
- `select_embedding(detection, variant)`: `raw`→`embedding`, `flip_avg`→
  `embedding_flip_avg`, `lowres_aug`→prefers `embedding_flip_avg` then
  `embedding` (classroom probes are already low-res; avoid degrading twice),
  falling back to raw; returns an L2-normalized vector or zeros. The
  `lowres_aug` *gallery* templates come from enrolment's degraded variant, so
  the same name no longer means a blend on one side and a pure probe on the
  other.
- `_fusion_frames(frames, variant)` adapts cached frames to the fusion input
  `{frame_id, width, height, detections: [{embedding, bbox, quality}]}`.
  `quality` is `None` when it was not measured; fusion treats that as unknown
  rather than as a failing score.

### 7.2 Stratified detection/identification recall
`stratified_metrics(frames, decisions, annotated_boxes, iou_threshold=0.5)`:
- Counts **detected** faces into `large/medium/small` buckets from
  `size_bucket` or bbox height, for every frame (before the no-GT early continue).
- With annotations, matches GT boxes one-to-one with
  `_match_gt_to_detections`: maximise the number of pairs ≥ IoU threshold first,
  then total IoU. The old greedy pass made recall depend on annotation order.
  Counts `gt/detected/identified` per GT row bucket; identification requires
  `status=="present"` and `str(matched_student_id)==str(gt.student_id)`.
- Returns overall recall + `buckets` + `has_annotations` + `detected_buckets`.
  Without boxes, `gt=0` and all recall values are 0; the UI hides these.

### 7.3 Evaluation (`evaluate_frames`)
Builds fusion frames, calls `fuse_multi_frame`, computes attendance metrics,
`present_sims` (fused scores of present students) and `max_impostor` (highest
score/top-candidate score on impostor detections, including false-positive
present matches to absent students while excluding missed genuine students),
stratified metrics, and faces detected. Returns `attendance`,
`present_sim_sum/n`, `mean_present_sim`, `max_impostor_sim`,
`predicted_student_ids`, `students`, `detection`, `per_frame`, `faces_detected`.

### 7.4 Threshold curves
- `threshold_curve(frames, gallery, present, **kwargs)` delegates to
  `threshold_curve_datasets([(frames, present)], ...)`.
- `threshold_curve_datasets(datasets, gallery, ...)` sweeps accept thresholds
  from **0.25** to 0.75 (step 0.01) by default. Per step:
  `review = min(review_threshold, threshold)` when a fixed review threshold is
  supplied, else `threshold * review_ratio`; passes `min_quality`; evaluates each
  dataset separately (avoids frame-id collisions) and aggregates attendance.
  Each row also carries the similarity stats.

### 7.5 Operating points (`pick_operating_points`)
- `best_f1`: max by `(f1, precision, threshold)` — highest, most conservative
  threshold on the F1 plateau.
- `far0`: among `fp==0 and tp>0`, max by `(recall, threshold)`.
- `prec98`: among `precision≥0.98 and tp>0`, max by `(recall, threshold)`.
- Empty curve → all `None`. This prevents the legacy "F1=1.0 @ τ=0.15" floor.

### 7.6 Aggregation & latency
- `aggregate_attendance(results)`: sums tp/fp/fn (each result has
  `attendance`), recomputes precision/recall/F1, adds weighted
  `mean_present_sim` and max `max_impostor_sim`.
- `aggregate_detection(results)`: merges `buckets` and `detected_buckets`.
- `latency_percentiles(timings)`: per-stage p50/p95 for any stage > 0 (`decode`,
  `detect`, `align_embed`, `quality_pose`, `match`, `annotate`); total =
  detect+align_embed+quality_pose+match; adds `total_inference_ms_mean` and
  `frames_per_sec` (1000/mean); `passes = len(timings)`.
- `load_cached_frames(dataset_id, detector_id, recognizer_id, det_params,
  flip_aug)`: loads cached artifacts for a dataset; returns `(frames, missing)`.

---

## 8. Inference cache (`facelab/cache.py`)

- `sha256_bytes(data)`.
- `analysis_params(det_size, det_thresh=0.35, nms_thresh=0.45, tiling_mode="none",
  tile_size=640, tile_overlap=0.25, lowres_aug=False, with_quality=True,
  with_pose=True)` — **callers must build this identically** or keys won't match.
- `_param_fingerprint` / `cache_key`: hashes
  `{v, image_sha256, detector_id, det_size, det_thresh, nms_thresh, tiling_mode,
  tile_size, tile_overlap, with_quality, with_pose, recognizer_id, flip_aug}`
  (4dp thresholds, 3dp overlap). Key =
  `<sha12>_<det_slug>_<rec_slug>_<digest16>`. `CACHE_VERSION = 2` (antialiased
  alignment, close-up recovery, clean probes); `lowres_aug` is excluded because
  it no longer changes the per-frame artifact.
- `save_analysis(payload)` writes JSON atomically (`*.json.tmp` → replace);
  numpy arrays encoded to lists; `crop` dropped (crop lives on disk).
- `load_analysis(key)` decodes and **touches the file mtime on a hit** so
  `prune_cache` evicts least-recently-used entries.
- `prune_cache(keep=None)` sorts `*.json` by mtime and removes the oldest beyond
  `cache_entries_to_keep`, including `<key>_det*.jpg` crops.
- Detection decode converts `bbox`, `embedding`, `embedding_flip_avg`,
  `embedding_lowres_aug` to float32 arrays (`kps` is not stored).

---

## 9. HTTP server (`server.py`)

### 9.1 Startup
- `settings.prepare_dirs()` runs in `server.py` before `StaticFiles` mounts.
- `_load_state()` runs at module import (bottom of `server.py`).
- Locks:
  - `_state_lock` (RLock): guards `_people`/`_next_number` and `_save_state`.
  - `_model_lock` (Lock): serializes heavy inference (enrollment, scans,
    benchmarks). Held for the whole scan/benchmark job.
  - `_ENGINE_LOCK` (in `face.py`): model construction/caches.
- `FastAPI(title="FaceLab", version="0.3.0")`; CORS per `cors_origins`.
- Mounts: `/media/faces`, `/media/scans`, `/media/crops`, `/media/datasets`,
  `/media/benchmarks`, `/static`. `GET /` serves `static/index.html`.

### 9.2 Auth
`require_token` compares `X-API-Token` or `Authorization: Bearer` to
`settings.api_token`; no-op when the token is empty. Applied via
`Depends(require_token)` on **all mutating routes** (people/photos/rename/delete,
re-embed, scan, result delete, result sweep, dataset create/delete, benchmark
run/sweep, reset) plus `GET /api/model`. Read routes are open.

### 9.3 Endpoint catalogue

| Method | Path | Auth | Notes |
|---|---|---|---|
| GET | `/` | – | UI |
| GET | `/api/health` | – | status, defaults, providers, installed pack names |
| GET | `/api/state` | – | roster, totals, settings, registry |
| GET | `/api/registry` | – | availability + license table |
| GET | `/api/models` | – | packs + registry + providers |
| GET | `/api/model?name=&detector=&recognizer=` | ✔ | warm/check a pair or legacy pack |
| GET | `/api/cache` | – | cache entry/byte/crop counts |
| POST | `/api/people` | ✔ | enroll (`images[]`, `name`, `recognizer`, `recognizers[]`) |
| POST | `/api/people/{id}/photos` | ✔ | add photos |
| POST | `/api/students/re-embed` | ✔ | `{recognizer_ids: []}` backfill |
| PATCH | `/api/people/{id}` | ✔ | `{name}` |
| DELETE | `/api/people/{id}/photos/{photoId}` | ✔ | remove photo |
| DELETE | `/api/people/{id}` | ✔ | remove student + `faces/p<number>/` |
| POST | `/api/scan` | ✔ | multipart frames + all scan params |
| GET | `/api/results` | – | summaries (optional `?model=`) |
| GET | `/api/results/{scanId}` | – | full result JSON |
| DELETE | `/api/results/{scanId}` | ✔ | JSON + scan frames |
| POST | `/api/results/{scanId}/sweep` | ✔ | re-score from cached frames |
| GET | `/api/datasets` | – | list |
| POST | `/api/datasets` | ✔ | multipart frames + `presentIds`/`boxes` JSON |
| GET | `/api/datasets/{id}` | – | manifest |
| DELETE | `/api/datasets/{id}` | ✔ | delete folder |
| POST | `/api/benchmark/run` | ✔ | run a grid (`BenchmarkRunRequest`) |
| POST | `/api/benchmark/sweep` | ✔ | re-score datasets from cache |
| GET | `/api/benchmarks` | – | run summaries |
| GET | `/api/benchmarks/{runId}` | – | full run |
| POST | `/api/reset` | ✔ | clear roster + all artifacts |

`MAX_IMAGE_BYTES = 12 MB` per uploaded image.

### 9.4 Scan flow (`POST /api/scan` → `_run_scan_sync`)
1. Validate frame count (≤ `face_scan_max_frames`), detector/recognizer, size
   320–2048 **and** `input_sizes` if declared; threshold range 0–1.
2. `_run_scan_sync`: build `det_params` via `_scan_det_params`, variant via
   `_variant_for` (`lowres_aug` → `lowres_aug`, else `flip_avg`/`raw`),
   warm up with the same det/nms thresholds.
3. Under `_model_lock`, decode each frame (`_decode`, warns per bad frame) and
   run `_analyze_or_cache` (cache key includes recognizer/flip).
4. Build fusion frames, `matching.fuse_multi_frame`, then per frame serialize
   faces (`_pad_decisions` guarantees one decision per detection, including an
   empty gallery; `_serialize_face`), write `frame-N.jpg` and `frame-N-annotated.jpg`
   (via `annotate_classroom_photo(..., label_unknown=...)`).
5. Build the full roster from the student snapshot (all students, with
   `status/confidence/framesHit/enrolled/photoUrl/templates`), compute counts,
   best frame, timings.
6. `_prune_scans()`, `cache_store.prune_cache()`, `_save_result()`.

Result JSON: `{scanId, createdAt, model (=recognizer), detectorId, recognizerId,
people[], frames[], summary, warnings[]}`. `summary` includes detector/recognizer,
det size/thresh/nms, tiling, accept/review/margin, assignment/template/topK,
minQuality, fusion mode/k, variant, flip/lowres, labelUnknown, detParams, frame
counts, facesDetected, unknown (max unmatched **per frame**), matchedCount,
needsReview, peopleCount, templateCount, bestIndex, cachedFrames, elapsedMs, and
a `timings` block (`decode/detect/align_embed/quality_pose/match/annotate/total`).
Each frame: `{order, sourceIndex, sourceName, width, height, imageUrl,
annotatedUrl, cacheKey, cached, timings, faces[]}`. Each face (from
`_serialize_face`): `{box, detScore, status, matched, personId, number, name,
confidence, margin, sizeBucket, faceHeightPx, quality, qualitySource, headPose,
cropUrl, landmarksCrop, landmarks5pt, topCandidates[]}`.

### 9.5 Live sweep (`POST /api/results/{scanId}/sweep`)
Re-scores a saved scan from its cached artifacts (no inference). Validates
thresholds, loads the result + gallery, loads each frame's cache by `cacheKey`
(409 if any missing), runs `fuse_multi_frame` with the request params, and
returns `{students, faces, matchedCount, needsReview, unknown (max per frame),
params}`. `params` is the request dump. The frontend merges this into
`state.scan` and re-renders.

### 9.6 Benchmark flow (`POST /api/benchmark/run` → `_run_benchmark_sync`)
- `_resolve_benchmark_config`: intersect requested detectors/recognizers with
  available ones, validate sizes, build the combo cross-product
  **detector → recognizer → size → tiling**, skipping sizes not in a detector's
  `input_sizes`, then cap at `benchmark_max_rows`.
- `run_id` is generated **before** the combo loop (preview paths need it).
- Per combo: warm up (any exception → error row), build gallery, build
  `det_params`, load/analyze dataset frames under `_model_lock` (capturing
  `cached`, decoded `img_w/img_h`, dataset id/filename), measure final evaluation
  (`match_ms`, copied into per-frame timings), aggregate attendance/detection,
  build `frames_meta` (per-frame face metadata for the viewer; `landmarks5pt`
  dropped to keep JSON small), render **one** downscaled annotated preview JPEG
  for frame 0, and record timings/FPS/margin/recipe.
- Run JSON: `{runId, createdAt, config, rows[]}`; `config` echoes datasets,
  grid, det/nms thresholds, variant, aug/quality/pose, `forceFresh`,
  `labelUnknown`, providers.
- Successful row keys: detector, recognizer, detSize, tilingMode, variant,
  providers, detector/recognizer SHA-256, acceptThreshold, reviewThreshold,
  minMargin, assignmentMode, templateStrategy, fusionMode, `recipe`, metrics
  (+similarity stats), meanPresentSim, maxImpostorSim, simMargin, tauF1, bestF1,
  far0, prec98, detection, detectedBuckets, facesDetected, expectedFaces,
  latency, fps, matchMs, wallMs, cachedFrames, frameCount, annotatedUrl, frames[],
  warmupMs, curve[]. Error rows only have detector/recognizer/detSize/tilingMode/
  error.
- `recipe` is a self-contained config (detector, recognizer, detSize, tiling,
  det/nms thresholds, accept/review/margin, minQuality, assignment, template,
  topK, fusion/k, aug flags, variant). The UI can apply/copy/download it.
- `_save_benchmark`: writes `<runId>.json` atomically, keeps
  `benchmarks_to_keep` newest runs, and prunes each removed run's preview folder.

`/api/benchmark/sweep` (`_run_sweep_sync` → `_sweep_dataset`) re-scores datasets
from cached frames (409 if missing) and returns `{attendance, datasets: [{datasetId,
name, attendance, detection, students, faces}], params}`.

### 9.7 Enrollment flow
- `POST /api/people`: `_resolve_recognizer_ids([*recognizers, recognizer])`
  (empty → all available), `_create_person_sync` → `_process_reference_photos`
  (`_embed_reference` per image, image decoded with EXIF handling) under
  `_model_lock`, then under `_state_lock` assign the next number, write face
  JPEGs to `faces/p<number>/<photoId>.jpg`, append photos, `_save_state`.
  Raises 422 for no face/undecodable/multiple? — multiple faces now picks the
  dominant one and only fails when no face is found.
- `_add_photos_sync`: same, respects `max_photos_per_person`.
- `_reembed_sync`: for available recognizers, backfills missing embeddings per
  photo; snapshots work under `_state_lock`, runs inference under `_model_lock`,
  and merges/saves back under `_state_lock` (never nesting the two locks).

### 9.8 Retention & reset
- Results JSON: `results_to_keep` (`_prune_results`).
- Scan frame folders: `scans_to_keep` (`_prune_scans`).
- Dataset folders: `datasets_to_keep` (`datasets._prune_datasets`).
- Cache: `cache_entries_to_keep` (LRU via mtime touch).
- Benchmarks: `benchmarks_to_keep` (JSON + preview dir).
- `/api/reset`: clears `_people`, re-saves an empty roster, `rmtree`s faces,
  scans, results, crops, cache, datasets, benchmarks, re-creates dirs.

### 9.9 Helpers worth knowing
`_require_detector`/`_require_recognizer` (registry + availability → 422),
`_legacy_ids` (pack name → default detector/recognizer), `_analyze_or_cache`
(cache-or-infer, `force_fresh` bypass), `_pick_detector`, `_student_snapshot`
(deep-ish copy under lock), `_registry_state` (adds template/people usage),
`_model_info_sync` (warm pair or legacy pack), `_write_jpeg_bgr/rgb` (relative to
`data_dir`), `_decode` (size/format guards), `_weight_sha256`
(`models_manifest.json` lookup).

---

## 10. Frontend (`static/`)

Three files, no build step. `app.js` is an IIFE-free script with a single global
`state` object; `index.html` holds all DOM; `styles.css` is the full design system.

### 10.1 State (`app.js` `const state`)
Roster/registry (`people`, `registry`, `packs`), selections (`detector`,
`recognizer`), saved artifacts (`results`, `datasets`, `benchmarks`), scan state
(`scan`, `selectedFrame`, `inspector`, `liveSweep`, `sweepTimer`), benchmark
display (`benchmarkRun`, `benchSort`, `benchSelected`), transient UI
(`faceFiles`, `frameFiles`, `datasetFiles`, `datasetPresent`, `datasetBoxes`,
`busy`, `toastTimer`), and `settings` (defaults mirrored from the API, including
`maxPhotosPerUpload`, `maxPhotosPerPerson`, `templateTopK`).

`localStorage`: `facelab.selection.v3` (`{detector, recognizer}`),
`facelab.token` (API token), `facelab.view` (`{labelUnknown, rosterCollapsed}`).

### 10.2 API client
`api(path, options, retry=true)` adds `X-API-Token` from storage, prompts once on
401, re-fetches, and converts FastAPI errors (`detail` string or array) into an
`Error` with `.status`. `jsonRequest(body)` = POST + JSON headers.

### 10.3 Init & load
`init()`: `wireEvents`, `applyViewPrefs`, initialize sliders/placeholders, then
`await loadState()` + `loadModels()` sequentially, then
`Promise.allSettled([loadResults, loadDatasets, loadBenchmarks,
refreshModelStatus])`, then `switchTab(hash || "attendance")`. `loadModels` picks
remembered/default ids only when available (`pick`), builds selects, bench
options. `applySettings` writes every control (using `setSelectValue` so unknown
sizes/modes are appended, and `constrainDetSize` disables sizes a detector
doesn't support).

### 10.4 Tabs
`TABS = ["attendance","benchmark","datasets"]`; `switchTab` toggles
`#tabs .tab.active`, hides/shows `.workspace[data-panel]`, closes the lightbox,
updates the hash (unless `silent`), and re-renders datasets/bench/dataset options.

### 10.5 Enrollment
`#add-person-form` submit → `addPerson` (FormData with `images`, `name`; leaves
`recognizer` empty ⇒ all installed) → `applyState`. People list supports add
photos (hidden file input), remove photo/person, double-click rename (`PATCH`),
and **Re-embed all**. `applyState` refreshes people, pickers, present picker, and
bench options.

### 10.6 Scan
`scanForm()`: FormData from `readSelection()` + `labelUnknown` + frames.
`runScan`: `state.busy` guard, disables button, posts `/api/scan`, stores
`state.scan`, closes inspector, renders, refreshes history; errors go to
`#scan-error`/toast.

`readSelection()` reads: detector, recognizer, detSize (fallback to settings),
tiling, det/nms thresholds, accept/review, margin, minQuality, assignment,
template strategy, fusion mode/k (rounded ≥1), flip/lowres/quality/pose.

### 10.7 Live sweep
`scheduleSweep = debounce(liveSweep, 220)`; wired to accept/review/margin/
minQuality/fusion-k/assignment/template/fusion controls. `liveSweep` keeps a
`sweepSeq` guard, posts the current selection + the scan's stored detector/
recognizer/size/tiling/variant/topK, then on success updates frame faces,
`scan.people` (preserving the full roster), summary counts/needsReview/unknown,
re-renders, and reopens the inspector if it was open. Detector thresholds are
intentionally not swept (they change detection, requiring a fresh scan).

### 10.8 Results / frame viewer / inspector
- `#results` vs `#empty-results` visibility. Summary cards (present/review/faces/
  latency), roster split (present+review vs absent), frame tabs.
- `renderFrameInto(container, frame, index)` is shared by the inline viewer and
  the lightbox: builds an `<img>` + `.box-layer` with one `.face-box`
  (status class `matched`/`review`/`unknown`, dataset `person`/`face`) and a
  `.box-label`; boxes <7% get `.compact` (labels hidden until zoomed).
- Toolbar: **Hide roster** (`#roster-toggle`), **Label unknown faces**
  (`#label-unknown`, toggles `body.show-unknown-labels`), **Fullscreen**
  (`#frame-fullscreen`).
- Lightbox (`#frame-lightbox`): zoom/pan via CSS transform on `#lightbox-stage`,
  wheel zoom around the cursor, window-level pointer drag, double-click reset,
  prev/next, arrow keys, `f` to open, `Esc` to close; counter-scales borders and
  labels with `--inv`. Shared by scan and benchmark frames.
- `openInspector(faceIndex, frame, scope)` shows aligned crop, landmarks SVG,
  status/name/confidence/margin, size bucket/height/det score/quality/pose, top
  candidates, and the box; it is `position: fixed` (z 50) at body level so it
  works from the benchmark tab too. `closeInspector` hides it and clears active
  boxes.

### 10.9 Benchmark UI
`renderBenchOptions` builds detector/recognizer checkbox lists (preserving prior
checks; fixed-size detectors annotated “Npx only”), size/tiling chips.
`runBenchmark` validates datasets/detectors/recognizers and non-empty
size/tiling, builds `BenchmarkRunRequest` (including `force_fresh`,
`label_unknown`, template/topK/k/minQuality), disables the button, posts, renders
the matrix. Matrix columns (`BENCH_COLUMNS`): detector, recognizer, res, tiling,
F1 (ties broken by `simMargin`), precision, recall, τ*, Mean Sim, Impostor,
L/M/S recall (`—` when no GT boxes), Faces L/M/S (detected counts), FPS
(`*` when cached), det p50, total p50. Clicking a row renders the detail:
metrics grid, threshold curve SVG, detected-by-size counts, GT recall grids (only
with annotations, else a note), and a frame strip (thumbnail per frame →
lightbox). Detail actions: **Apply to scan** (`applyRecipeToScan` writes every
control + detector/recognizer + `refreshModelStatus` + switches tab), **Copy
settings** (clipboard JSON via `benchmarkRecipe`), **Download JSON**.

### 10.10 Datasets UI
Builder: name, present-student chips (`state.datasetPresent`), notes, frames.
`createDataset` posts multipart with `presentIds`/`boxes` JSON. `state.datasetBoxes`
is **never populated by the UI** — so datasets currently have no annotated boxes
(see §15). List supports benchmark/delete.

### 10.11 CSS map (`styles.css` sections)
Top bar (141), workspace grid (320), inputs (438), enrollment (702), scan setup
(1004), results/viewer (1148), history (1571), datasets (1685), benchmark (1738),
inspector (2017), empty (2139), toast (2163), responsive (2198; breakpoints
1360/1180/920/560), reduced motion (2447). Key layout: attendance is a 3-pane
grid; results are a 2-column `matches-col`/`frames-col`; lightbox z 40, inspector
z 50, toast z 60; `.frame-media img` max-height `min(78vh,900px)`.

---

## 11. Helper scripts

### 11.1 `scripts/fetch_models.py`
- Targets InsightFace packs from `KNOWN_PACKS` + optional YuNet + UniFace weights
  (`required_uniface_weights()` = registry weights + quality/pose predictors).
- `--all` = every pack + YuNet + those UniFace weights; `--model/--pack`
  repeatable; `--uniface`; `--models-dir`/`--data-dir`; `--base-url`; `--force`;
  `--list`; `--yunet`.
- `needs_download` skips a pack only if it has `.onnx` files and **all**
  `required` files. `_extract_flat` flattens an optional single top folder and
  blocks `..`. `write_manifest` SHA-256s every `.onnx` into
  `.data/models_manifest.json` (used for `detectorSha256`/`recognizerSha256`).
- Re-execs into `.venv` unless `FACELAB_SKIP_REEXEC=1`.

### 11.2 `scripts/crop_faces.py`
- Standalone batch helper (no server/roster/cache changes).
- Defaults: input `scripts/incoming`, output `scripts/face-crops` (both ignored).
- Decodes via `image_to_bgr` (EXIF-aware); detects **all** faces with the chosen
  detector; filters `--min-face`; optional `--dominant`/`--max-faces`.
- Writes a **padded, natural** crop from the original image (`padded_box`:
  square around the face centre, `--padding` of the larger side, clamped), an
  optional 112×112 `--aligned` crop, a JSON/CSV `manifest`, and a labelled
  `contact-sheet.jpg` (up to 240 thumbnails).
- `--tiling` for wide shots; `--dry-run`, `--overwrite`, `--quality`, `--recursive`.
- Pure helpers (`padded_box`, `sanitize_stem`, `base_name`) are importable for
  tests; the venv re-exec is guarded under `__main__`.
- Use the padded crops as enrollment photos (single-face portraits).

### 11.3 `scripts/start.sh`
Container entrypoint: optionally `fetch_models.py --model $FACELAB_DEFAULT_MODEL`
(or `--all` when `FACELAB_FETCH_MODELS=all`), then `server.py --host --port`.

---

## 12. Tests (`tests/`)

No ONNX required; all pure Python.

- `test_matching.py`: size buckets, flip averaging, similarity strategies, matrix
  shape/order, single-frame assignment (present/review/unmatched, margin gate,
  Hungarian vs greedy vs margin, empty inputs), k-of-n (impostor suppression,
  accept requirement, Hungarian/margin/min-quality, k clamping), max-gate
  quality/review, cluster merge, attendance metrics, dominant-face selection,
  annotation bbox-only drawing.
- `test_benchmark.py`: gallery variants/fallback, embedding selection (incl.
  lowres preference), perfect attendance, calibration/FAR0, stratified buckets,
  aggregate attendance/detection, latency percentiles, threshold-curve
  aggregation, curve↔direct agreement, dataset id traversal, operating-point
  plateau, default sweep floor 0.25, similarity aggregates, detected buckets,
  unknown-label toggle.
- `test_crop_faces.py`: padded-box geometry/clamping/non-square, name sanitizing,
  recursive base names.

Run: `.venv/bin/python -m pytest -q` (currently **82 passing**).

---

## 13. End-to-end flows

### 13.1 Enrollment
```
upload images ──► _resolve_recognizer_ids (empty ⇒ all available)
             ──► _process_reference_photos  (per image: decode → detect → dominant face
                                             → align 112 → embed with every recognizer
                                             → flip/lowres variants → quality/pose)
             ──► _state_lock: assign number, write faces/p<N>/*.jpg, append, _save_state
```
Embeddings are stored per recognizer so switching models never needs re-enrollment;
**Re-embed all** backfills recognizers added later.

### 13.2 Scan (inference)
```
frames ──► decode (≤12MB each, ≤max frames)
       ──► _analyze_or_cache(detector, det_params, recognizer, flip)
             cache hit  → payload
             cache miss → analyze_frame → cache JSON + aligned crop JPEG
       ──► fusion_frames (select variant embedding)
       ──► fuse_multi_frame (assignment + k-of-n/max/cluster)
       ──► serialize faces, write frame + annotated JPEG
       ──► roster (all students, status/counts), summary/timings
       ──► prune scans/cache, save results/<scanId>.json
```

### 13.3 Live sweep (no inference)
```
slider change ──► debounce 220ms ──► POST /api/results/<id>/sweep
             ──► load result + cached frames ──► fuse_multi_frame(new params)
             ──► update faces + roster rows + summary counts ──► re-render
```

### 13.4 Benchmark
```
grid request ──► resolve config (availability, input_sizes, cap)
             ──► per combo: warm-up → gallery → per-frame analyze/cache
                 → threshold_curve_datasets (0.25..0.75) → pick operating points
                 → evaluate_frames chosen threshold (attendance + detection)
                 → frames metadata + 1 preview JPEG → recipe/timings/FPS
             ──► run JSON + prune old runs (JSON + preview dirs)
```

### 13.5 Dataset / calibration
```
frames + present roster (+ optional boxes) ──► dataset manifest
sweeps read cached frames ◄── detector/recognizer/params must match cache keys
```

---

## 14. Invariants & gotchas (read before changing things)

1. **Cache key parity.** Scans and benchmarks must build `analysis_params`
   identically; any new detection parameter must go into both
   `analysis_params` and `_param_fingerprint`, or sweeps/benchmarks miss.
2. **Crop-index embeddings.** `analyze_frame` indexes embeddings by crop
   position; don't reintroduce detection-index loops (keypoints may be absent).
3. **Bounded detector cache.** `_DETECTORS` is an LRU of 6; every distinct
   threshold/size combo creates a session. Don't make it unbounded.
4. **Fixed detector sizes.** Registry `input_sizes` (YOLOv8 = 640) is enforced in
   benchmark combination building and `POST /api/scan`; add it for any new
   fixed-shape ONNX model.
5. **`input_sizes` → UI.** The scan-size select disables unsupported options and
   the benchmark detector list annotates fixed sizes.
6. **Threshold semantics.** `accept ≥ review`; the margin gate rejects ambiguous
   top-1 matches; `max_with_margin_gate` reports review-level faces now.
7. **k-of-n**. `required = max(1, min(k, num_frames))`; single-frame fallback was
   deliberately removed (it ignored `min_quality`).
8. **Calibration floor.** Sweep starts at 0.25; operating points pick the top of
   the plateau. Do not restore the old `F1=1.0 @ 0.15` behaviour.
9. **Size-stratified recall needs annotated boxes.** The Datasets UI does not
   collect boxes (`state.datasetBoxes` is always empty), so L/M/S recall is `—`
   and `detected_buckets` is the available size view. Adding a box editor is the
   missing feature.
10. **Benchmark truncates the grid** at `benchmark_max_rows` (default 32) after
    filtering. Combos are detector→recognizer→size→tiling; raise the setting for
    large grids.
11. **One preview JPEG per combo**, not per frame. Frames are metadata only
    (image URLs point at dataset frames). `landmarks5pt` is stripped from
    benchmark frames to limit JSON size.
12. **Cached timings.** On a cache hit, benchmark rows report the timings recorded
    when the frame was first analyzed; `cachedFrames`/`fps *` flag it. Use
    `force_fresh` for honest speed numbers (it re-runs and overwrites the cache).
13. **Benchmark run JSON size** grows with frames × faces. `_list_benchmarks`
    reads lightweight `<runId>.summary.json` sidecars (backfilled on demand).
14. **Ephemeral storage wipes `.data` at server start.** `server.py` calls
    `prepare_dirs()` before mounts and `_load_state()` at import; importing
    `facelab.config` alone never wipes data.
15. **Locks.** Never nest `_state_lock` and `_model_lock` across inference:
    snapshot under `_state_lock`, run ONNX under `_model_lock`, then merge
    under `_state_lock`.
16. **Largest-face enrolment.** Multiple faces no longer 422; the dominant face is
    chosen. Keep the "no face" error.
17. **Unknown labels** default to box-only, in both the burned-in JPEG
    (`label_unknown`) and the HTML overlay (`body.show-unknown-labels`). The scan
    JPEG is rendered once; the overlay toggle is instant.
18. **Inspector/lightbox layers.** Inspector is fixed at body level (z 50), above
    the lightbox (z 40) and below the toast (z 60). Keep it outside the hidden
    `<main>` panels.
19. **Auth scope.** Mutating routes (including sweeps) and `GET /api/model`
    (evicts/warms GPU models) use `require_token`. Other read routes are open.
20. **Path guards.** Result/benchmark ids are hex/dash-only; dataset ids match
    `^[A-Za-z0-9_-]{1,80}$`; media is served from dedicated mounts.
21. **onnxruntime warning** (`{1,512}` vs `{N,512}` for ArcFace output) is benign;
    batching returns correct embeddings.
22. **Frontend contract drift.** New benchmark row fields must be guarded in JS
    (old saved runs lack `recipe`, `frames`, `meanPresentSim`, `detectedBuckets`,
    etc.). `dash()` renders nulls as `—` (unlike `fixed(null)` → `0.000`).
23. **`:focus`/`aria`** — the UI is keyboard-navigable; keep button/ids stable if
    you refactor.

---

## 15. How to extend

### Add a detector
1. Add an entry to `KNOWN_DETECTORS` (`library`, `file_candidates` or
   `uniface_weights`, `landmarks_5pt`, license, `input_sizes` if fixed).
2. Make weights resolvable (`models/<pack>/…` or UniFace cache); add a
   `KNOWN_PACKS` entry + fetch support if it ships as a pack.
3. It appears in `/api/registry`, the scan picker, and benchmark grids.

### Add a recognizer
1. Add to `KNOWN_RECOGNIZERS` with `embedding_dim` and weight source.
2. Enrollment automatically embeds every available recognizer; **Re-embed all**
   backfills existing students. Galleries select by `variant` with raw fallback.

### Change matching/fusion
- Edit `facelab/matching.py` (pure), keep status vocabulary
  (`present`/`needs_review`/`unmatched`/`absent`) and decision keys
  (`matched_student_id`, `score`, `margin`, `top_candidates`) stable; add tests.

### Add an API endpoint
1. Pydantic model in `server.py` if it has a body.
2. `@app.<method>` route, `Depends(require_token)` for mutations.
3. Return the same public shapes (`_public_state`, `_serialize_face`, …) so the
   UI keeps working. Document it in the README API table.

### Add a UI control / benchmark column
- HTML in `index.html`, CSS in `styles.css`, state/defaults in `state.settings`
  (and `_settings_payload` + `.env.example`), wiring in `wireEvents`, and
  `readSelection`/`runBenchmark`/`scanForm`.
- Benchmark columns: add to `BENCH_COLUMNS` with `text` + `value` (sorting) and
  guard missing fields; error-row colspan is computed from the column count.

### Add a config setting
`Settings` field → derived property if needed → `_settings_payload` →
`.env.example` → frontend `state.settings` default + `applySettings`/`readSelection`.

---

## 16. Data lifecycle & file index

```
.data/
  students.json                 roster (v3; legacy people.json migrated)
  faces/p<N>/<photoId>.jpg      enrolled reference photos
  scans/<scanId>/frame-N.jpg    decoded classroom frames
  scans/<scanId>/frame-N-annotated.jpg
  results/<scanId>.json         saved scan + summary + faces
  crops/<cacheKey>_det<i>.jpg   aligned 112×112 crops for the inspector
  cache/<cacheKey>.json         raw inference artifacts (LRU, mtime-touched)
  datasets/<id>/manifest.json + frames/
  benchmarks/<runId>.json       run rows + curves + frame metadata
  benchmarks/<runId>/<NN-slug>.jpg   one annotated preview per combo
  models_manifest.json          SHA-256 map of installed ONNX files
```

`models/<pack>/*.onnx` holds detector/recognizer/alignment weights;
`~/.uniface/models/` holds UniFace weights.

---

## 17. Glossary

- **Detector / recognizer** — independent ONNX stages; any pairing is allowed.
- **Gallery / template** — per-student set of normalized embeddings.
- **Variant** — which stored embedding is scored: `raw`, `flip_avg`, `lowres_aug`.
- **SAHI** — slicing-aided hyper inference (tile + NMS for small faces).
- **Decision** — per-face assignment (`present`/`needs_review`/`unmatched`).
- **Fusion** — combining a frame burst into one attendance roster.
- **Operating point** — a chosen accept threshold (best F1 / FAR=0 / P≥0.98).
- **Recipe** — a self-contained benchmark row config for export/replay.
- **Dataset** — ground-truth frames + present roster (+ optional boxes).
- **Run** — one saved benchmark execution (`runId`).
- **Scan** — one saved classroom inference execution (`scanId`).