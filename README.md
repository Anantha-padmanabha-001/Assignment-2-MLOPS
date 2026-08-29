# Cats vs Dogs — End-to-End MLOps Pipeline

**MLOps (AIMLCZG523) — Assignment 2**
BITS Pilani WILP | Anantha Padmanabha

## Use case

Binary image classification (Cats vs Dogs) for a pet adoption platform, using
the Kaggle Cats and Dogs dataset. Images are preprocessed to 224x224 RGB and
split 80/10/10 into train/val/test. A 300-image-per-class subset was used for
this demo run to keep training fast and reproducible; the pipeline itself
works unchanged at full dataset scale.

## What's in this repo

| File / Folder | Module | Purpose |
|---|---|---|
| `data_prep.py` | M1.1 | Resize to 224x224 + train/val/test split logic |
| `data/raw.dvc`, `.dvc/` | M1.1 | DVC dataset version tracking (raw images kept out of Git) |
| `train.py` | M1.2, M1.3 | Builds baseline CNN, trains it, logs params/metrics/artifacts to MLflow |
| `catsdogs_model.pt` | M1.2 | Trained model weights (serialized via `torch.save`) |
| `loss_curve.png`, `confusion_matrix.png` | M1.3 | Training artifacts logged to MLflow |
| `app/main.py` | M2.1 | FastAPI inference service — `/`, `/health`, `/predict`, `/metrics` |
| `requirements.txt` | M2.2 | Pinned dependencies |
| `Dockerfile` | M2.3 | Containerizes the inference service |
| `tests/test_model.py` | M3.1 | Unit tests — one data-preprocessing function, one model/inference function |
| `.github/workflows/ci.yml` | M3.2, M3.3, M4 | CI/CD pipeline: lint → test → Docker build → deploy + smoke test |
| `docker-compose.yml` | M4.1 | Deployment target used by the CD job |
| `k8s/deployment.yaml`, `k8s/service.yaml` | M4.1 | Alternate Kubernetes deployment target |
| `scripts/smoke_test.sh` | M4.3 | Post-deploy health + prediction check; fails the pipeline on error |
| `MLOps_Assignment_2.ipynb` | — | Full pipeline run log (data prep → train → test → push), runnable in Colab or locally |

## How it was run

1. Dataset subset (300 cat + 300 dog images) copied into `data/raw/cat` and `data/raw/dog`
2. `dvc init` + `dvc add data/raw` — dataset versioned via DVC, raw images excluded from Git
3. `data_prep.py` — resized and split into `data/processed/{train,val,test}`
4. `train.py` — trained `SimpleCNN` for 5 epochs, tracked via MLflow (params, per-epoch metrics, loss curve, confusion matrix, model artifact)
5. `pytest tests/test_model.py` — 5/5 tests passing
6. `docker build` + `docker run` — verified `/health` and `/predict` return correct responses from inside the container
7. Pushed to GitHub — `.github/workflows/ci.yml` runs automatically on every push:
   - **lint** — flake8 on `app/main.py`
   - **test** — pytest suite
   - **docker** — builds the image, saves it as a workflow artifact
   - **deploy** — loads the image, brings it up via `docker compose`, runs `scripts/smoke_test.sh` against the live `/health` and `/predict` endpoints, tears down; fails the whole pipeline if either check fails

All four CI/CD jobs have run successfully on GitHub Actions for this repository.

## Monitoring (M5)

- `app/main.py` logs every prediction request (filename, predicted label, latency)
  to `api_requests.log` inside the container — no raw image data or other
  sensitive content is logged.
- `GET /metrics` reports running request count and average latency:
  ```json
  {"request_count": 1, "avg_latency_ms": 270.36}
  ```

## How to run this yourself

**Fastest — Docker only, no Python needed:**
```bash
docker build -t catsdogs-api .
docker run -p 8000:8000 catsdogs-api
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict -F "file=@path/to/image.jpg;type=image/jpeg"
```

**Full pipeline from scratch:**
1. Open `MLOps_Assignment_2.ipynb` (Colab or local Jupyter)
2. Run cells top to bottom — mounts Drive (Colab only), installs deps, copies dataset subset, runs DVC, preprocesses, trains with MLflow tracking, runs tests, sanity-checks the API
3. When prompted, paste a GitHub personal access token to push (input is hidden, never saved to the notebook file)

**Kubernetes alternative (instead of Docker Compose):**
```bash
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
```

## Notes

- The dataset itself is not committed to Git — only `data/raw.dvc` (a small
  DVC pointer file) is tracked, per the assignment's dataset-versioning
  requirement. The raw images live locally / in Drive and are excluded via
  `.gitignore`.
- Model accuracy is intentionally modest (~55–65% on a 5-epoch, 300-image
  subset) since the assignment's focus is the MLOps pipeline itself, not
  model performance.
