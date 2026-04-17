# Changepoint Detection in the Presence of Outliers

This repository contains an implementation of the R-FPOP changepoint detection approach for time series with outliers, packaged with an interactive Streamlit app to explore the algorithm's behavior across different loss functions and hyperparameters.

The project is based on:
> Fearnhead, P., & Rigaill, G. (2019). *Changepoint Detection in the Presence of Outliers*. Journal of the American Statistical Association, 114(525), 169-183.

## About This Project

This project is part of the **"Mise en Production des Projets de Data Science"** (Bringing Data Science to Production) course at ENSAE (2026), taught by [Lino Galiana](https://github.com/linogaliana) and Romain Avouac.

For more information about the course, visit: [ensae-reproductibilite.github.io](https://ensae-reproductibilite.github.io)

## What This Project Includes

- Robust changepoint detection algorithms implemented in Python (`src/`)
- A Streamlit web application (`app.py`) to test the algorithm interactively
- Toy datasets located in `data/`
- Docker image support for both local execution and cloud deployment
- Kubernetes manifests in `deployment/` (configured with Kustomize for reproducibility)

## Repository Layout

- `app.py`: Streamlit application entry point
- `src/`: Core algorithm, loss functions, model selection, and visualization
- `data/`: Built-in CSV examples used by the app
- `deployment/`: Kubernetes Deployment, Service, Ingress, and Kustomization manifests
- `scripts/run_docker.sh`: Script to build and run the Docker image locally
- `scripts/run_deployment.sh`: Script to deploy to Kubernetes and port-forward locally

---

## Quick Start For Developers

### Prerequisites

- Python 3.x
- `pip`
- Optional: Docker
- Optional: `kubectl` with a configured cluster context (e.g., SSPCloud)

### Local Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Run The App Locally

```bash
streamlit run app.py --server.address=0.0.0.0 --server.port=8501
```
Open your browser and navigate to: `http://localhost:8501`

---

## Run With Docker

### Option 1: Build & Run Locally

```bash
docker build -t rfpop-streamlit-app .
docker run --rm -p 8501:8501 rfpop-streamlit-app
```

### Option 2: Use the Helper Script

```bash
chmod +x scripts/run_docker.sh
scripts/run_docker.sh
```
*(To use a custom host port, pass it as an argument: `scripts/run_docker.sh 8504`)*

### Option 3: Pull From Docker Hub

```bash
docker pull audricms/r-fpop-change-points-detection:latest
docker run --rm -p 8501:8501 audricms/r-fpop-change-points-detection:latest
```
*(Note for Apple Silicon users: append `--platform linux/amd64` to the run command if necessary).*

---

## Run On Kubernetes (SSPCloud)

This project is built for reproducibility. We use **Kustomize** so you can easily deploy the app to your personal Kubernetes namespace without altering the core deployment files.

### 1. Configure Your Environment
Before deploying, open the `deployment/kustomization.yaml` file and update it with your personal SSPCloud username:
1. Change `namespace: user-asicard` to your active namespace (e.g., `user-jsmith`).
2. Update the two Ingress URL hostnames under the `patches:` section to ensure your URL is unique (e.g., `rfpop-change-points-jsmith.lab.sspcloud.fr`).

### 2. Deploy
Once configured, deploy the application using the `-k` (Kustomize) flag:
```bash
kubectl apply -k deployment/
```

### 3. Monitor
Check the status of your pods and view logs:
```bash
kubectl get pods -l app=rfpop-app
kubectl logs -l app=rfpop-app -f --tail=200
```

### Clean Up
To remove the application from your cluster:
```bash
kubectl delete -k deployment/
```

---

## Data Sources

The application loads toy CSV datasets from two potential sources:

1. **Local files** stored in the `data/` directory.
2. **Public S3 Storage** (SSPCloud MinIO).

The default remote base URL is configured as:
`https://minio.lab.sspcloud.fr/asicard/MPPDS%20-%20Projet`

You can override this remote base URL by setting the following environment variable in your deployment configuration:
- `PUBLIC_DATA_URL`

## Development Commands

Ensure you have installed the development tools from `requirements.txt`. You can run the following checks:

```bash
pytest
ruff check .
black .
pylint src app.py
```

It is highly recommended to use `pre-commit` to automate formatting before pushing code:

```bash
pip install pre-commit
pre-commit install
```
