#!/usr/bin/env bash
set -euo pipefail

NAMESPACE="user-asicard"
APP_LABEL="rfpop-app"
DEPLOYMENT_NAME="rfpop-deployment"
SERVICE_NAME="rfpop-service"
LOCAL_PORT="${1:-8501}"

if [[ "${LOCAL_PORT}" =~ ^-h|--help$ ]]; then
  echo "Usage: scripts/run_deployment.sh [LOCAL_PORT]"
  echo "Example: scripts/run_deployment.sh 8502"
  exit 0
fi

if ! [[ "${LOCAL_PORT}" =~ ^[0-9]+$ ]]; then
  echo "Error: LOCAL_PORT must be a number." >&2
  exit 1
fi

echo "Applying Kubernetes resources..."
kubectl apply -f deployment/

NAMESPACE_ARGS=("-n" "${NAMESPACE}")

echo "Waiting for deployment rollout..."
kubectl "${NAMESPACE_ARGS[@]}" rollout status deployment/"${DEPLOYMENT_NAME}" --timeout=180s

echo "Current pods:"
kubectl "${NAMESPACE_ARGS[@]}" get pods -l app="${APP_LABEL}" -o wide

echo "Recent app logs:"
kubectl "${NAMESPACE_ARGS[@]}" logs -l app="${APP_LABEL}" --tail=60

echo "Starting port-forward on http://localhost:${LOCAL_PORT} ..."
kubectl "${NAMESPACE_ARGS[@]}" port-forward service/"${SERVICE_NAME}" "${LOCAL_PORT}:80"
