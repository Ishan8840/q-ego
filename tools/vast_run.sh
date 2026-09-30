#!/usr/bin/env bash
set -euo pipefail

MAX_GPU_PER_HOUR="${MAX_GPU_PER_HOUR:-0.25}"
MAX_RUNTIME_SECONDS="${MAX_RUNTIME_SECONDS:-7200}"
GPU_RAM_GB="${GPU_RAM_GB:-8}"
DISK_GB="${DISK_GB:-50}"

BRANCH="${1:-}"
shift || true

COMMAND="$*"

if [[ -z "$BRANCH" ]]; then
    echo "Usage: $0 <git-branch> <command>"
    exit 2
fi

if [[ -z "$COMMAND" ]]; then
    echo "Usage: $0 <git-branch> <command>"
    exit 2
fi

INSTANCE_ID=""
START_TIME=""

cleanup() {
    if [[ -n "${INSTANCE_ID}" ]]; then
        echo "Destroying Vast instance ${INSTANCE_ID}..."
        vastai destroy instance "$INSTANCE_ID" -y || true
    fi
}

trap cleanup EXIT INT TERM

echo "Searching for GPU..."

OFFER_JSON="$(
    vastai search offers \
        "num_gpus=1 gpu_ram>=${GPU_RAM_GB} reliability>=0.98 rentable=true dph<${MAX_GPU_PER_HOUR}" \
        --limit 1 \
        --order 'dlperf_usd-' \
        --raw
)"

OFFER_ID="$(echo "$OFFER_JSON" | jq -r '.[0].id')"

if [[ -z "$OFFER_ID" || "$OFFER_ID" == "null" ]]; then
    echo "No suitable GPU found."
    exit 1
fi

echo "Selected offer: ${OFFER_ID}"

CREATE_JSON="$(
    vastai create instance "$OFFER_ID" \
        --image vastai/pytorch:@vastai-automatic-tag \
        --disk "$DISK_GB" \
        --ssh \
        --direct \
        --label q-ego-agent \
        --cancel-unavail \
        --raw
)"

INSTANCE_ID="$(echo "$CREATE_JSON" | jq -r '.new_contract')"

if [[ -z "$INSTANCE_ID" || "$INSTANCE_ID" == "null" ]]; then
    echo "Failed to create Vast instance:"
    echo "$CREATE_JSON"
    exit 1
fi

echo "Created instance: ${INSTANCE_ID}"

START_TIME="$(date +%s)"

echo "Waiting for instance..."

while true; do
    STATUS_JSON="$(vastai show instance "$INSTANCE_ID" --raw)"
    STATUS="$(echo "$STATUS_JSON" | jq -r '.actual_status')"

    echo "Status: ${STATUS}"

    if [[ "$STATUS" == "running" ]]; then
        break
    fi

    if [[ "$STATUS" == "exited" || "$STATUS" == "offline" || "$STATUS" == "unknown" ]]; then
        echo "Instance failed with status: ${STATUS}"
        exit 1
    fi

    if (( $(date +%s) - START_TIME > 600 )); then
        echo "Timed out waiting for GPU."
        exit 1
    fi

    sleep 10
done

echo "Instance is running."

SSH_URL="$(vastai ssh-url "$INSTANCE_ID")"

echo "SSH URL: ${SSH_URL}"

SSH_HOST="$(echo "$SSH_URL" | sed -E 's#ssh://root@([^:]+):.*#\1#')"
SSH_PORT="$(echo "$SSH_URL" | sed -E 's#.*:([0-9]+)$#\1#')"

echo "Connecting to ${SSH_HOST}:${SSH_PORT}"

SSH_OPTS=(
    -i "$HOME/.ssh/vast_worker"
    -o StrictHostKeyChecking=no
    -o UserKnownHostsFile=/dev/null
    -o ConnectTimeout=30
)

REMOTE="root@${SSH_HOST}"

echo "Waiting for SSH..."

for _ in {1..30}; do
    if ssh "${SSH_OPTS[@]}" -p "$SSH_PORT" "$REMOTE" "echo ready" >/dev/null 2>&1; then
        break
    fi

    sleep 5
done

ssh "${SSH_OPTS[@]}" -p "$SSH_PORT" "$REMOTE" "nvidia-smi"

echo "Preparing repository..."

ssh "${SSH_OPTS[@]}" -p "$SSH_PORT" "$REMOTE" \
    "rm -rf /workspace/q-ego && git clone https://github.com/Ishan8840/q-ego.git /workspace/q-ego && cd /workspace/q-ego && git checkout '$BRANCH'"

echo "Running command:"
echo "$COMMAND"

ssh "${SSH_OPTS[@]}" -p "$SSH_PORT" "$REMOTE" \
    "cd /workspace/q-ego && $COMMAND"

ELAPSED="$(( $(date +%s) - START_TIME ))"

if (( ELAPSED > MAX_RUNTIME_SECONDS )); then
    echo "Runtime limit exceeded: ${ELAPSED}s"
    exit 1
fi

echo "GPU job completed successfully in ${ELAPSED}s."
