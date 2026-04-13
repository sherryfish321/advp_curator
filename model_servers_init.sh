#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# activate venv if local
if [ -f "$SCRIPT_DIR/venv/bin/activate" ]; then
    source "$SCRIPT_DIR/venv/bin/activate"
fi

# start infinity in background
infinity_emb v2 \
    --model-id NeuML/pubmedbert-base-embeddings \
    --model-id jinaai/jina-reranker-v1-turbo-en \
    --port "${INFINITY_PORT:-7997}" &
echo "Infinity started on port ${INFINITY_PORT:-7997}"

# start llama-cpp in foreground
echo "Starting llama-cpp on port ${LLAMA_PORT:-8001}..."
python -m llama_cpp.server \
    --model "$SCRIPT_DIR/qwen2.5-7b-instruct-q4/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf" \
    --n_gpu_layers -1 \
    --n_ctx 16384 \
    --port "${LLAMA_PORT:-8001}"
