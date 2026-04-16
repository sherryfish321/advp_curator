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
    --model-id BAAI/bge-reranker-base \
    --port "${INFINITY_PORT:-7997}" &
echo "Infinity started on port ${INFINITY_PORT:-7997}"
# run mainly on gpu if gpu is available, or else automatically run on cpu

# start llama-cpp in foreground
echo "Starting llama-cpp on port ${LLAMA_PORT:-8001}..."
python -m llama_cpp.server \
    --model "$SCRIPT_DIR/qwen2.5-3b-instruct-q8/qwen2.5-3b-instruct-q8_0.gguf" \
    --n_gpu_layers 0 \
    --n_ctx 8192 \
    --port "${LLAMA_PORT:-8001}"
#   --n_gpu_layers -1 \ if we have gpu