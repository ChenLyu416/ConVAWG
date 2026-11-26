#!/bin/bash
# Start a vLLM server for the open-weight baseline models
# =======================================================
# Usage:
#   ./start_vllm_baselines.sh qwen3.5-35b         # Qwen/Qwen3.5-35B-A3B on port 8003
#   ./start_vllm_baselines.sh llama3.1-70b-awq    # LLaMA-3.1-70B-Instruct (AWQ) on port 8001
#
#   TP=2 ./start_vllm_baselines.sh qwen3.5-35b    # override tensor-parallel size
#   Select GPUs with CUDA_VISIBLE_DEVICES.
#
# NOTE: gpt5.2, gemini-flash-lite and deepseek-v4-flash are remote API models:
#       no vLLM server is needed; set OPENAI_API_KEY / GEMINI_API_KEY / DEEPSEEK_API_KEY.

case "$1" in
  qwen3.5-35b|qwen3.5)
    MODEL="Qwen/Qwen3.5-35B-A3B"
    PORT=8003
    DEFAULT_TP=4   # MoE: full weights ~70 GB
    # qwen3_5_moe is native in vLLM >= 0.17 (do NOT pass --model-impl
    # transformers: the transformers fallback rejects this architecture).
    # On 24 GB cards, cap concurrency and utilization or the sampler warmup OOMs.
    EXTRA_ARGS="--max-num-seqs 32 --gpu-memory-utilization 0.85"
    ;;
  llama3.1-70b-awq|llama3.1)
    MODEL="hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4"
    PORT=8001
    DEFAULT_TP=4
    ;;
  *)
    echo "Usage: $0 <qwen3.5-35b|llama3.1-70b-awq>"
    echo ""
    echo "Remote-API models (no server needed):"
    echo "  gpt5.2             → set OPENAI_API_KEY"
    echo "  gemini-flash-lite  → set GEMINI_API_KEY"
    echo "  deepseek-v4-flash  → set DEEPSEEK_API_KEY"
    exit 1
    ;;
esac

TENSOR_PARALLEL="${TP:-$DEFAULT_TP}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
HOST="0.0.0.0"

echo "Model           : $MODEL"
echo "Port            : $PORT"
echo "Tensor parallel : $TENSOR_PARALLEL"
echo ""
echo "Once healthy, run baseline_generate.py --model $1 ..."
echo "Press Ctrl+C to stop."
echo ""

python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --host "$HOST" \
    --port "$PORT" \
    --tensor-parallel-size "$TENSOR_PARALLEL" \
    --dtype auto \
    --enable-chunked-prefill \
    $EXTRA_ARGS \
    --max-model-len 32768
