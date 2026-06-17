export API_KEY=dummy
export AZURE_ENDPOINT=dummy
export API_VERSION=dummy

export VLLM_BASE_URL=http://127.0.0.1:8000/v1
export VLLM_API_KEY=EMPTY

export CHEMEAGLE_ROOT=/root/autodl-tmp/ipm_eagle/external/ChemEagle


CUDA_VISIBLE_DEVICES=0,1 \
vllm serve Qwen3-VL-32B-Instruct-AWQ \
  --host 0.0.0.0 \
  --port 8000 \
  --served-model-name ipm-vlm \
  --tensor-parallel-size 2 \
  --trust-remote-code \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --limit-mm-per-prompt '{"video": 0}' \
  --gpu-memory-utilization 0.85