python -m vllm.entrypoints.openai.api_server \
    --model /root/autodl-fs/models/TheBloke/Mixtral-8x7B-Instruct-v0___1-GPTQ/ \
    --quantization gptq \
    --dtype float16 \
    --tensor-parallel-size 1 \
    --port 8000 \
    --host 0.0.0.0