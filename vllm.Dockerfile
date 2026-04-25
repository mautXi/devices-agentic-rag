FROM vllm/vllm-openai-cpu:latest
RUN pip install --no-cache-dir torchvision --index-url https://download.pytorch.org/whl/cpu
