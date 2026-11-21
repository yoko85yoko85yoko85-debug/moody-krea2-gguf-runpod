FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 TOKENIZERS_PARALLELISM=false
RUN apt-get update && apt-get install -y --no-install-recommends git build-essential libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
WORKDIR /opt
ARG COMFY_REV=f99de883d5e65317c7f4a4b617b5aa7936622029
ARG GGUF_REV=3ae2b682c5fdb77609171ceb66cc93f18b976ba1
RUN git init ComfyUI && cd ComfyUI && git remote add origin https://github.com/Comfy-Org/ComfyUI.git && git fetch --depth 1 origin ${COMFY_REV} && git checkout --detach FETCH_HEAD && rm -rf .git
RUN git init /opt/ComfyUI/custom_nodes/ComfyUI-GGUF && cd /opt/ComfyUI/custom_nodes/ComfyUI-GGUF && git remote add origin https://github.com/molbal/ComfyUI-GGUF.git && git fetch --depth 1 origin ${GGUF_REV} && git checkout --detach FETCH_HEAD && rm -rf .git
RUN python -c "import torch; print('torch=='+torch.__version__.split('+')[0])" > /tmp/torch-constraint.txt && pip install -c /tmp/torch-constraint.txt -r /opt/ComfyUI/requirements.txt -r /opt/ComfyUI/custom_nodes/ComfyUI-GGUF/requirements.txt runpod==1.7.9 'huggingface-hub>=0.36,<2'
WORKDIR /app
COPY patch_gguf.py /app/patch_gguf.py
RUN python /app/patch_gguf.py && python -m py_compile /opt/ComfyUI/custom_nodes/ComfyUI-GGUF/loader.py
COPY handler.py runtime.py models.json workflow_api.json ./
RUN python -m py_compile handler.py runtime.py && python runtime.py --check
ENV COMFY_DIR=/opt/ComfyUI MODEL_ROOT=/models HF_HOME=/models/huggingface
# Model files are intentionally NOT downloaded in Docker build.
CMD ["python", "-u", "handler.py"]
