# NeuroGraph assistant on a free Hugging Face Space (Docker SDK, 2 vCPU, 16 GB RAM, no GPU)
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
RUN useradd -m -u 1000 user
WORKDIR /home/user/app
COPY requirements.txt .
# CPU torch (for the embedding and NLI models) + the prebuilt CPU wheel of llama.cpp
RUN pip install --no-cache-dir torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu \
 && python -m spacy download en_core_web_sm
USER user
ENV HOME=/home/user PYTHONPATH=/home/user/app/src PYTHONUNBUFFERED=1 HF_HOME=/home/user/.cache/huggingface \
    APP_DATA=/home/user/app/app_data LOG_DIR=/home/user/logs LLM_BACKEND=llamacpp TOKENIZERS_PARALLELISM=false
# small public models baked into the image, so a restart only downloads the GGUF
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('BAAI/bge-small-en-v1.5'); CrossEncoder('cross-encoder/nli-deberta-v3-small')"
COPY --chown=user . .
EXPOSE 7860
CMD ["bash", "start.sh"]
