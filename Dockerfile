FROM text-gen-base-cpu:latest

COPY poetry.lock /app
COPY poetry.toml /app
COPY pyproject.toml /app

RUN poetry install --only main --no-root

COPY src/ /app/src/

WORKDIR /app/src

ENV MODEL_PATH="/models/Llama-2-7B-Chat-GGUF/llama-2-7b-chat.Q4_K_M.gguf"
ENV INDEX_HTML_PATH="/app/src/index.html"

CMD /app/.venv/bin/gunicorn -c hooks_config.py --bind 0.0.0.0:8000 --workers=1 --threads=1 --timeout 600 routes:app
