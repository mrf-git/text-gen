FROM text-gen-base-mistral-cpu:latest

COPY poetry.lock /app
COPY poetry.toml /app
COPY pyproject.toml /app

RUN poetry install --only main --no-root

COPY src/ /app/src/

WORKDIR /app/src

ENV MODEL_PATH="/models/Mistral-7B-Instruct-v0.2/mistral-7b-instruct-v0.2.Q5_K_S.gguf"
ENV INDEX_HTML_PATH="/app/src/index.html"

CMD /app/.venv/bin/gunicorn -c hooks_config.py --bind 0.0.0.0:8000 --workers=1 --threads=1 --timeout 600 routes:app
