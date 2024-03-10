FROM text-gen-base:latest

COPY poetry.lock /app
COPY poetry.toml /app
COPY pyproject.toml /app

RUN poetry install --only main --no-root

COPY src/ /app/src/

WORKDIR /app/src

ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CMD /app/.venv/bin/gunicorn -c hooks_config.py --bind 0.0.0.0:8000 --workers=1 --threads=1 routes:app
