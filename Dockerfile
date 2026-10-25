# syntax=docker/dockerfile:1
FROM python:3.11-slim AS builder

ENV VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

RUN python -m venv "$VIRTUAL_ENV"
WORKDIR /build
COPY requirements.txt ./
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

FROM python:3.11-slim AS runtime

ENV VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN groupadd --system app && useradd --system --gid app --create-home app \
    && mkdir -p /app/results /app/figures \
    && chown -R app:app /app
COPY --from=builder /opt/venv /opt/venv
WORKDIR /app
RUN python -m spacy download en_core_web_sm
COPY --chown=app:app src/ ./src/
COPY --chown=app:app data/ ./data/
USER app

CMD ["python", "src/link_prediction.py"]
