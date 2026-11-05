# AETHER-ML backend image. CPU-only by default; pass a CUDA base image and device to accelerate.
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# libgl/libglib are required by Pillow's optional codecs; curl backs the health check.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md alembic.ini ./
COPY aether ./aether
COPY backend ./backend
COPY configs ./configs

# The CPU wheel index keeps the image small and reproducible on machines without CUDA.
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu . \
    && pip install "psycopg[binary]>=3.1"

RUN useradd --create-home --uid 10001 aether \
    && mkdir -p /app/artifacts/db /app/artifacts/uploads \
    && chown -R aether:aether /app
USER aether

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/health || exit 1

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000", "--log-config", "/dev/null"]
