FROM ghcr.io/astral-sh/uv:0.11.31@sha256:ecd4de2f060c64bea0ff8ecb182ddf46ba3fcccdc8a60cfdbaf20d1a047d7437 AS uv

FROM python:3.11-slim-bookworm@sha256:0bee7276f83efd4a1ee05bbbf4281d95ed28e079220a9457f25a93e3f1e3c31b

COPY --from=uv /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml uv.lock ./
COPY config ./config
COPY src ./src

RUN uv sync --locked --no-dev --no-editable

ENV PATH="/app/.venv/bin:${PATH}"

ENTRYPOINT ["marin-dna-cpt-data"]
