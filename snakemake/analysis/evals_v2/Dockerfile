FROM ghcr.io/astral-sh/uv:0.11.31@sha256:ecd4de2f060c64bea0ff8ecb182ddf46ba3fcccdc8a60cfdbaf20d1a047d7437 AS uv

FROM python:3.13-slim-bookworm@sha256:c45a22ea000adfd9cda29364bbe7edd23001ce5cc2ad15857cfbf7766943b9ca

COPY --from=uv /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml ./pyproject.toml
COPY src ./src
COPY snakemake/analysis/evals_v2/pyproject.toml ./snakemake/analysis/evals_v2/pyproject.toml
COPY snakemake/analysis/evals_v2/uv.lock ./snakemake/analysis/evals_v2/uv.lock
COPY snakemake/analysis/evals_v2/src ./snakemake/analysis/evals_v2/src

RUN cd snakemake/analysis/evals_v2 && \
    uv sync --locked --no-dev --group genome-s3 --no-editable

ENV PATH="/app/snakemake/analysis/evals_v2/.venv/bin:${PATH}"
ENV WANDB_DISABLED="true"

ENTRYPOINT ["marin-dna-eval-cell"]
