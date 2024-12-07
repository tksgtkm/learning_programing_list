# FROM python3.11-bookworm-slim
# COPY --from=ghcr.io/astral-sh/uv:0.5.5 /uv /uvx /bin/
FROM python:3.11-slim-bookworm

# The installer requires curl (and certificates) to download the release archive
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates

# Download the latest installer
ADD https://astral.sh/uv/install.sh /uv-installer.sh

# Run the installer then remove it
RUN sh /uv-installer.sh && rm /uv-installer.sh

# Ensure the installed binary is on the `PATH`
ENV PATH="/root/.local/bin/:$PATH"

ENV PROJECT_DIR model_db

ADD . /${PROJECT_DIR}}
WORKDIR /${PROJECT_DIR}

RUN uv sync --frozen

COPY src/ /${PROJECT_DIR}/src/

CMD [ "uv", "run", "python" ]

# COPY run.sh /${PROJECT_DIR}/run.sh
# RUN chmod +x /${PROJECT_DIR}/run.sh
# CMD [ "./run.sh" ]