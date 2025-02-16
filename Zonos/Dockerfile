FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel
RUN pip install uv

RUN apt update && \
    apt install -y espeak-ng ffmpeg && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml .
RUN uv pip install --system -r pyproject.toml
COPY . .
RUN uv pip install --system -e . && uv pip install --system -e .[compile]

ENTRYPOINT [ "uvicorn", "server:app"]
CMD [ "--host", "0.0.0.0", "--port", "8001"]
