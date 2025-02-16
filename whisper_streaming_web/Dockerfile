FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel
RUN pip install uv

RUN apt update && \
    apt install -y ffmpeg && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml .
RUN uv pip install --system -r pyproject.toml
COPY . .
RUN chmod +x /app/entrypoint.sh

ENTRYPOINT [ "/app/entrypoint.sh" ]
CMD [ "python", "whisper_fastapi_online_server.py", "--host", "0.0.0.0", "--port", "8000", "--lan", "en", "--task", "translate", "--model", "large-v3", "--vac", "--vad", "--warmup", "jfk.wav"]