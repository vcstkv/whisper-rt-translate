import io
import argparse
import asyncio
import os
import numpy as np
import ffmpeg
import websockets
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

from src.whisper_streaming.whisper_online import backend_factory, online_factory, add_shared_args

import subprocess
import math
import logging


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logging.getLogger().setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

##### LOAD ARGS #####

parser = argparse.ArgumentParser(description="Whisper FastAPI Online Server")
parser.add_argument(
    "--host",
    type=str,
    default="localhost",
    help="The host address to bind the server to.",
)
parser.add_argument(
    "--port", type=int, default=8000, help="The port number to bind the server to."
)
parser.add_argument(
    "--warmup-file",
    type=str,
    dest="warmup_file",
    help="The path to a speech audio wav file to warm up Whisper so that the very first chunk processing is fast. It can be e.g. https://github.com/ggerganov/whisper.cpp/raw/master/samples/jfk.wav .",
)

parser.add_argument(
    "--diarization",
    type=bool,
    default=False,
    help="Whether to enable speaker diarization.",
)


add_shared_args(parser)
args = parser.parse_args()

SAMPLE_RATE = 8000
CHANNELS = 1
SAMPLES_PER_SEC = SAMPLE_RATE * int(args.min_chunk_size)
BYTES_PER_SAMPLE = 2  # s16le = 2 bytes per sample
BYTES_PER_SEC = SAMPLES_PER_SEC * BYTES_PER_SAMPLE
MAX_BYTES_PER_SEC = BYTES_PER_SEC * 5  # 5 seconds of audio at 32 kHz
TIMEOUT = 5

if args.diarization:
    from src.diarization.diarization_online import DiartDiarization


##### LOAD APP #####

@asynccontextmanager
async def lifespan(app: FastAPI):
    global asr, tokenizer
    asr, tokenizer = backend_factory(args)
    yield

app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Load demo HTML for the root endpoint
with open("src/web/live_transcription.html", "r", encoding="utf-8") as f:
    html = f.read()

# Load demo HTML for the root endpoint
with open("src/web/live_tts.html", "r", encoding="utf-8") as f:
    html_tts = f.read()

async def start_ffmpeg_decoder():
    """
    Start an FFmpeg process in async streaming mode that reads WebM from stdin
    and outputs raw s16le PCM on stdout. Returns the process object.
    """
    process = (
        ffmpeg.input("pipe:0", format="webm")
        .output(
            "pipe:1",
            format="s16le",
            acodec="pcm_s16le",
            ac=CHANNELS,
            ar=str(SAMPLE_RATE),
        )
        .run_async(pipe_stdin=True, pipe_stdout=True, pipe_stderr=True)
    )
    return process


##### ENDPOINTS #####

@app.get("/")
async def get():
    return HTMLResponse(html)

@app.get("/tts-test")
async def get():
    return HTMLResponse(html_tts)

@app.websocket("/asr")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    logger.info("WebSocket connection opened.")
    TTS_SERVER_HOSTNAME = os.environ.get("TTS_SERVER_HOSTNAME", "localhost:8001")

    pcm_buffer = bytearray()
    online = online_factory(args, asr, tokenizer)
    diarization = DiartDiarization(SAMPLE_RATE) if args.diarization else None

    try:
        async with websockets.connect(f"ws://{TTS_SERVER_HOSTNAME}/ws") as tts_ws:
            while True:
                try:
                # Receive incoming WebM audio chunks from the client
                    message = await asyncio.wait_for(websocket.receive_bytes(), timeout=5)
                    pcm_buffer.extend(message)
                    # logger.info(f"Received Message: {len(message)} bytes")
                    # logger.info(f"Bytes per Second: {BYTES_PER_SEC} bytes")
                    # logger.info(f"Length of pcm buffer: {len(pcm_buffer)}")
                    # logger.info(f"timer: {curr_time - start_time}")
                except asyncio.TimeoutError:
                    logger.warning("Timeout waiting for audio chunk. Continuing loop...")
                    if len(pcm_buffer) > 0:
                        logger.info(f"Buffer size: {len(pcm_buffer)}")
                        pcm_array = (
                                np.frombuffer(pcm_buffer[:MAX_BYTES_PER_SEC], dtype=np.int16).astype(np.float32)
                                    / 32768.0
                        )
                        pcm_buffer = pcm_buffer[MAX_BYTES_PER_SEC:]
                        logger.info(f"{len(online.audio_buffer) / online.SAMPLING_RATE} seconds of audio will be processed by the model.")
                        online.insert_audio_chunk(pcm_array)
                        transcription = online.process_iter()
                                
                        if transcription.text == "":
                            continue

                        print("Send:", transcription.text)
                        await tts_ws.send(transcription.text)
                        tts = await tts_ws.recv()
                        await websocket.send_bytes(tts)
                    

                    if len(pcm_buffer) >= BYTES_PER_SEC:
                        if len(pcm_buffer) > MAX_BYTES_PER_SEC:
                            logger.warning(
                                f"""Audio buffer is too large: {len(pcm_buffer) / BYTES_PER_SEC:.2f} seconds.
                                The model probably struggles to keep up. Consider using a smaller model.
                                """)
                        # Convert int16 -> float32
                        pcm_array = (
                            np.frombuffer(pcm_buffer[:MAX_BYTES_PER_SEC], dtype=np.int16).astype(np.float32)
                            / 32768.0
                        )
                        pcm_buffer = pcm_buffer[MAX_BYTES_PER_SEC:]
                        logger.info(f"{len(online.audio_buffer) / online.SAMPLING_RATE} seconds of audio will be processed by the model.")
                        online.insert_audio_chunk(pcm_array)
                        transcription = online.process_iter()
                        
                        if transcription.text == "":
                            continue

                        print("Send:", transcription.text)
                        await tts_ws.send(transcription.text)
                        tts = await tts_ws.recv()
                        await websocket.send_bytes(tts)

    except WebSocketDisconnect:
        logger.warning("WebSocket disconnected.")
    finally:
        if args.diarization:
            diarization.close()



if __name__ == "__main__":
    import uvicorn

    uvicorn.run("whisper_fastapi_online_server:app", host=args.host, port=args.port, log_level="info")