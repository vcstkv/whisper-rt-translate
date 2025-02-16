import asyncio
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import torchaudio
import uvicorn
import io
import wave
import numpy as np
import torch
import ffmpeg
from zonos.model import Zonos
from zonos.conditioning import make_cond_dict
from zonos.utils import DEFAULT_DEVICE as device

@asynccontextmanager
async def lifespan(app: FastAPI):
    global tts_model, speaker
    tts_model = Zonos.from_pretrained("Zyphra/Zonos-v0.1-hybrid", device=device)

    wav, sampling_rate = torchaudio.load("assets/exampleaudio.mp3")
    speaker = tts_model.make_speaker_embedding(wav, sampling_rate)

    torch.manual_seed(421)

    cond_dict = make_cond_dict(text="Hello World! Warm up text!", speaker=speaker, language="en-us")
    conditioning = tts_model.prepare_conditioning(cond_dict)
    codes = tts_model.generate(conditioning)

    yield

app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Receive text from the client
            text = await websocket.receive_text()
            print(f"Received text: {text}")

            # Generate audio from text
            # Replace this with your actual audio-generation code:
            cond_dict = make_cond_dict(text=text, speaker=speaker, language="en-us")
            conditioning = tts_model.prepare_conditioning(cond_dict)
            codes = tts_model.generate(conditioning)
            wavs = tts_model.autoencoder.decode(codes).cpu()

            # Write the audio to an in-memory WAV file.
            wav_buffer = io.BytesIO()
            torchaudio.save(wav_buffer, wavs[0], tts_model.autoencoder.sampling_rate, format="wav")
            wav_buffer.seek(0)
            wav_bytes = wav_buffer.read()

            await websocket.send_bytes(wav_bytes)

            # process = (
            #     ffmpeg
            #     .input('pipe:0')
            #     .output('pipe:1', format='webm', acodec='libopus', audio_bitrate='64k')
            #     .run_async(pipe_stdin=True, pipe_stdout=True, pipe_stderr=True)
            # )

            # # Write the WAV bytes to ffmpeg's stdin.
            # process.stdin.write(wav_bytes)
            # process.stdin.close()

            # # Read the ffmpeg output in chunks and send them over the WebSocket.
            # while True:
            #     chunk = process.stdout.read(4096)
            #     if not chunk:
            #         break
            #     await websocket.send_bytes(chunk)

            # process.wait()

    except WebSocketDisconnect:
        print("Client disconnected")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("server:app", host="0.0.0.0", port=port)
