# Whisper Streaming with FastAPI and WebSocket Integration And Real-Time live translated audio using Zonos Hybrid

This project extends the [Whisper Streaming Web](https://github.com/QuentinFuxa/whisper_streaming_web) implementation by incorporating a live translated audio using [Zonos Hybrid](https://github.com/Zyphra/Zonos). 

![Demo Screenshot](src/web/demo.png)

##  Code Origins

This project reuses and extends code from the original Whisper Streaming repository:
- whisper_online.py, backends.py and online_asr.py: Contains code from whisper_streaming
- silero_vad_iterator.py: Originally from the Silero VAD repository, included in the whisper_streaming project.


### How to Launch the Server
1. **Install Docker Compose**

2. **Run the Servers**:
    
    ```bash
    docker compose up -d
    ```

3. **Open the Provided HTML**:

    - By default, the server root endpoint `/` serves a simple `live_transcription.html` page.  
    - Open your browser at `http://localhost:8000` (or replace `localhost` and `8000` with whatever you specified).  
    - The page uses vanilla JavaScript and the WebSocket API to capture your microphone and stream audio to the server in real time.

