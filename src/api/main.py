"""
This script contains all the code related with the PlainHealth API
"""

import os

from src.whisper.Whisper import WhisperInference

from dotenv import load_dotenv
from huggingface_hub import login

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from models.AudioRequest import AudioRequestModel


# set up env environment variables for faster model downloads
load_dotenv()
HF_TOKEN = os.getenv('HF_TOKEN')
if HF_TOKEN:
    login(token=HF_TOKEN)
else:
    print("[main] :: HF_TOKEN not set; continuing without Hugging Face login", flush=True)

# set up the API configuration, specially the CORS
app = FastAPI()

origins = [
    "http://localhost:5173"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# load the whisper model with its configs
MODEL_SIZE: str = 'medium'
PRECISION: str = 'int8_float16' # precision for gpu
RT_BATCH_DURATION: int = 1 # seconds
# because the WhisperInference class allows the inference using files. And guess what it is sent
# to the API... Exactly, a .webm!
RT_INFERENCE = False

# set up and download the model
model = WhisperInference(model_size=MODEL_SIZE, precision=PRECISION, batch_duration=RT_BATCH_DURATION, rt=RT_INFERENCE)
MODEL_LOAD_ERROR = None
SESSION_STATE: dict[str, dict[str, str | int]] = {}
try:
    print(f"[main] :: Downloading the Whisper model with size {MODEL_SIZE}", flush=True)
    model.load_model() # download and load into memory
except Exception as e:
    MODEL_LOAD_ERROR = str(e)
    print(f"[main] :: An error has ocurred when loading the Whisper model\n{e}\n", flush=True)


@app.post("/api/audio")
async def transcribe_audio(
    session_id: str = Form(...),
    chunk_index: int = Form(...),
    is_last: bool = Form(...),
    audio: UploadFile = File(...),
):
    metadata = AudioRequestModel(
        session_id=session_id,
        chunk_index=chunk_index,
        is_last=is_last,
    )

    if MODEL_LOAD_ERROR or model.model is None:
        raise HTTPException(
            status_code=503,
            detail=f"Whisper model is not available: {MODEL_LOAD_ERROR or 'model not loaded'}",
        )

    print(f"[api] :: Received chunk={metadata.chunk_index} session={metadata.session_id} is_last={metadata.is_last}", flush=True)

    session_state = SESSION_STATE.get(metadata.session_id, {"last_text": "", "last_chunk_index": -1})
    last_chunk_index = int(session_state["last_chunk_index"])
    if metadata.chunk_index <= last_chunk_index:
        return {
            "message": "Chunk ignored (out-of-order or duplicated)",
            "session_id": metadata.session_id,
            "chunk_index": metadata.chunk_index,
            "is_last": metadata.is_last,
            "text": "",
        }

    # Read uploaded webm bytes and transcribe directly from memory.
    audio_bytes = await audio.read()
    audio_temp = bytearray()
    audio_temp.extend(audio_bytes)

    segments, info = model.inference(audio_temp, beam_size=1)

    print("[Whisper] :: Detected language '%s' with probability %f" % (info.language, info.language_probability), flush=True)

    segment_iterator = iter(segments)

    texts: list[str] = []
    while True:
        try:
            text = model.write_logs(segment_iterator, segment_index=chunk_index)
            if text:
                texts.append(text.strip())
        except StopIteration:
            break

    transcription = " ".join(texts)

    previous_full_text = str(session_state["last_text"])
    incremental_text = transcription
    if transcription.startswith(previous_full_text):
        incremental_text = transcription[len(previous_full_text):].strip()

    SESSION_STATE[metadata.session_id] = {
        "last_text": transcription,
        "last_chunk_index": metadata.chunk_index,
    }

    if metadata.is_last:
        SESSION_STATE.pop(metadata.session_id, None)

    return {
        "message": "Chunk received",
        "session_id": metadata.session_id,
        "chunk_index": metadata.chunk_index,
        "is_last": metadata.is_last,
        "text": incremental_text,
    }