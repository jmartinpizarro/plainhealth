"""
This script contains all the code related with the PlainHealth API
"""

import os

from src.whisper.Whisper import WhisperInference

from dotenv import load_dotenv
from huggingface_hub import login

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware


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
PRECISION: str = 'float16' # precision
RT_BATCH_DURATION: int = 1 # seconds
# because the WhisperInference class allows the inference using files. And guess what it is sent
# to the API... Exactly, a .webm!
RT_INFERENCE = False

# set up and download the model
MODEL_LOAD_ERROR = None
try:
    print(f"[main] :: Downloading the Whisper model with size {MODEL_SIZE}", flush=True)
    model = WhisperInference(model_size=MODEL_SIZE, precision=PRECISION, batch_duration=RT_BATCH_DURATION, rt=RT_INFERENCE)
    model.load_model() # download and load into memory
except Exception as e:
    MODEL_LOAD_ERROR = str(e)
    model = WhisperInference(model_size=MODEL_SIZE, precision=PRECISION, batch_duration=RT_BATCH_DURATION, rt=RT_INFERENCE)
    print(f"[main] :: An error has ocurred when loading the Whisper model\n{e}\n", flush=True)


@app.post("/api/audio")
async def transcribe_audio(
    audio: UploadFile = File(...),
    session_id: str | None = Form(None),
):
    if MODEL_LOAD_ERROR or model.model is None:
        raise HTTPException(
            status_code=503,
            detail=f"Whisper model is not available: {MODEL_LOAD_ERROR or 'model not loaded'}",
        )

    print(f"[api] :: Received full audio session={session_id}", flush=True)

    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Uploaded audio is empty")

    try:
        segments, info = model.inference(bytearray(audio_bytes), beam_size=1)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not decode/transcribe audio: {exc}") from exc

    print("[Whisper] :: Detected language '%s' with probability %f" % (info.language, info.language_probability), flush=True)

    segment_iterator = iter(segments)

    texts: list[str] = []
    segment_index = 0
    while True:
        try:
            text = model.write_logs(segment_iterator, segment_index=segment_index)
            if text:
                texts.append(text.strip())
            segment_index += 1
        except StopIteration:
            break

    transcription = " ".join(texts).strip()

    return {
        "message": "Audio processed",
        "session_id": session_id,
        "text": transcription,
    }