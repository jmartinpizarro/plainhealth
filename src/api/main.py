"""
This script contains all the code related with the PlainHealth API
"""

import os
from typing import Any
import httpx

from src.whisper.Whisper import WhisperInference

from dotenv import load_dotenv
from huggingface_hub import login
from av.error import EOFError as AVEOFError

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from models.AudioRequest import AudioRequestModel


# set up env environment variables for faster model downloads
load_dotenv()
HF_TOKEN = os.getenv('HF_TOKEN')
GEMINI_API_KEY = os.getenv('ENV_API_KEY') or os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY')
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-latest:generateContent"

REPORT_SECTIONS = [
    "MOTIVO_DE_CONSULTA",
    "RESUMEN_CLINICO",
    "ANAMNESIS",
    "SINTOMAS_REFERIDOS",
    "ANTECEDENTES_PERSONALES",
    "ANTECEDENTES_FAMILIARES",
    "MEDICACION_ACTUAL",
    "ALERGIAS",
    "EXPLORACION_FISICA",
    "PRUEBAS_COMPLEMENTARIAS",
    "VALORACION",
    "PLAN",
    "RED_FLAGS",
]

MEDICAL_PROMPT_TEMPLATE = """Eres un asistente clinico. Convierte la transcripcion medica libre en un informe medico estructurado.

Reglas de salida (obligatorias):
1) Responde SOLO con texto plano (no JSON, no markdown, no bloque de codigo).
2) Usa EXACTAMENTE este formato de secciones y no omitas ninguna:
MOTIVO_DE_CONSULTA:
RESUMEN_CLINICO:
ANAMNESIS:
SINTOMAS_REFERIDOS:
ANTECEDENTES_PERSONALES:
ANTECEDENTES_FAMILIARES:
MEDICACION_ACTUAL:
ALERGIAS:
EXPLORACION_FISICA:
PRUEBAS_COMPLEMENTARIAS:
VALORACION:
PLAN:
RED_FLAGS:
3) Si falta un dato, escribe exactamente: No referido.
4) No inventes diagnosticos, hallazgos o tratamientos no mencionados.
5) Usa lenguaje medico formal, claro y conciso.
6) Mantener idioma: espanol.

Transcripcion:
"""

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
model = WhisperInference(model_size=MODEL_SIZE, precision=PRECISION, batch_duration=RT_BATCH_DURATION, rt=RT_INFERENCE)
MODEL_LOAD_ERROR = None
SESSION_STATE: dict[str, dict[str, str | int]] = {}
try:
    print(f"[main] :: Downloading the Whisper model with size {MODEL_SIZE}", flush=True)
    model.load_model() # download and load into memory
except Exception as e:
    MODEL_LOAD_ERROR = str(e)
    print(f"[main] :: An error has ocurred when loading the Whisper model\n{e}\n", flush=True)


def _extract_gemini_text(payload: dict[str, Any]) -> str:
    candidates = payload.get("candidates") or []
    if not candidates:
        return ""

    content = candidates[0].get("content") or {}
    parts = content.get("parts") or []
    texts: list[str] = []
    for part in parts:
        text = part.get("text")
        if isinstance(text, str) and text.strip():
            texts.append(text)
    return "\n".join(texts).strip()


async def _generate_medical_report(transcription: str) -> str | None:
    if not GEMINI_API_KEY:
        print("[Gemini] :: API key missing. Set ENV_API_KEY (or GEMINI_API_KEY/GOOGLE_API_KEY).", flush=True)
        return None

    if not transcription.strip():
        return None

    request_payload: dict[str, Any] = {
        "contents": [
            {
                "parts": [
                    {
                        "text": f"{MEDICAL_PROMPT_TEMPLATE}{transcription}",
                    }
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.2,
            "topP": 0.9,
            "topK": 20,
        },
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            GEMINI_API_URL,
            headers={
                "Content-Type": "application/json",
                "X-goog-api-key": GEMINI_API_KEY,
            },
            json=request_payload,
        )

    if response.status_code != 200:
        print(f"[Gemini] :: Request failed ({response.status_code}): {response.text}", flush=True)
        return None

    payload = response.json()
    model_text = _extract_gemini_text(payload)
    if not model_text:
        print("[Gemini] :: Empty model response", flush=True)
        return None

    return _ensure_complete_report(model_text)


def _ensure_complete_report(report_text: str) -> str:
    cleaned = report_text.strip()
    normalized = cleaned.upper()

    missing_sections: list[str] = []
    for section in REPORT_SECTIONS:
        marker = f"{section}:"
        if marker not in normalized:
            missing_sections.append(section)

    if not missing_sections:
        return cleaned

    # Ensure every section exists to avoid incomplete rendering downstream.
    completed = cleaned
    if completed:
        completed += "\n\n"
    completed += "\n\n".join([f"{section}:\nNo referido." for section in missing_sections])
    return completed


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

    try:
        segments, info = model.inference(audio_temp, beam_size=1)
    except AVEOFError:
        # Keep session index updated and skip undecodable chunk.
        SESSION_STATE[metadata.session_id] = {
            "last_text": str(session_state["last_text"]),
            "last_chunk_index": metadata.chunk_index,
        }
        return {
            "message": "Chunk received but not decodable",
            "session_id": metadata.session_id,
            "chunk_index": metadata.chunk_index,
            "is_last": metadata.is_last,
            "text": "",
        }

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
    delta_text = transcription
    if transcription.startswith(previous_full_text):
        delta_text = transcription[len(previous_full_text):].strip()

    SESSION_STATE[metadata.session_id] = {
        "last_text": transcription,
        "last_chunk_index": metadata.chunk_index,
    }

    medical_report: str | None = None
    medical_report_status = "not_requested"
    if metadata.is_last:
        medical_report = await _generate_medical_report(transcription)
        medical_report_status = "ok" if medical_report else "failed_or_missing_api_key"
        SESSION_STATE.pop(metadata.session_id, None)

    return {
        "message": "Chunk received",
        "session_id": metadata.session_id,
        "chunk_index": metadata.chunk_index,
        "is_last": metadata.is_last,
        "text": delta_text,
        "full_transcript": transcription if metadata.is_last else "",
        "medical_report_status": medical_report_status,
        "medical_report": medical_report,
    }
