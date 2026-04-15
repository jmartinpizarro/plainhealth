"""
This script contains the classes related with all the audio request headers and its parameters
for multipart/form-data requests.
"""

from pydantic import BaseModel

class AudioRequestModel(BaseModel):
    """
    Metadata for requesting a transcription chunk from the frontend.

    Audio is not included in this model because for RT it is sent as an uploaded file
    in multipart/form-data (for example, audio/webm;codecs=opus).
    """
    session_id: str
    chunk_index: int
    is_last: bool