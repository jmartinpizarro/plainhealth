"""
This script contains the class definition for the Whisper model inference
"""

import io
from time import perf_counter
from typing import Iterator, TextIO
import ctranslate2
from faster_whisper import WhisperModel


class WhisperInference():
    def __init__(self, model_size: str, precision: str, batch_duration: int, rt: bool) -> None:
        self.model_size: str = model_size
        self.precision: str = precision
        self.batch_duration: int = batch_duration
        self.rt_mode = True if rt else False # either is RT inference or I/O-based inference
        self.model = None
        self.device = "cpu"


    def load_model(self):
        cuda_devices = 0
        try:
            cuda_devices = ctranslate2.get_cuda_device_count()
        except Exception:
            cuda_devices = 0

        # Prefer GPU when available, else fallback to CPU.
        if cuda_devices > 0:
            self.device = "cuda"
            compute_type = self.precision
        else:
            self.device = "cpu"
            compute_type = "int8"

        self.model = WhisperModel(self.model_size, device=self.device, compute_type=compute_type)
        print(f"[WhisperInference] :: Model loaded on {self.device} (compute_type={compute_type})", flush=True)
        return 1


    def inference(self, audio: str | bytearray, beam_size: int = 1):
        """
        Runs the inference of the model

        :param audio: str | bytearray: it can be either a .mp3 file or a bytearray (used for RT
        inference). If a bytearray, it must be in int16, non normalised (format [-32768, 32768])
        :param beam_size: int: tree expansion of the model. The bigger the number, the more precision
        and the bigger the latency
        :returns segments, info. segments(iterable) corresponds for the transcribed data. Info is just
        some information about the model (language predicted)
        """

        if self.model is None:
            raise RuntimeError("[WhisperInference] :: Model is not loaded. Call load_model() first")

        # For browser RT, audio usually arrives encoded (webm/opus).
        # Use an in-memory stream so we do not rely on temporary files.
        if isinstance(audio, str):
            segments, info = self.model.transcribe(
                audio,
                beam_size=beam_size,
                language="es",
                vad_filter=True,
                no_repeat_ngram_size=2,
            )
            return segments, info

        if not isinstance(audio, bytearray):
            raise TypeError("[WhisperInference] :: The <audio> parameter must be a file path or a bytearray")

        audio_stream = io.BytesIO(bytes(audio))
        segments, info = self.model.transcribe(
            audio_stream,
            beam_size=beam_size,
            language="es",
            vad_filter=True,
            no_repeat_ngram_size=2,
        )

        return segments, info
    

    def write_logs(self, segments: Iterator, segment_index: int, f: TextIO | None=None) -> str:
        """
        Write the logs of the inference in a .txt file

        :param segments: Iterable[Segment]: an iterable with all the contents of the predictions
        :param f: TextIO: file decriptor
        :param segment_idx: int: current segment index

        :returns the text transcripted
        """
        start_time = perf_counter()
        try:
            segment = next(segments)
        except StopIteration:
            raise StopIteration
        process_time = perf_counter() - start_time
        batch_duration: float = segment.end - segment.start
        rtf: float = process_time / batch_duration if batch_duration > 0 else 0.0
        if f:
            f.write(
                "[idx=%d] [%.2fs -> %.2fs] [audio=%.2fs] [process=%.4fs] [rtf=%.4f] %s\n"
                % (
                    segment_index,
                    segment.start,
                    segment.end,
                    batch_duration,
                    process_time,
                    rtf,
                    segment.text,
                )
            )
        print(f"[WhisperInference - write_logs()] :: idx {segment_index} - {segment.text}", flush=True)
        return segment.text
