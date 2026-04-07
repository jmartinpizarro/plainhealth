"""
This script contains the class definition for the Whisper model inference
"""

from time import perf_counter
from typing import Iterator, TextIO

import numpy as np
from faster_whisper import WhisperModel


class WhisperInference():
    def __init__(self, model_size: str, precision: str, batch_duration: int, rt: bool) -> None:
        self.model_size: str = model_size
        self.precision: str = precision
        self.batch_duration: int = batch_duration
        self.rt_mode = True if rt else False # either is RT inference or I/O-based inference


    def load_model(self):
        # TODO: for the moment, assume always cuda
        self.model = WhisperModel(self.model_size, device="cuda", compute_type=self.precision)
        print("[WhisperInference] :: Model loaded")
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

        if self.rt_mode:
            if not isinstance(audio, (bytearray)):
                raise TypeError("[WhisperInference] :: For RT inference, the <audio> parameter must be a bytearray")
            # now it is needed to normalised to [-1, 1]
            # x_norm = x_int16 / 32768.0 (float32 if possible for precission)
            audio_int16 = np.frombuffer(bytes(audio), dtype=np.int16)
            audio_float32 = audio_int16.astype(np.float32) / 32768.0

            segments, info = self.model.transcribe(audio_float32, beam_size=beam_size)

        else:
            if not isinstance(audio, (str)):
                raise TypeError("[WhisperInference] :: For I/O inference, the <audio> parameter must be a string")
            segments, info = self.model.transcribe(audio, beam_size=beam_size)

        return segments, info
    

    def write_logs(self, segments: Iterator, f: TextIO, segment_index: int):
        """
        Write the logs of the inference in a .txt file

        :param segments: Iterable[Segment]: an iterable with all the contents of the predictions
        :param f: TextIO: file decriptor
        :param segment_idx: int: current segment index
        """
        start_time = perf_counter()
        try:
            segment = next(segments)
        except StopIteration:
            raise StopIteration
        process_time = perf_counter() - start_time
        batch_duration: float = segment.end - segment.start
        rtf: float = process_time / batch_duration if batch_duration > 0 else 0.0
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
