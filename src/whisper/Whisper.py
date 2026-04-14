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
        :param beam_size: int: tree expansion of the model. The bigger the number, the more 
        precision and the bigger the latency
        :returns segments, info. segments(iterable) corresponds for the transcribed data. Info
        is just some information about the model (language predicted)
        """

        if self.rt_mode:
            if not isinstance(audio, (bytearray)):
                raise TypeError("""[WhisperInference] :: For RT inference, the <audio> parameter 
                                must be a bytearray""")
            # now it is needed to normalised to [-1, 1]
            # x_norm = x_int16 / 32768.0 (float32 if possible for precission)
            audio_int16 = np.frombuffer(bytes(audio), dtype=np.int16)
            audio_float32 = audio_int16.astype(np.float32) / 32768.0

            segments, info = self.model.transcribe(audio_float32, beam_size=beam_size, language="es", 
                                                   vad_filter=True, no_repeat_ngram_size=2)

        else:
            if not isinstance(audio, (str)):
                raise TypeError("""[WhisperInference] :: For I/O inference, the <audio> parameter 
                                must be a string""")
            segments, info = self.model.transcribe(audio, beam_size=beam_size, language="es", 
                                                   vad_filter=False, no_repeat_ngram_size=2)

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
        print(f"{segment.text} ", flush=True)


    @staticmethod
    def _normalize_metric_text(text: str) -> str:
        """
        Normalize text before metric computation.
        The '#' character is removed and ignored.
        """
        text = text.replace("#", "")
        return " ".join(text.split()).strip().lower()


    @staticmethod
    def _levenshtein_distance(a: list[str], b: list[str]) -> int:
        """
        Compute Levenshtein distance between two token sequences.
        """
        if not a:
            return len(b)
        if not b:
            return len(a)

        prev_row = list(range(len(b) + 1))
        for i, a_item in enumerate(a, start=1):
            curr_row = [i]
            for j, b_item in enumerate(b, start=1):
                cost = 0 if a_item == b_item else 1
                curr_row.append(
                    min(
                        prev_row[j] + 1,
                        curr_row[j - 1] + 1,
                        prev_row[j - 1] + cost,
                    )
                )
            prev_row = curr_row

        return prev_row[-1]


    def compute_wer_counts(self, transcript: str, reference_text: str) -> tuple[int, int]:
        """
        Return absolute word errors and number of reference words.
        """
        normalized_reference = self._normalize_metric_text(reference_text)
        normalized_transcript = self._normalize_metric_text(transcript)

        ref_words = normalized_reference.split()
        hyp_words = normalized_transcript.split()
        ref_count = len(ref_words)
        if ref_count == 0:
            return 0, 0

        errors = self._levenshtein_distance(ref_words, hyp_words)
        return errors, ref_count


    def compute_cer_counts(self, transcript: str, reference_text: str) -> tuple[int, int]:
        """
        Return absolute char errors and number of reference chars.
        """
        normalized_reference = self._normalize_metric_text(reference_text)
        normalized_transcript = self._normalize_metric_text(transcript)

        ref_chars = list(normalized_reference)
        hyp_chars = list(normalized_transcript)
        ref_count = len(ref_chars)
        if ref_count == 0:
            return 0, 0

        errors = self._levenshtein_distance(ref_chars, hyp_chars)
        return errors, ref_count


    def compute_wer(self, transcript: str, reference_text: str) -> float:
        """
        Compute Word Error Rate (WER) for a transcript/reference pair.
        """
        errors, ref_count = self.compute_wer_counts(transcript, reference_text)
        if ref_count == 0:
            return 0.0
        return errors / ref_count


    def compute_cer(self, transcript: str, reference_text: str) -> float:
        """
        Compute Character Error Rate (CER) for a transcript/reference pair.
        """
        errors, ref_count = self.compute_cer_counts(transcript, reference_text)
        if ref_count == 0:
            return 0.0
        return errors / ref_count
