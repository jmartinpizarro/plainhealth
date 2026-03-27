"""
This file contains a brief study of the inference for the whisper models and its variations.

For maximising the usefulness of this study, all the experiments will be runned on a CUDA 11.x
environment.
"""

from typing import List
from time import perf_counter

from faster_whisper import WhisperModel

MODEL_SIZES: List[str] = ['tiny', 'base'] 
PRECISION: str = 'int8_float16' # this allows us to run on a GPU with int8.
# if we want it to run it on a cpu, please use 'int8'

def main():
    print("[Whisper] :: Starting script")

    for MODEL_SIZE in MODEL_SIZES:
        print(f"[Whisper] :: Starting testing for Whisper model {MODEL_SIZE}...")
        model = WhisperModel(MODEL_SIZE, device="cuda", compute_type=PRECISION)
        print("[Whisper] :: Model loaded")

        # create a descriptor for logs
        with open(f"output/{MODEL_SIZE}_transcript.txt", "w") as f:
            segments, info = model.transcribe("data/sample.mp3", beam_size=1)

            print("[Whisper] :: Detected language '%s' with probability %f" % (info.language, info.language_probability))

            segment_iterator = iter(segments)
            segment_index: int = 0

            while True:
                start_time = perf_counter()
                try:
                    segment = next(segment_iterator)
                except StopIteration:
                    break
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
                segment_index += 1

        print(f"[Whisper] :: Testing has ended for Whisper model {MODEL_SIZE}\n\n")


if __name__ == '__main__':
    main()