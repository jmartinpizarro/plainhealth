"""
This file contains a brief study of the inference for the whisper models and its variations.

For maximising the usefulness of this study, all the experiments will be runned on a CUDA 11.x
environment.
"""

import time
import argparse
import os
from dotenv import load_dotenv
from typing import List
from time import perf_counter

from src.whisper.Whisper import WhisperInference

import alsaaudio, audioop
from huggingface_hub import login

MODEL_SIZES: List[str] = ['tiny', 'medium', 'base'] 
PRECISION: str = 'int8_float16' # this allows us to run on a GPU with int8.
# if we want it to run it on a cpu, please use 'int8'
RT_BATCH_DURATION: int = 1 # seconds

def get_args():
    parser = argparse.ArgumentParser()

    # the program must use RT inference (the user is speaking into the microphone)
    parser.add_argument("--rt", action="store_true")

    return parser.parse_args()


def main():
    print("[Whisper] :: Starting script")

    load_dotenv()

    HF_TOKEN = os.getenv('HF_TOKEN')
    login(token=HF_TOKEN)

    args = get_args()
    mode = args.rt # True for RT inference, False for loading a .mp3

    if not mode: # not rt
        # Currently, this script only runs some shitty ass benchmarks i just invented for testing
        # the accuracy between different size models.
        for MODEL_SIZE in MODEL_SIZES:
            print(f"[Whisper] :: Starting testing for Whisper model {MODEL_SIZE}...")

            model = WhisperInference(model_size=MODEL_SIZE, precision=PRECISION, batch_duration=RT_BATCH_DURATION, rt=mode)
            try:
                model.load_model() # download and load into memory
            except Exception as e:
                print(f"[main] :: An error has ocurred when loading the Whisper model\n{e}\n")

            time.sleep(0.5)

            # create a descriptor for logs
            with open(f"output/{MODEL_SIZE}_transcript.txt", "w") as f:
                segments, info = model.inference("data/sample.mp3", beam_size=1)

                print("[Whisper] :: Detected language '%s' with probability %f" % (info.language, info.language_probability))

                segment_iterator = iter(segments)

                while True:
                    try:
                        model.write_logs(segment_iterator, f, segment_index)
                        segment_index += 1
                    except StopIteration: # no more segments to process for this batch
                        f.flush()
                        start = perf_counter() # restart the init timer
                        historic = bytearray() # restart the historic data to an empty array
                        break

            print(f"[Whisper] :: Testing has ended for Whisper model {MODEL_SIZE}\n\n")

        exit(0)
        
    # if here, then rt inference

    inp = alsaaudio.PCM(alsaaudio.PCM_CAPTURE,alsaaudio.PCM_NONBLOCK, channels=1, rate=16000, format=alsaaudio.PCM_FORMAT_S16_LE, periods=160)

    rt_model_size = MODEL_SIZES[1]

    model = WhisperInference(model_size=rt_model_size, precision=PRECISION, batch_duration=RT_BATCH_DURATION, rt=mode)
    
    try:
        model.load_model() # download and load into memory

    except Exception as e:
        print(f"[main] :: An error has ocurred when loading the Whisper model\n{e}\n")
    time.sleep(0.5)

    os.makedirs("output/rt", exist_ok=True)
    output_path = f"output/rt/{rt_model_size}_transcript.txt"

    start = perf_counter()
    historic = bytearray()
    segment_index = 0

    with open(output_path, "w") as f:
        while True:
            try:
                # Read data from device
                l,data = inp.read()
                if l: # if any frame obtained
                    # data is in hexadecimal. In order to give the audio to the
                    # model several conditions must occur:
                    # audio mono (if stereo then convert it), PCM normalised ([-1, 1]), 16kHz
                    # window size of 2 seconds
                    # send the data from memory, never saved in I/O

                    # data is currently in [-32768, 32767]
                    historic.extend(data)
                    now = perf_counter()
                    if now - start >= RT_BATCH_DURATION and len(historic) > 0:
                        # do the inference
                        segments, _ = model.inference(historic, beam_size=1)

                        segment_iterator = iter(segments)

                        while True:
                            try:
                                model.write_logs(segment_iterator, f, segment_index)
                                segment_index += 1
                            except StopIteration: # no more segments to process for this batch
                                f.flush()
                                start = perf_counter() # restart the init timer
                                historic = bytearray() # restart the historic data to an empty array
                                break

                time.sleep(.001)
            except KeyboardInterrupt:
                print("[Whisper] :: The RT inference was manually terminated\n")
                exit(0)
            except audioop.error as e:
                print(f"[Whisper] :: An error  when processing the audio ocurred.\n{e}.\n")
                exit(-1)


if __name__ == '__main__':
    main()