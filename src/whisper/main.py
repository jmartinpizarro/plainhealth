"""
This file contains a brief study of the inference for the whisper models and its variations.

For maximising the usefulness of this study, all the experiments will be runned on a CUDA 11.x
environment.
"""

import time
import argparse
import os
from typing import List
from time import perf_counter

from src.whisper.Whisper import WhisperInference

import numpy as np
import alsaaudio, audioop
from faster_whisper import WhisperModel

MODEL_SIZES: List[str] = ['tiny', 'medium', 'base'] 
PRECISION: str = 'int8_float16' # this allows us to run on a GPU with int8.
# if we want it to run it on a cpu, please use 'int8'
RT_BATCH_DURATION: int = 2 # seconds

def get_args():
    parser = argparse.ArgumentParser()

    # the program must use RT inference (the user is speaking into the microphone)
    parser.add_argument("--rt", action="store_true")

    return parser.parse_args()


def main():
    print("[Whisper] :: Starting script")

    args = get_args()
    mode = args.rt # True for RT inference, False for loading a .mp3

    model = WhisperInference(model_size=MODEL_SIZES[1], precision=PRECISION, batch_duration=RT_BATCH_DURATION, rt=mode)
    try:
        model.load_model() # download and load into memory
    except Exception as e:
        print(f"[main] :: An error has ocurred when loading the Whisper model\n{e}\n")

    time.sleep(0.5)

    if not mode: # not rt
        # Currently, this script only runs some shitty ass benchmarks i just invented for testing
        # the accuracy between different size models.
        for MODEL_SIZE in MODEL_SIZES:
            print(f"[Whisper] :: Starting testing for Whisper model {MODEL_SIZE}...")


            # create a descriptor for logs
            with open(f"output/{MODEL_SIZE}_transcript.txt", "w") as f:
                segments, info = model.inference("data/sample.mp3", beam_size=1)

                print("[Whisper] :: Detected language '%s' with probability %f" % (info.language, info.language_probability))

                segment_iterator = iter(segments)
                segment_index: int = 0

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

    # may god bless Stackoverflow ;)

    # Open the device in nonblocking capture mode. The last argument could
    # just as well have been zero for blocking mode. Then we could have
    # left out the sleep call in the bottom of the loop
    inp = alsaaudio.PCM(alsaaudio.PCM_CAPTURE,alsaaudio.PCM_NONBLOCK, channels=1, rate=16000, format=alsaaudio.PCM_FORMAT_S16_LE, periods=160)

    # The period size controls the internal number of frames per period.
    # The significance of this parameter is documented in the ALSA api.
    # For our purposes, it is suficcient to know that reads from the device
    # will return this many frames. Each frame being 2 bytes long.
    # This means that the reads below will return either 320 bytes of data
    # or 0 bytes of data. The latter is possible because we are in nonblocking
    # mode.

    rt_model_size = MODEL_SIZES[1]

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
                        segments, info = model.inference(historic, beam_size=1)
                        # TODO: look if i can predefine a language
                        print("[Whisper] :: Detected language '%s' with probability %f" % (info.language, info.language_probability))

                        segment_iterator = iter(segments)
                        segment_index: int = 0

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