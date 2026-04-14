"""
This script contains the code for measuring the metrics between the original text and the
transcribed audio.

The test metrics are being calculated using .flac audios and their corresponding .md files
"""

import os
import yaml
import logging
import argparse
from pathlib import Path

from src.whisper.Whisper import WhisperInference

logger = logging.getLogger(__name__)

def get_args():
    parse = argparse.ArgumentParser()

    parse.add_argument('--data', type=str, required=True,
                       default='data/notasClinicas/data.yaml',
                       help='Route to where the data.yaml file is located')
    
    # For the model arguments, they are not required. If not introduced, the program
    # will run with a default configuration
    parse.add_argument('--model-size', type=str, required=False, default='medium',
                       help='Size of the Whisper Model')
    
    parse.add_argument('--precision', type=str, required=False, default='float16',
                       help='Precision of the Whisper Model')

    return parse.parse_args()


def load_data_config(data: str):
    """
    Retrieves the data config from the the data.yaml file of the dataset

    :param data: str -> route of the data.yaml file
    :return config: Dict -> a dictionary with the routes for the audios and texts
    """
    with open(data, 'r', encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        return config


def run_inference(model: WhisperInference, audio_file: str) -> str:
    """
    Run inference for one audio and return the complete transcript as a single string.
    """
    segments, _ = model.inference(audio=audio_file, beam_size=1)
    transcript_parts = []

    for segment in segments:
        text = (segment.text or "").strip()
        if text:
            transcript_parts.append(text)

    return " ".join(transcript_parts).strip()


def main():
    logging.basicConfig(level=logging.INFO)
    logging.info("[metrics] :: Init of the script\n")
    args = get_args()

    data = args.data
    MODEL_SIZE = args.model_size
    MODEL_PRECISION = args.precision

    try:
        # obtain the dataset data configuration (routes to the data)
        config = load_data_config(data)

        audios_route = config['audio_files']
        texts_route = config['texts']

        # remove data.yaml so it is possible to access to the *_route
        rel_route = os.path.dirname(data)
        audios_route = os.path.normpath(os.path.join(rel_route, audios_route))
        texts_route = os.path.normpath(os.path.join(rel_route, texts_route))

        # load model only once for the full dataset
        rt_batch_duration = 1  # not used for file-based inference
        rt_inference = False
        model = WhisperInference(
            model_size=MODEL_SIZE,
            precision=MODEL_PRECISION,
            batch_duration=rt_batch_duration,
            rt=rt_inference,
        )

        logging.info("[metrics] :: Loading Whisper model (size=%s, precision=%s)", 
                     MODEL_SIZE, MODEL_PRECISION)
        model.load_model()

        audios = sorted(os.listdir(audios_route))
        processed = 0

        # for every audio in the list, look up the text file and run inference
        for audio in audios:
            audio_file = os.path.normpath(os.path.join(audios_route, audio))
            if not os.path.isfile(audio_file):
                continue

            audio_stem = Path(audio).stem # remove termination for file
            text_file = os.path.normpath(os.path.join(texts_route, f"{audio_stem}.txt"))
            if not os.path.isfile(text_file):
                logging.warning(
                    "[metrics] :: Missing reference text for %s. File expected at %s",
                    audio,
                    text_file,
                )
                continue

            try:
                transcript = run_inference(model, audio_file)
            except Exception:
                logger.exception("[metrics] :: Inference failed for file %s", audio_file)
                continue

            processed += 1
            logging.info(
                "[metrics] :: Inference OK for %s (%d chars)",
                audio,
                len(transcript),
            )

            with open(f'{text_file}.transcript', 'w') as f:
                f.write(transcript)

        logging.info("[metrics] :: Total files processed with inference: %d", processed)
    except Exception:
        logger.exception("[metrics] :: Fatal error while running metrics inference")
        raise SystemExit(1)


    logging.info("[metrics] :: Metrics calculation has finished\n")
    

if __name__ == '__main__':
    main()