import pyrallis

from transformers import logging

from src.configs.config_classes import InferenceConfig
from src.inference.models import TextLangPredictor


@pyrallis.wrap()
def parse(cfg: InferenceConfig):
    logging.set_verbosity_error()
    lang_predictor = TextLangPredictor(cfg)

    while True:
        text = input("Enter a text to parse (Ctrl-C to exit):\n").strip()
        print("Parsing results:")
        for lang, sent in lang_predictor.parse_text(text).items():
            print(f"{lang}: {sent}")


if __name__ == "__main__":
    parse()
