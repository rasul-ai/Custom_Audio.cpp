"""
Standalone smoke test for the NeMo 3.5 ASR model: loads the checkpoint and
transcribes a single wav file passed on the command line.

Usage:

python test_nemo_asr.py /home/admin_noeticx/projects/audio.cpp/assets/resources/speech_16k.wav

Reads NEMO_ASR_MODEL_PATH from backend/.env by default (same checkpoint the
app loads), so no flags are required if .env is already configured.
"""

import argparse
import logging
import os
import sys
import time

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("test_nemo_asr")


def pin_prompt_language(model, target_lang: str) -> None:
    """
    Nemotron 3.5 ASR is prompt-conditioned: a plain transcribe() call crashes
    with "ValueError: Unknown prompt key: 'None'" because the dataloader
    expects a per-utterance language prompt index that transcribe() never
    populates (https://github.com/NVIDIA-NeMo/Speech/issues/15820, open as of
    nemo_toolkit 3.0.0). Patch the dataset lookup to always return the fixed
    prompt index for `target_lang` instead.
    """
    from nemo.collections.asr.data.audio_to_text_lhotse_prompt_index import (
        LhotseSpeechToTextBpeDatasetWithPromptIndex,
    )
    from omegaconf import open_dict

    prompt_dict = model.cfg.model_defaults.get("prompt_dictionary", {})
    if target_lang not in prompt_dict:
        logger.warning(
            "target_lang=%r not in prompt_dictionary (%s...); falling back to 'auto'",
            target_lang, sorted(prompt_dict)[:10],
        )
        target_lang = "auto"
    prompt_id = prompt_dict[target_lang]
    LhotseSpeechToTextBpeDatasetWithPromptIndex._get_prompt_index_for_cut = (
        lambda self, cut: prompt_id
    )

    decoding_cfg = model.cfg.decoding
    with open_dict(decoding_cfg):
        decoding_cfg.strip_lang_tags = True
    model.change_decoding_strategy(decoding_cfg, verbose=False)

    logger.info("Prompt language pinned to '%s' (index %s)", target_lang, prompt_id)


def load_model(model_path: str, device: str):
    import nemo.collections.asr as nemo_asr
    from nemo.utils import logging as nemo_logging

    logger.info("Loading NeMo ASR model from '%s' on %s ...", model_path, device)
    prev_level = nemo_logging.get_verbosity()
    nemo_logging.setLevel(logging.ERROR)
    try:
        model = nemo_asr.models.ASRModel.restore_from(model_path, map_location=device)
    finally:
        nemo_logging.setLevel(prev_level)
    model.eval()
    return model


def transcribe(model, wav_path: str, lang: str) -> str:
    import torch

    if hasattr(model, "cfg") and "prompt_dictionary" in model.cfg.get("model_defaults", {}):
        pin_prompt_language(model, lang)

    with torch.no_grad():
        transcriptions = model.transcribe([wav_path], verbose=False)

    if isinstance(transcriptions, tuple):
        transcriptions = transcriptions[0]
    hyp = transcriptions[0]
    return hyp.text if hasattr(hyp, "text") else str(hyp)


def main():
    parser = argparse.ArgumentParser(description="Test the NeMo 3.5 ASR model on a wav file")
    parser.add_argument("wav_path", help="Path to the .wav file to transcribe")
    parser.add_argument(
        "--model-path",
        default="/home/admin_noeticx/llm_models/nemo_asr/nemotron-3.5-asr-streaming-0.6b.nemo",
        help="Path to the .nemo checkpoint (default: NEMO_ASR_MODEL_PATH from .env)",
    )
    parser.add_argument(
        "--lang",
        default=os.getenv("NEMO_ASR_LANG", "en-US"),
        help="BCP-47 target language tag pinned via the model's prompt dictionary (default: en-US)",
    )
    parser.add_argument(
        "--device",
        default=os.getenv("DEVICE"),
        help="cuda or cpu (default: cuda if available, else cpu)",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.wav_path):
        parser.error(f"wav file not found: {args.wav_path}")
    if not args.model_path:
        parser.error("No model path given and NEMO_ASR_MODEL_PATH is not set in .env")
    if not os.path.isfile(args.model_path):
        parser.error(f".nemo checkpoint not found: {args.model_path}")

    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(args.model_path, device)

    start = time.time()
    text = transcribe(model, args.wav_path, args.lang)
    elapsed = time.time() - start

    print("\n" + "=" * 60)
    print(f"File:       {args.wav_path}")
    print(f"Language:   {args.lang}")
    print(f"Time:       {elapsed:.2f}s")
    print("-" * 60)
    print(f"Transcript: {text}")
    print("=" * 60)


if __name__ == "__main__":
    sys.exit(main() or 0)
