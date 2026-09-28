#!/usr/bin/env python3
"""Convert hishab/titu_stt_bn_fastconformer (a raw NeMo .nemo EncDecCTCModelBPE
checkpoint) into the staging layout audio.cpp's fastconformer_ctc family
expects: model.safetensors + config.json + tokenizer.model.

See conversion.md at the repo root for the architecture analysis this is
based on and the corresponding C++ family
(src/models/fastconformer_ctc/{assets,weights,runtime,session}.cpp).
"""

import argparse
import json
import tarfile
import tempfile
from pathlib import Path

import torch
import yaml
from safetensors.torch import save_file


def find(names, suffix):
    matches = [n for n in names if n.lstrip("./").endswith(suffix)]
    if not matches:
        raise KeyError(f"no archive member ends with {suffix!r}")
    return matches[0]


def load_nemo(path: Path):
    with tarfile.open(path, mode="r:*") as archive:
        names = archive.getnames()
        config = yaml.safe_load(archive.extractfile(find(names, "model_config.yaml")).read())
        with tempfile.NamedTemporaryFile(suffix=".ckpt", delete=False) as ckpt:
            handle = archive.extractfile(find(names, "model_weights.ckpt"))
            while block := handle.read(16 * 1024 * 1024):
                ckpt.write(block)
        state = torch.load(ckpt.name, map_location="cpu", weights_only=False)
        state = state.get("state_dict", state)

        tokenizer_path = config["tokenizer"]["model_path"].split(":", 1)[1]
        tokenizer_member = find(names, Path(tokenizer_path).name)
        tokenizer_bytes = archive.extractfile(tokenizer_member).read()
    return config, state, tokenizer_bytes


def stage_tensors(state: dict, xscale: float) -> dict:
    # Encoder and CTC-head tensor names are used verbatim -- they already
    # match the raw NeMo names src/models/fastconformer_ctc/weights.cpp
    # reads directly (encoder.pre_encode.*, encoder.layers.N.*,
    # decoder.decoder_layers.0.*). Only the preprocessor tensors are
    # renamed/reshaped, matching tests/nemotron_3_diar/convert_gguf.py's
    # existing convention for the same NeMo mel-frontend tensors.
    #
    # NeMo's RelPositionalEncoding multiplies the subsampling output by
    # xscale = sqrt(d_model) before the encoder stack whenever the encoder
    # config sets `xscaling: true` (true for this checkpoint). The shared
    # audio.cpp FastConformer modules (conformer_modules.cpp,
    # relative_attention.cpp) don't apply this themselves -- src/community_
    # models/parakeet_tdt/weights.cpp instead folds it permanently into the
    # subsampling projection's weight+bias at load time (see its
    # `scaled_f32` calls), since a linear layer's output scales linearly
    # with a constant scale folded into its weights. Do the same here in
    # Python since we already have the raw tensor at conversion time.
    out = {}
    for name, value in state.items():
        if not isinstance(value, torch.Tensor):
            continue
        if name == "preprocessor.featurizer.fb":
            out["preprocessor.fb"] = value.squeeze(0).detach().cpu().contiguous()
        elif name == "preprocessor.featurizer.window":
            out["preprocessor.window"] = value.detach().cpu().contiguous()
        elif name in ("encoder.pre_encode.out.weight", "encoder.pre_encode.out.bias"):
            out[name] = (value * xscale).detach().cpu().contiguous()
        else:
            out[name] = value.detach().cpu().contiguous()
    return out


def build_config_json(nemo_cfg: dict) -> dict:
    enc = nemo_cfg["encoder"]
    pre = nemo_cfg["preprocessor"]
    dec = nemo_cfg["decoder"]
    # NeMo's decoder.num_classes counts vocabulary tokens only; CTC appends
    # one blank class on top, so the head's real output width (and the
    # decoder.decoder_layers.0 tensor shape) is num_classes + 1.
    vocab_size = dec["num_classes"] + 1
    return {
        "model_type": "fastconformer_ctc",
        "vocab_size": vocab_size,
        "blank_token_id": vocab_size - 1,
        "encoder_config": {
            "feat_in": enc["feat_in"],
            "hidden_size": enc["d_model"],
            "intermediate_size": int(round(enc.get("ff_expansion_factor", 4) * enc["d_model"])),
            "num_hidden_layers": enc["n_layers"],
            "num_attention_heads": enc["n_heads"],
            "conv_kernel_size": enc["conv_kernel_size"],
            "subsampling_factor": enc["subsampling_factor"],
            "subsampling_conv_channels": enc.get("subsampling_conv_channels", 256),
            "subsampling_conv_kernel_size": 3,
            "subsampling_conv_stride": 2,
        },
        "processor_config": {
            "feature_extractor": {
                "sampling_rate": pre["sample_rate"],
                "feature_size": pre["features"],
                "n_fft": pre["n_fft"],
                "win_length": int(round(pre["window_size"] * pre["sample_rate"])),
                "hop_length": int(round(pre["window_stride"] * pre["sample_rate"])),
                "preemphasis": pre.get("preemph", 0.97),
            }
        },
    }


def convert(nemo_path: Path, output_dir: Path) -> None:
    config, state, tokenizer_bytes = load_nemo(nemo_path)
    if config.get("encoder", {}).get("self_attention_model") != "rel_pos":
        raise ValueError("fastconformer_ctc requires rel_pos self-attention")
    if config.get("encoder", {}).get("subsampling") != "dw_striding":
        raise ValueError("fastconformer_ctc requires dw_striding subsampling")
    if config.get("decoder", {}).get("_target_") != "nemo.collections.asr.modules.ConvASRDecoder":
        raise ValueError("fastconformer_ctc requires a plain ConvASRDecoder (CTC) head")
    if not config["encoder"].get("xscaling", True):
        raise ValueError("this converter always folds xscaling=true into the subsampling projection; "
                          "the checkpoint declares xscaling=false, so stage_tensors must be changed to skip it")

    output_dir.mkdir(parents=True, exist_ok=True)
    xscale = config["encoder"]["d_model"] ** 0.5
    tensors = stage_tensors(state, xscale)
    save_file(tensors, output_dir / "model.safetensors")
    (output_dir / "config.json").write_text(json.dumps(build_config_json(config), indent=2) + "\n")
    (output_dir / "tokenizer.model").write_bytes(tokenizer_bytes)
    print(f"staged {len(tensors)} tensors -> {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("nemo_path", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    convert(args.nemo_path, args.output_dir)
