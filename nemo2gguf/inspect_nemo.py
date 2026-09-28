# inspect_nemo.py
import sys, tarfile, yaml, torch

path = sys.argv[1]
with tarfile.open(path, mode="r:*") as archive:
    names = archive.getnames()
    print("archive members:", names)

    def find(suffix):
        matches = [n for n in names if n.lstrip("./").endswith(suffix)]
        if not matches:
            raise KeyError(f"no archive member ends with {suffix!r}")
        return matches[0]

    config = yaml.safe_load(archive.extractfile(find("model_config.yaml")).read())
    print("target class:", config.get("target"))
    print("encoder:", {k: config["encoder"][k] for k in (
        "self_attention_model", "subsampling", "subsampling_factor",
        "subsampling_conv_channels", "n_layers", "d_model", "n_heads",
        "conv_kernel_size", "causal_downsampling", "xscaling", "ff_expansion_factor",
    ) if k in config["encoder"]})
    print("decoder:", config.get("decoder"))
    print("has aux ctc head:", "aux_ctc" in config)
    print("tokenizer:", config.get("tokenizer"))

    weights_member = archive.extractfile(find("model_weights.ckpt"))
    with open("/tmp/nemo_weights.ckpt", "wb") as fh:
        fh.write(weights_member.read())

state = torch.load("/tmp/nemo_weights.ckpt", map_location="cpu", weights_only=False)
state = state.get("state_dict", state)
print("tensor count:", len(state))
for name in sorted(state)[:40]:
    print(" ", name, tuple(state[name].shape) if hasattr(state[name], "shape") else state[name])
