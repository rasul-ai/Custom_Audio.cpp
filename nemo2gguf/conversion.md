# Porting a NeMo ASR model to audio.cpp

A general guide for taking a NVIDIA NeMo `.nemo` checkpoint, converting it
to a self-contained audio.cpp GGUF, and — when no existing family's C++
already matches the architecture — adding the new family's code to this
repo so `audiocpp_cli`/`audiocpp_server` can run it.

Worked example throughout: **`hishab/titu_stt_bn_fastconformer`**, a
standard NeMo FastConformer-CTC-BPE model for Bangla, ported here as the
`fastconformer_ctc` family (`include/engine/models/fastconformer_ctc/`,
`src/models/fastconformer_ctc/`). Verified working transcript on the
model's own test clip:

```
আজ সরকারি ছুটির দিন দেশের সব শিক্ষা প্রতিষ্ঠান সহ সরকারি আধা সরকারি
স্বায়ত্তশাসিত প্রতিষ্ঠান ও ভবনে জাতীয় পতাকা অর্ধনমিত ও কালো পতাকা
উত্তোলন করা হয়েছে
```

The scripts in this folder (`inspect_nemo.py`, `convert_titu_to_safetensors.py`)
are the concrete, working implementation of every generic step below.

---

## The pipeline, end to end

```
.nemo archive  --[Step 1: inspect]-->  architecture facts
               --[Step 2: match]-->    existing family reusable, or new family needed
               --[Step 3: build C++]-->  new family compiled into audiocpp_gguf/audiocpp_cli
               --[Step 4: convert]-->  model.safetensors + config.json + tokenizer
               --[Step 5: package]-->  audiocpp_gguf --family <family> ... -> model.gguf
               --[Step 6: run]-->      audiocpp_cli --family <family> --model model.gguf ...
```

GGUF here is audio.cpp's own tensor container, not a universal adapter
(`docs/gguf.md`): the tensor names and shapes inside it must match a
specific family's C++ loader exactly. There is no generic "load any NeMo
model" path — every family in this repo (`gigaam_asr`, `parakeet_tdt`,
`canary_asr`, `nemotron_asr`, `citrinet_asr`, ...) has hand-written C++
that expects a specific tensor layout, and a matching Python converter.
Porting a new checkpoint means either matching one of those exactly, or
adding a new family.

---

## Step 1 — Inspect the `.nemo` checkpoint

A `.nemo` file is a plain tar archive containing `model_config.yaml` (a
NeMo/OmegaConf YAML config) and `model_weights.ckpt` (a tarred PyTorch
`state_dict`, not safetensors). Extract and print both:

```bash
source .venv/bin/activate   # or prefix commands with .venv/bin/
pip install torch pyyaml safetensors sentencepiece huggingface_hub

hf download <org>/<model> <model>.nemo --local-dir models/<model>
python3 inspect_nemo.py models/<model>/<model>.nemo
```

`inspect_nemo.py` (repo root) handles the one archive quirk that trips up
a naive `tarfile.extractfile("model_config.yaml")`: members are stored with
a `./` prefix, so it matches on suffix instead of exact name. It prints:

- `target`: the NeMo model class (`nemo.collections.asr.models.<...>`) —
  tells you CTC vs RNNT vs hybrid vs encoder-decoder.
- `encoder`: the fields that actually determine architecture compatibility
  — `self_attention_model` (`rel_pos` / `rope`/`rotary` / ...),
  `subsampling` (`dw_striding` / `conv1d` / ...), `subsampling_factor`,
  `n_layers`, `d_model`, `n_heads`, `conv_kernel_size`, `xscaling`,
  `ff_expansion_factor`.
- `decoder`: its `_target_` (`ConvASRDecoder` = plain CTC head;
  `RNNTDecoder`/`RNNTJoint` = transducer; both present = hybrid) and
  `num_classes`.
- `tokenizer`: type and the archive-internal path to the tokenizer file(s)
  (usually random-hash-prefixed inside the tar — resolve the real name via
  `config["tokenizer"]["model_path"]`, stripping the `nemo:` prefix, which
  just means "look inside this archive").
- The first 40 tensor names and shapes from the real `state_dict` — this is
  ground truth for exactly which tensors exist and what shape they are.
  Don't trust the YAML config's stated dimensions over the actual tensor
  shapes when they disagree (see the `vocab_size` gotcha in Step 4).

For the worked example, this printed:

```
target:                         nemo.collections.asr.models.ctc_bpe_models.EncDecCTCModelBPE
encoder.self_attention_model:   rel_pos
encoder.subsampling:            dw_striding, factor 8
encoder.n_layers / d_model:     18 / 512
encoder.n_heads:                8
encoder.conv_kernel_size:       9
encoder.ff_expansion_factor:    4
encoder.xscaling:               true
decoder._target_:               nemo.collections.asr.modules.ConvASRDecoder
decoder.num_classes:            1024
tokenizer:                      SentencePiece BPE
```

## Step 2 — Match against an existing family, or scope a new one

Check the encoder/decoder facts from Step 1 against what's already
implemented. As of this writing:

| Family | Encoder | Decoder | Notes |
|---|---|---|---|
| `gigaam_asr` | Conformer, `self_attention_model: rotary`, `subsampling: conv1d`, factor 4 | CTC or RNNT | Loader in `src/models/gigaam_asr/assets.cpp` **hard-rejects** any other encoder combination |
| `parakeet_tdt` | FastConformer, `rel_pos`, `dw_striding`, factor 8 | TDT (transducer) | C++ graph hardcodes no bias on FFN/attention/pointwise-conv linears — real checkpoints with those biases need the fix described in Step 3 or a separate family |
| `canary_asr` | FastConformer, `rel_pos`, `dw_striding`, factor 8 | Transformer decoder (seq2seq, translation) | Already bias-aware; its encoder-side modules are the best starting point for a new FastConformer family (see below) |
| `nemotron_asr` | FastConformer-like, prompt-conditioned streaming | Multi-task hybrid | Not a plain CTC/RNNT path |
| `citrinet_asr` | 1D depthwise-separable convs, no attention | CTC | Different architecture family entirely (Citrinet, not Conformer) |

If your checkpoint's `encoder`/`decoder` facts match an existing family's
row exactly, skip to Step 4 — you don't need new C++, just a converter
targeting that family's expected tensor names (read its `weights.cpp` to
get them) and `--family <existing_family>` in Step 5.

If nothing matches — as with `hishab/titu_stt_bn_fastconformer`
(`rel_pos` + `dw_striding` + plain `ConvASRDecoder`, which none of the
above rows cover) — you need a new family. Continue to Step 3.

**Before writing a full custom encoder, search for reusable generic
modules.** The single most valuable discovery in the `fastconformer_ctc`
port was that `src/models/canary_asr/` already builds its FastConformer
encoder from fully generic, already-bias-aware framework modules —
`modules::DepthwiseConvSubsamplingModule` and
`modules::RelativeConformerBlockModule`
(`include/engine/framework/modules/conformer_modules.h`) — rather than a
bespoke hand-rolled ggml graph like `parakeet_tdt`'s. If your encoder is
`rel_pos` + `dw_striding`, these same modules almost certainly cover it;
only the CTC/RNNT/hybrid **decoder head** actually needs new code. Grep for
other consumers of `RelativeSelfAttentionModule`
(`src/framework/modules/attention/relative_attention.h`) —
`canary_asr`, `cohere_asr`, `zipformer_modules.cpp`, `index_tts2/gpt.cpp`,
`kroko_asr/zipformer.cpp` all reuse shared attention modules rather than
duplicating them; do the same.

## Step 3 — Build the new C++ family (skip if Step 2 matched an existing one)

### Layout

```
include/engine/models/<family>/model.h
src/models/<family>/assets.cpp     # parse config.json, load tokenizer, build frontend
src/models/<family>/weights.cpp    # load tensors from the safetensors/GGUF source
src/models/<family>/runtime.cpp    # build the ggml graph, run inference, decode
src/models/<family>/session.cpp    # wire it into runtime::IOfflineVoiceTaskSession
model_specs/<family>.json          # package spec (schema v1)
```

This mirrors every existing family (`gigaam_asr`, `canary_asr`,
`nemotron_3_diar`, ...) — follow that convention rather than inventing a
new one. Read `CONTRIBUTING.md`'s "New Model PRs" section first.

### `model.h`

Define: a config struct with the encoder/decoder hyperparameters you'll
read from `config.json`; an `Assets` struct (`ResourceBundle`, tensor
source, config, tokenizer pieces, frontend); a `Weights` struct (backend
weight store + whatever module weight structs your encoder/decoder need —
e.g. `modules::DepthwiseConvSubsamplingWeights` +
`std::vector<modules::RelativeConformerBlockWeights>` + your own decode
head weights); a `Runtime` class with a `transcribe(...)` method; and
`make_<family>_loader()`. See
`include/engine/models/fastconformer_ctc/model.h` for a complete, minimal
example (CTC-only, offline-only — no streaming state).

### `assets.cpp`

```cpp
out->resources = model_spec::load_resource_bundle_for_family(path, "<family>");
out->source = out->resources.open_tensor_source("weights");
const auto config = out->resources.parse_json("config");
// read + validate encoder/decoder hyperparameters from config, throwing on
// any architecture assumption you rely on (self_attention_model,
// subsampling type/factor, etc.) so a mismatched checkpoint fails loudly
// at load time instead of producing silently wrong output.
out->resources.add_model_file("tokenizer", "tokenizer.model");
out->pieces = tokenizers::load_sentencepiece_model(out->resources.require_file("tokenizer"));
// build the frontend -- see "Frontend" below.
```

### `weights.cpp`

Load tensors under **the checkpoint's raw NeMo names directly** — don't
invent a renamed convention unless you have a reason to. This is the
single biggest simplification available: if you reuse generic modules like
`canary_asr` does, their weight structs' loaders already expect names like
`encoder.pre_encode.conv.0`, `encoder.layers.N.self_attn.linear_q`,
`encoder.layers.N.conv.batch_norm`, etc. — exactly what's already in the
`.nemo` checkpoint. Keeping names 1:1 means your Python conversion script
in Step 4 does almost no renaming, which is both less code and less risk.

Use `engine::modules::binding::*` helpers
(`include/engine/framework/modules/weight_binding.h`) —
`conv2d_from_source`, `depthwise_conv1d_from_source`,
`batch_norm_eval_from_source`, `linear_from_named_source`,
`norm_from_named_source` — rather than hand-writing tensor loads; they're
what every existing family uses and they handle bias-presence, shape
checks, and storage-type conversion consistently.

For a decoder head that's a Conv1d with kernel size 1 (very common for
NeMo's `ConvASRDecoder`), load it as a plain `Linear` — a kernel-1 Conv1d
weight `[out, in, 1]` is mathematically identical to a Linear weight
`[out, in]`:

```cpp
out->ctc_head = {
    store.load_tensor_as_shape(source, "decoder.decoder_layers.0.weight", type,
        {vocab_size, hidden_size, 1}, core::TensorShape::from_dims({vocab_size, hidden_size})),
    store.load_f32_tensor(source, "decoder.decoder_layers.0.bias", {vocab_size})};
```

### `runtime.cpp`

Build the ggml graph: subsampling → N encoder blocks → your decode head.
If you're reusing `DepthwiseConvSubsamplingModule` +
`RelativeConformerBlockModule` (rel_pos FastConformer), copy the wiring
pattern from `src/models/canary_asr/runtime.cpp`'s `build_encoder()`:

```cpp
auto x = modules::DepthwiseConvSubsamplingModule({feat_in, hidden_size, subsampling_channels})
    .build(ctx, features, weights.subsampling, {stage1_keep, stage2_keep, keep});
modules::ConformerBlockConfig block_config{hidden_size, num_heads, intermediate_size, conv_kernel_size};
block_config.contiguous_glu_gate = true;
for (const auto & layer : weights.layers) {
    x = modules::RelativeConformerBlockModule(block_config).build(ctx, x, pos, layer, mask, keep, keep);
}
```

You still need, in your own code:
- **The relative positional encoding table.** The sinusoidal formula is
  the same across every `rel_pos` NeMo model; copy it verbatim from
  `src/models/canary_asr/runtime.cpp` (or `src/community_models/
  parakeet_tdt/encoder.cpp`'s `make_relative_positional_encoding`, same
  math). `RelativeSelfAttentionModule` projects this raw table internally
  via each layer's `linear_pos` weight — pass the raw un-projected table,
  not a precomputed projection.
- **Subsampled frame-count bookkeeping.** For NeMo's standard `dw_striding`
  (kernel 3, stride 2, padding 1) applied 3 times for factor 8: per-stage
  valid-frame counts collapse to closed forms `(valid+1)/2`, `(valid+3)/4`,
  `(valid+7)/8` (integer division) — see `src/models/canary_asr/runtime.cpp`,
  cross-checked against `src/community_models/parakeet_tdt/encoder.cpp`'s
  per-stage `conv_out` formula.
- **The decode head + decoding algorithm.** For plain CTC: a linear
  projection to `vocab_size` classes, `ggml_argmax` per frame (reshape to
  `{frames, vocab_size}` first, since `ggml_argmax` reduces over the fastest
  dimension), then greedy decode in C++ (collapse consecutive duplicate
  labels, drop the blank id). Far simpler than a transducer's per-frame
  joint-network search loop (compare `src/community_models/parakeet_tdt/decoder.cpp`,
  376 lines, vs. `src/models/fastconformer_ctc/runtime.cpp`'s ~15-line
  greedy CTC loop).

### `session.cpp`

Wire into `runtime::RuntimeSessionBase` +
`runtime::IOfflineVoiceTaskSession`, following
`src/models/gigaam_asr/session.cpp` almost verbatim if your family is also
offline CTC: audio chunking (`audio::parse_audio_chunk_mode`,
`plan_quiet_energy_audio_chunks`/`plan_audio_chunks`/`plan_vad_audio_chunks`
from `engine/framework/audio/chunking.h`), and word-timestamp derivation
from token emission frames (walk decoded token ids, split on the SentencePiece
word-boundary marker `▁` = bytes `\xE2\x96\x81`, using each token's stored
frame index to compute timestamps).

### Frontend

Use `engine::audio::NemoMelFrontend`
(`include/engine/framework/audio/nemo_mel_frontend.h`) — a generic,
shared, already-validated implementation of NeMo's
`AudioToMelSpectrogramPreprocessor`. Read the checkpoint's own
`preprocessor.featurizer.fb` (mel filterbank) and
`preprocessor.featurizer.window` tensors and construct it in
`MelWindow::FromArgument` / `MelBank::FromArgument` mode — this is
bit-exact to the original NeMo frontend, not a re-derived approximation.
Map the YAML preprocessor config to `NemoMelFrontendConfig`:

| NeMo yaml key | `NemoMelFrontendConfig` field |
|---|---|
| `normalize: per_feature` | `norm = MelNorm::PerBinF32` (or `PerBinF64` for more precision) |
| `window: hann` | `window = MelWindow::FromArgument` (bit-exact) or `SymmetricPrecise` |
| `n_fft`/`window_size`/`window_stride`/`sample_rate` | `stft = {n_fft, hop_length, win_length, center=true, ...}` |
| `preemph` (default 0.97 if absent) | `preemphasis` |
| `log: true` | `mel_path = MelPath::LogMelSpectrogram` |
| (dither is usually skippable at inference; NeMo's own dither stddev is tiny, e.g. `1e-5`) | leave `dither_stddev` at 0 |

`layout = MelLayout::FeatureMajor` gives you `values[mel * frames + t]`
(matches `canary_asr`'s and this family's transpose-on-read convention);
`TimeMajor` gives `values[t * mel + m]` directly if your encoder input
wants that instead.

### `model_specs/<family>.json`

Minimum valid schema-v1 spec (schema requirements checked against
`src/framework/model_spec/schema.cpp`):

```json
{
  "schema_version": 1,
  "family": "<family>",
  "display_name": "...",
  "description": "...",
  "category": "asr",
  "status": "experimental",
  "tasks": ["asr"],
  "modes": ["offline"],
  "languages": ["<bcp-47 or language code>"],
  "capabilities": { "asr": ["word_timestamps"] },
  "dependencies": [],
  "options": { "request": [...], "session": [...], "load": [] },
  "runtime": { "tags": ["gguf"] },
  "ui": { "tags": ["ASR", "GGUF"], "docs": ["conversion.md"] },
  "packages": [],
  "sources": [
    { "format": "safetensors", "roots": {"model": "."},
      "files": {"config": "model:config.json"},
      "tensors": {"weights": "model:model.safetensors"} },
    { "format": "gguf", "roots": {"model": ".", "weights": "$gguf"},
      "files": {"config": "model:config.json"},
      "tensors": {"weights": "weights:"} }
  ]
}
```

Notes: `status: "experimental"` is what lets `packages: []` pass
validation (any other status requires at least one real package with a
`default: true` entry and a real download source — don't set that until
you've actually published a GGUF release). Copy the `request`/`session`
options block from `model_specs/gigaam_asr.json` if you're doing the same
CTC + chunking + timestamps pattern.

### Register in `CMakeLists.txt`

```cmake
audiocpp_add_model(<family>
    SOURCES
        src/models/<family>/assets.cpp
        src/models/<family>/weights.cpp
        src/models/<family>/runtime.cpp
        src/models/<family>/session.cpp
    INCLUDES
        engine/models/<family>/model.h
    LOADERS
        engine::models::<family>::make_<family>_loader
)
```

Add this block next to a similar existing family (alphabetical-ish
grouping isn't strict, just keep it near a related entry for readability).
This macro (`function(audiocpp_add_model ...)` in `CMakeLists.txt`) does
all the registration work — no other file needs manual editing for the
runtime to find the new family.

### Build (CPU)

```bash
cmake -S . -B build/linux-cpu-release   # if not already configured
cmake --build build/linux-cpu-release --parallel --target engine_model_<family>
cmake --build build/linux-cpu-release --parallel --target audiocpp_gguf audiocpp_cli
```

If it doesn't compile cleanly, that's a real signal to fix before moving
on — a family that compiles clean on the first try (as `fastconformer_ctc`
did here) is a strong sign the module APIs were used correctly.

### Build (CUDA)

No CUDA-specific code needed — the shared modules already run on GPU.

```bash
scripts/build_linux.sh --backend cuda --cuda-arch native \
  --target audiocpp_cli --target audiocpp_server
```

Creates `build/linux-cuda-release`. `--cuda-arch native` autodetects the
local GPU; pass an explicit arch (`89` Ada, `86` Ampere, `80` A100) to
build for a different machine. `audiocpp_gguf` needs no CUDA build —
package with `build/linux-cpu-release` and just run with the CUDA binary:

```bash
build/linux-cuda-release/bin/audiocpp_cli --task asr --family fastconformer_ctc \
  --model models/mch_best_e72_gguf/mch_best_e72_gguf_f16.gguf \
  --backend cuda --audio keywords_16k.wav --text-out keywords_16k.txt --log
```

If `audiocpp_cli` reports `unsupported model family hint: <family>`, the
build directory was configured before the family was added to
`CMakeLists.txt` — reconfigure, then rebuild:

```bash
cmake -S . -B build/linux-cuda-release
cmake --build build/linux-cuda-release --parallel --target audiocpp_cli
```

## Step 4 — Write the `.nemo` → safetensors conversion script

Structure (see `convert_titu_to_safetensors.py` in the repo root for the
complete, working version):

```python
python3 convert_titu_to_safetensors.py \
  /home/admin_noeticx/llm_models/nemo_asr/mch_best_e72.nemo models/mch_best_e72_gguf/staging
```

### Checklist of gotchas (each one was hit and fixed in the worked example)

1. **`vocab_size` off-by-one.** NeMo's `decoder.num_classes` in the YAML
   counts vocabulary tokens *only*; CTC appends one blank class on top, so
   the actual classifier head width — and the `decoder.decoder_layers.0`
   tensor's real output dimension — is `num_classes + 1`. **Verify against
   the actual tensor shape**, not just the YAML field:
   ```python
   assert state["decoder.decoder_layers.0.weight"].shape[0] == num_classes + 1
   ```
   Symptom if wrong: an immediate, clear error at load time (tokenizer
   size vs. classifier size mismatch) — the easy one to catch.

2. **`xscaling` (the dangerous one — no error, just wrong output).** If
   `encoder.self_attention_model: rel_pos` and `encoder.xscaling: true`,
   NeMo multiplies the subsampling module's output by
   `xscale = sqrt(d_model)` before the encoder stack. None of audio.cpp's
   shared FastConformer framework modules apply this automatically — it
   must be folded into the subsampling output projection's weight and bias
   at conversion time (a linear layer's output scales linearly with its
   weights, so `Linear(x) * k == Linear_with_scaled_weights(x)` exactly):
   ```python
   xscale = encoder_cfg["d_model"] ** 0.5
   for name in ("encoder.pre_encode.out.weight", "encoder.pre_encode.out.bias"):
       tensors[name] = tensors[name] * xscale
   ```
   **Symptom if skipped: no crash, no error — just wrong transcripts.**
   The model runs to completion and produces fluent-*looking* text that
   repeats a small handful of extremely common words regardless of the
   actual audio content (e.g. "থেকে করে থেকে করে এই..." on repeat for any
   input). This happens because every encoder layer's first op is a
   LayerNorm, which renormalizes away most of a missing constant input
   scale for the content branches — but the *residual stream's relative
   magnitude* still degrades badly: with embeddings ~22x too small
   (`sqrt(512)`), each layer's normally-scaled branch output dominates the
   much smaller carried-over signal on every residual add, so real
   temporal information gets overwritten across all N layers, leaving the
   model expressing mostly its generic word-frequency prior. **If you see
   this symptom on a new checkpoint, check `encoder.xscaling` first.**

3. **Verify shapes empirically, not just from the YAML**, especially for
   anything you compute rather than read directly (`intermediate_size =
   ff_expansion_factor * d_model`, subsampling channel counts, etc.) —
   cross-check against the corresponding tensor's actual shape in the
   `state_dict` dump from Step 1.

4. **Don't rename tensors you don't have to.** If your C++ loader reads
   raw NeMo names (recommended per Step 3), the only tensors that need
   renaming are ones with no direct C++ analog — typically just
   `preprocessor.featurizer.fb`/`.window` → whatever your `assets.cpp`
   expects them called, plus squeezing `fb`'s leading batch dim.

## Step 5 — safetensors to GGUF

```bash
build/linux-cpu-release/bin/audiocpp_gguf \
  --input models/mch_best_e72_gguf/staging/model.safetensors \
  --root models/mch_best_e72_gguf/staging \
  --family fastconformer_ctc \
  --model-spec model_specs/fastconformer_ctc.json \
  --type f16 \
  --output models/mch_best_e72_gguf_f16.gguf \
  --overwrite
```

`docs/gguf.md` writes this step against `build/debug`, which doesn't exist
unless you specifically configure it — this repo's normal install flow
(`scripts/build_linux.sh`) only creates named preset directories such as
`build/linux-cpu-release`/`build/linux-cuda-release`. `audiocpp_gguf` is a
plain host-side CMake target with no GPU-backend dependency, so it builds
and runs fine inside whichever preset you already have configured.

`--type f16`/`bf16`/`q8_0`/`orig` etc. control the packaged weight
precision (see `docs/gguf.md`'s "Type Notes"); `orig` preserves the
checkpoint's native dtype exactly, useful for a first correctness check
before trying quantized variants.

Inspect the result before running it:

```bash
build/linux-cpu-release/bin/audiocpp_gguf --inspect models/<Model>-GGUF/<model>-f16.gguf
```

## Step 6 — Run it, and a debugging playbook

```bash
ffmpeg -y -i keywords.mp3 -ar 16000 -ac 1 keywords_16k.wav

build/linux-cpu-release/bin/audiocpp_cli --task asr --family fastconformer_ctc \
  --model models/mch_best_e72_gguf/mch_best_e72_gguf_f16.gguf \
  --backend cpu --audio keywords_16k.wav --text-out keywords_16k.txt --log
```

Run with the CUDA build's binary and `--backend cuda` (same `.gguf` file
either backend produced it with — weight precision, not backend, is what
`--type` in Step 5 controls):

```bash
build/linux-cuda-release/bin/audiocpp_cli --task asr --family fastconformer_ctc \
  --model models/mch_best_e72_gguf/mch_best_e72_gguf_f16.gguf \
  --backend cuda --audio keywords_16k.wav --text-out keywords_16k.txt --log
```

If something's wrong, match the symptom:

| Symptom | Likely cause |
|---|---|
| Fails immediately: "model package spec path does not exist" | `model_specs/<family>.json` missing, or wrong `--model-spec` path |
| Fails immediately: tensor/namespace mismatch at packaging time | Your converter's tensor names don't match what `weights.cpp` reads — recheck against the real `state_dict` names from Step 1 |
| Fails immediately: tokenizer/classifier size mismatch | `vocab_size` wrong — see Step 4, gotcha #1 |
| Runs fine, but output is empty / all blanks | Check `blank_token_id` matches the actual last class index; check frame-count/masking math didn't zero out all valid frames |
| Runs fine, produces *fluent-looking but content-free* repeated common words regardless of audio | `xscaling` not folded — see Step 4, gotcha #2 |
| Runs, output is plausible-length but scrambled/wrong words throughout | Check subsampling channel/feature-count math, and that any tensor you reshape (`load_tensor_as_shape`) uses the *exact* source shape, not an assumed one |
| Crashes deep in graph compute (NaN, shape assert) | Check attention head_dim divides evenly, mask tensor shapes match `{encoded_frames, encoded_frames}`, and every `TensorShape::from_dims` matches ggml's fastest-dimension-first convention consistently with how you fill the buffer |

If you're debugging a "runs but wrong" case and can't tell whether it's
random garbage or dominated by a small class subset, add a temporary debug
dump of the raw per-frame argmax/label sequence (see the git history of
`src/models/fastconformer_ctc/runtime.cpp` for exactly what this looked
like during the Titu port) — a small number of repeating classes across a
long utterance is the `xscaling` signature; scattered near-random classes
point more toward a tensor-shape or masking bug.

**Cross-check the tokenizer independently of your C++ if in doubt**:
```python
import sentencepiece as spm
sp = spm.SentencePieceProcessor(model_file="tokenizer.model")
# compare sp.id_to_piece(i) against nemo_cfg["decoder"]["vocabulary"][i]
# for a range of i -- if these match, your decode ids are landing on the
# right text and the bug is upstream (encoder), not in the tokenizer wiring.
```

---

## Worked example summary: `fastconformer_ctc`

What exists in this repo as a result of porting
`hishab/titu_stt_bn_fastconformer`:

- `include/engine/models/fastconformer_ctc/model.h`,
  `src/models/fastconformer_ctc/{assets,weights,runtime,session}.cpp` — a
  complete offline CTC family reusing `canary_asr`'s generic
  `DepthwiseConvSubsamplingModule`/`RelativeConformerBlockModule` for the
  encoder, with a small linear CTC head and greedy decode.
- `model_specs/fastconformer_ctc.json` — `status: experimental`, no
  published package yet.
- `CMakeLists.txt` registration via `audiocpp_add_model(fastconformer_ctc ...)`.
- `inspect_nemo.py` / `convert_titu_to_safetensors.py` (repo root; also
  mirrored in `nemo2gguf/`) — the Step 1 inspector and Step 4 converter,
  both gotchas from the checklist fixed in the shipped version.

Reproduce end to end:

```bash
source .venv/bin/activate
hf download hishab/titu_stt_bn_fastconformer titu_stt_bn_fastconformer.nemo \
  --local-dir models/titu_stt_bn_fastconformer
python3 inspect_nemo.py models/titu_stt_bn_fastconformer/titu_stt_bn_fastconformer.nemo
python3 convert_titu_to_safetensors.py \
  models/titu_stt_bn_fastconformer/titu_stt_bn_fastconformer.nemo \
  models/titu_stt_bn_fastconformer/staging

cmake --build build/linux-cpu-release --parallel --target audiocpp_gguf audiocpp_cli
build/linux-cpu-release/bin/audiocpp_gguf \
  --input models/titu_stt_bn_fastconformer/staging/model.safetensors \
  --root models/titu_stt_bn_fastconformer/staging \
  --family fastconformer_ctc --model-spec model_specs/fastconformer_ctc.json \
  --type f16 --output models/Titu-STT-Bn-GGUF/titu-stt-bn-fastconformer-f16.gguf --overwrite

hf download hishab/titu_stt_bn_fastconformer test_bn_fastconformer.wav \
  --local-dir models/titu_stt_bn_fastconformer
build/linux-cpu-release/bin/audiocpp_cli --task asr --family fastconformer_ctc \
  --model models/Titu-STT-Bn-GGUF/titu-stt-bn-fastconformer-f16.gguf \
  --backend cpu --audio models/titu_stt_bn_fastconformer/test_bn_fastconformer.wav \
  --text-out transcript.txt
```

## Applying this to a different NeMo FastConformer-CTC checkpoint

**Confirmed**, not just theoretical: a second, unrelated NeMo
FastConformer-CTC checkpoint (`mch_best_e72.nemo`) converted and ran
correctly through this same `fastconformer_ctc` family with zero C++
changes — see the real commands substituted into Steps 4–6 above.

If your new checkpoint has the *same* architecture shape as Titu
(`self_attention_model: rel_pos`, `subsampling: dw_striding`, decoder
`ConvASRDecoder`), you don't need new C++ at all — reuse the
`fastconformer_ctc` family directly:

1. Run `inspect_nemo.py` and confirm those three facts.
2. Run `convert_titu_to_safetensors.py` (generalize its hardcoded filename
   handling if needed — nothing else in it is Bangla- or Titu-specific).
   It already validates `xscaling` and fails loudly if it sees `false`
   (that code path was never exercised, so it isn't safe to assume-fold).
3. Package and run with `--family fastconformer_ctc` as above.

Architectures `fastconformer_ctc` does **not** cover, needing a different
family or a new one: hybrid CTC+RNNT heads (both a `decoder` and an
`aux_ctc`/joint network present), `self_attention_model: rope`/`rotary`
(GigaAM-style — use `gigaam_asr`), or non-`dw_striding` subsampling.
