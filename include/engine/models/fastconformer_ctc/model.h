#pragma once

#include "engine/framework/assets/resource_bundle.h"
#include "engine/framework/audio/dsp.h"
#include "engine/framework/audio/nemo_mel_frontend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/modules/conformer_modules.h"
#include "engine/framework/runtime/model.h"
#include "engine/framework/tokenizers/sentencepiece.h"

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace engine::models::fastconformer_ctc {

// Standard NeMo FastConformer encoder (dw_striding subsampling, factor 8,
// rel_pos self-attention) with a plain CTC head (nemo.collections.asr.
// modules.ConvASRDecoder) — the architecture used by
// nemo.collections.asr.models.ctc_bpe_models.EncDecCTCModelBPE checkpoints
// such as hishab/titu_stt_bn_fastconformer. See conversion.md at the repo
// root for the .nemo -> safetensors conversion this family expects.
struct FastConformerCtcConfig {
    int64_t feat_in = 80;
    int64_t hidden_size = 512;
    int64_t intermediate_size = 2048;
    int64_t num_layers = 18;
    int64_t num_heads = 8;
    int64_t conv_kernel_size = 9;
    int64_t subsampling_channels = 256;
    int64_t vocab_size = 1025;
    int64_t blank_id = 1024;
};

struct FastConformerCtcAssets {
    assets::ResourceBundle resources;
    std::shared_ptr<const assets::TensorSource> source;
    FastConformerCtcConfig config;
    std::vector<tokenizers::SentencePiecePiece> pieces;
    std::vector<float> window;
    audio::AudioTensor filterbank;
    std::shared_ptr<const audio::NemoMelFrontend> frontend;
    std::string decode(const std::vector<int32_t> & ids) const;
};

struct FastConformerCtcWeights {
    std::unique_ptr<core::BackendWeightStore> store;
    modules::DepthwiseConvSubsamplingWeights subsampling;
    std::vector<modules::RelativeConformerBlockWeights> layers;
    modules::LinearWeights ctc_head;
};

std::shared_ptr<const FastConformerCtcAssets> load_fastconformer_ctc_assets(const std::filesystem::path & path);
std::unique_ptr<FastConformerCtcWeights> load_fastconformer_ctc_weights(
    const FastConformerCtcAssets & assets, core::ExecutionContext & execution, assets::TensorStorageType type);

struct FastConformerCtcDecodedTokens {
    std::vector<int32_t> ids;
    std::vector<int64_t> frames;
    int64_t encoder_frames = 0;
    double encoder_ms = 0.0;
    double inference_ms = 0.0;
};

class FastConformerCtcRuntime {
public:
    FastConformerCtcRuntime(
        const FastConformerCtcAssets & assets, const FastConformerCtcWeights & weights, core::ExecutionContext & execution);
    ~FastConformerCtcRuntime();
    FastConformerCtcDecodedTokens transcribe(const audio::NemoMelFeatures & features);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

std::shared_ptr<runtime::IVoiceModelLoader> make_fastconformer_ctc_loader();

}  // namespace engine::models::fastconformer_ctc
