#include "engine/models/fastconformer_ctc/model.h"

#include "engine/framework/model_spec/package.h"

#include <stdexcept>

namespace engine::models::fastconformer_ctc {
namespace json = engine::io::json;

std::string FastConformerCtcAssets::decode(const std::vector<int32_t> & ids) const {
    return tokenizers::decode_sentencepiece(pieces, ids);
}

std::shared_ptr<const FastConformerCtcAssets> load_fastconformer_ctc_assets(const std::filesystem::path & path) {
    auto out = std::make_shared<FastConformerCtcAssets>();
    out->resources = model_spec::load_resource_bundle_for_family(path, "fastconformer_ctc");
    out->source = out->resources.open_tensor_source("weights");
    const auto config = out->resources.parse_json("config");
    auto & c = out->config;
    const auto & encoder = config.require("encoder_config");
    c.feat_in = json::require_i64(encoder, "feat_in");
    c.hidden_size = json::require_i64(encoder, "hidden_size");
    c.intermediate_size = json::require_i64(encoder, "intermediate_size");
    c.num_layers = json::require_i64(encoder, "num_hidden_layers");
    c.num_heads = json::require_i64(encoder, "num_attention_heads");
    c.conv_kernel_size = json::require_i64(encoder, "conv_kernel_size");
    c.subsampling_channels = json::require_i64(encoder, "subsampling_conv_channels");
    if (json::require_i64(encoder, "subsampling_factor") != 8 ||
        json::require_i64(encoder, "subsampling_conv_kernel_size") != 3 ||
        json::require_i64(encoder, "subsampling_conv_stride") != 2) {
        throw std::runtime_error("fastconformer_ctc requires factor-8 dw_striding subsampling (kernel 3, stride 2)");
    }
    c.vocab_size = json::require_i64(config, "vocab_size");
    c.blank_id = json::require_i64(config, "blank_token_id");
    if (c.blank_id != c.vocab_size - 1) {
        throw std::runtime_error("fastconformer_ctc expects the blank id to be the last class");
    }

    out->resources.add_model_file("tokenizer", "tokenizer.model");
    out->pieces = tokenizers::load_sentencepiece_model(out->resources.require_file("tokenizer"));
    if (static_cast<int64_t>(out->pieces.size()) + 1 != c.vocab_size) {
        throw std::runtime_error("fastconformer_ctc tokenizer size does not match the classifier");
    }

    const auto & feature = config.require("processor_config").require("feature_extractor");
    if (json::require_i64(feature, "sampling_rate") != 16000 ||
        json::require_i64(feature, "feature_size") != c.feat_in) {
        throw std::runtime_error("fastconformer_ctc frontend configuration does not match the encoder");
    }
    const int64_t n_fft = json::require_i64(feature, "n_fft");
    const int64_t win_length = json::require_i64(feature, "win_length");
    const int64_t hop_length = json::require_i64(feature, "hop_length");
    const int64_t bins = n_fft / 2 + 1;

    out->window = out->source->require_f32("preprocessor.window", {win_length});
    out->filterbank = {out->source->require_f32("preprocessor.fb", {c.feat_in, bins}), {c.feat_in, bins}};

    audio::NemoMelFrontendConfig frontend_config;
    frontend_config.sample_rate = 16000;
    frontend_config.n_mels = c.feat_in;
    frontend_config.stft = {n_fft, hop_length, win_length, true, audio::STFTPadMode::Constant};
    frontend_config.preemphasis = json::optional_f32(feature, "preemphasis", 0.97f);
    frontend_config.window = audio::MelWindow::FromArgument;
    frontend_config.mel_bank = audio::MelBank::FromArgument;
    frontend_config.mel_path = audio::MelPath::LogMelSpectrogram;
    frontend_config.norm = audio::MelNorm::PerBinF32;
    frontend_config.layout = audio::MelLayout::FeatureMajor;
    out->frontend = std::make_shared<audio::NemoMelFrontend>(frontend_config, out->window, out->filterbank);
    return out;
}

}  // namespace engine::models::fastconformer_ctc
