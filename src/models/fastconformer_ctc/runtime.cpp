#include "engine/models/fastconformer_ctc/model.h"

#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/runtime/cache_slots.h"
#include "engine/framework/runtime/graph_optimizer.h"

#include <ggml-alloc.h>

#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace engine::models::fastconformer_ctc {
namespace {

using Clock = std::chrono::steady_clock;
using core::TensorShape;
using core::TensorValue;

struct Graph {
    core::ExecutionContext & execution;
    std::string name;
    std::unique_ptr<ggml_context, decltype(&ggml_free)> context{nullptr, ggml_free};
    ggml_cgraph * graph = nullptr;
    ggml_gallocr_t allocator = nullptr;
    core::HostGraphPlan host_plan;

    Graph(core::ExecutionContext & execution, size_t nodes, std::string name)
        : execution(execution), name(std::move(name)) {
        const size_t bytes = ggml_graph_overhead_custom(nodes, false) + nodes * ggml_tensor_overhead();
        context.reset(ggml_init({bytes, nullptr, true}));
        if (!context) {
            throw std::runtime_error("fastconformer_ctc graph descriptor allocation failed");
        }
        graph = ggml_new_graph_custom(context.get(), nodes, false);
        allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution.backend()));
    }
    ~Graph() {
        host_plan.reset();
        core::release_backend_graph_resources(execution.backend(), graph, true);
        ggml_gallocr_free(allocator);
    }
    core::ModuleBuildContext build_context() {
        return {context.get(), "fastconformer_ctc", execution.backend_type()};
    }
    void output(const TensorValue & value) {
        ggml_set_output(value.tensor);
        ggml_build_forward_expand(graph, value.tensor);
    }
    void allocate() {
        if (execution.backend_type() == core::BackendType::Cuda) {
            runtime::optimize_graph(*graph, runtime::GraphOptimizationBackend::Gpu);
        } else if (execution.backend_type() == core::BackendType::Cpu) {
            auto options = runtime::graph_optimization_options_for_backend(runtime::GraphOptimizationBackend::Cpu);
            options.elide_metadata_only_ops = false;
            runtime::optimize_graph(*graph, options);
        }
        core::validate_backend_graph_supported(execution.backend(), graph, "fastconformer_ctc");
        if (!ggml_gallocr_alloc_graph(allocator, graph)) {
            throw std::runtime_error("fastconformer_ctc graph buffer allocation failed");
        }
        core::prepare_host_graph_plan(execution, graph, host_plan);
        debug::timing_log_context_reservation(name, context.get());
        debug::timing_log_scalar(name + ".buffer_bytes", ggml_gallocr_get_buffer_size(allocator, 0));
    }
    void compute() {
        if (core::compute_graph(execution, graph, host_plan, "fastconformer_ctc") != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("fastconformer_ctc graph execution failed");
        }
    }
};

// NeMo dw_striding subsampling: kernel 3, stride 2, padding 1. Collapses to
// this closed form after three applications (see conversion.md for the
// derivation cross-checked against src/models/canary_asr/runtime.cpp, which
// uses the same closed forms for the same subsampling parameters).
int64_t subsampled_frames(int64_t frames) {
    return (frames + 7) / 8;
}

void write_relative_positional_encoding(TensorValue & pos, int64_t hidden, int64_t frames) {
    std::vector<float> values(static_cast<size_t>((2 * frames - 1) * hidden));
    for (int64_t p = 0; p < 2 * frames - 1; ++p) {
        for (int64_t i = 0; i < hidden / 2; ++i) {
            const double phase = (frames - 1 - p) * std::pow(10000.0, -2.0 * static_cast<double>(i) / static_cast<double>(hidden));
            values[static_cast<size_t>(p * hidden + 2 * i)] = static_cast<float>(std::sin(phase));
            values[static_cast<size_t>(p * hidden + 2 * i + 1)] = static_cast<float>(std::cos(phase));
        }
    }
    core::write_tensor_f32(pos, values);
}

}  // namespace

struct FastConformerCtcRuntime::Impl {
    const FastConformerCtcAssets & assets;
    const FastConformerCtcWeights & weights;
    core::ExecutionContext & execution;

    struct EncoderGraph {
        std::unique_ptr<Graph> graph;
        TensorValue features, pos, mask, keep, stage1_keep, stage2_keep, encoded;
        int64_t frames = 0;
        int64_t encoded_frames = 0;
    };
    runtime::CacheSlots<int64_t, std::unique_ptr<EncoderGraph>> encoders{2};

    Impl(const FastConformerCtcAssets & assets, const FastConformerCtcWeights & weights, core::ExecutionContext & execution)
        : assets(assets), weights(weights), execution(execution) {}

    EncoderGraph & build_encoder(int64_t frames) {
        if (auto * cached = encoders.find(frames)) {
            return **cached;
        }
        encoders.set_capacity(1);
        encoders.set_capacity(2);
        auto state = std::make_unique<EncoderGraph>();
        auto & g = *state;
        const auto & c = assets.config;
        g.frames = frames;
        const int64_t stage1_frames = (frames + 1) / 2;
        const int64_t stage2_frames = (frames + 3) / 4;
        g.encoded_frames = subsampled_frames(frames);

        g.graph = std::make_unique<Graph>(
            execution, 1024 + 256 * static_cast<size_t>(weights.layers.size()), "fastconformer_ctc.encoder");
        auto ctx = g.graph->build_context();
        g.features = core::make_tensor(ctx, GGML_TYPE_F32, TensorShape::from_dims({1, frames, c.feat_in}));
        g.pos = core::make_tensor(ctx, GGML_TYPE_F32, TensorShape::from_dims({1, 2 * g.encoded_frames - 1, c.hidden_size}));
        g.mask = core::make_tensor(ctx, GGML_TYPE_F32, TensorShape::from_dims({g.encoded_frames, g.encoded_frames}));
        g.keep = core::make_tensor(ctx, GGML_TYPE_I32, TensorShape::from_dims({1, g.encoded_frames}));
        g.stage1_keep = core::make_tensor(ctx, GGML_TYPE_I32, TensorShape::from_dims({1, stage1_frames}));
        g.stage2_keep = core::make_tensor(ctx, GGML_TYPE_I32, TensorShape::from_dims({1, stage2_frames}));
        for (auto * tensor : {g.features.tensor, g.pos.tensor, g.mask.tensor, g.keep.tensor,
                              g.stage1_keep.tensor, g.stage2_keep.tensor}) {
            ggml_set_input(tensor);
        }

        auto x = modules::DepthwiseConvSubsamplingModule({c.feat_in, c.hidden_size, c.subsampling_channels})
            .build(ctx, g.features, weights.subsampling, {g.stage1_keep, g.stage2_keep, g.keep});
        modules::ConformerBlockConfig block_config{c.hidden_size, c.num_heads, c.intermediate_size, c.conv_kernel_size};
        block_config.contiguous_glu_gate = true;
        for (const auto & layer : weights.layers) {
            x = modules::RelativeConformerBlockModule(block_config)
                .build(ctx, x, g.pos, layer, g.mask, g.keep, g.keep);
        }
        x = modules::LinearModule({c.hidden_size, c.vocab_size, true}).build(ctx, x, weights.ctc_head);
        x = core::reshape_tensor(ctx, x, TensorShape::from_dims({g.encoded_frames, c.vocab_size}));
        g.encoded = core::wrap_tensor(ggml_argmax(ctx.ggml, x.tensor), TensorShape::from_dims({g.encoded_frames}), GGML_TYPE_I32);

        g.graph->output(g.encoded);
        g.graph->allocate();
        write_relative_positional_encoding(g.pos, c.hidden_size, g.encoded_frames);

        auto & result = *state;
        encoders.put(frames, std::move(state));
        return result;
    }

    FastConformerCtcDecodedTokens infer(const audio::NemoMelFeatures & features) {
        const auto started = Clock::now();
        auto & g = build_encoder(features.raw_frames);

        // features.values is feature-major ([mel][time]); the subsampling
        // module wants time-major ([time][mel]). See
        // src/models/canary_asr/runtime.cpp's identical transpose.
        std::vector<float> input(static_cast<size_t>(g.frames * assets.config.feat_in), 0.0f);
        for (int64_t t = 0; t < features.valid_frames; ++t) {
            for (int64_t m = 0; m < assets.config.feat_in; ++m) {
                input[static_cast<size_t>(t * assets.config.feat_in + m)] =
                    features.values[static_cast<size_t>(m * features.raw_frames + t)];
            }
        }
        core::write_tensor_f32(g.features, input);

        const int64_t stage1_valid = (features.valid_frames + 1) / 2;
        const int64_t stage2_valid = (features.valid_frames + 3) / 4;
        const int64_t valid_encoded = std::min(g.encoded_frames, subsampled_frames(features.valid_frames));
        const auto fill_keep = [](TensorValue & tensor, int64_t total, int64_t valid) {
            std::vector<int32_t> values(static_cast<size_t>(total), 0);
            std::fill_n(values.begin(), std::min(total, valid), 1);
            core::write_tensor_i32(tensor, values);
        };
        fill_keep(g.stage1_keep, g.stage1_keep.shape.dims[1], stage1_valid);
        fill_keep(g.stage2_keep, g.stage2_keep.shape.dims[1], stage2_valid);
        fill_keep(g.keep, g.encoded_frames, valid_encoded);

        std::vector<float> mask(static_cast<size_t>(g.encoded_frames * g.encoded_frames), 0.0f);
        if (valid_encoded < g.encoded_frames) {
            for (int64_t query = 0; query < g.encoded_frames; ++query) {
                std::fill(mask.begin() + static_cast<std::ptrdiff_t>(query * g.encoded_frames + valid_encoded),
                          mask.begin() + static_cast<std::ptrdiff_t>((query + 1) * g.encoded_frames),
                          -std::numeric_limits<float>::infinity());
            }
        }
        core::write_tensor_f32(g.mask, mask);

        const auto encoder_start = Clock::now();
        g.graph->compute();
        const double encoder_ms = debug::elapsed_ms(encoder_start, Clock::now());

        const auto labels = core::read_tensor_i32(g.encoded.tensor);
        const int32_t blank = static_cast<int32_t>(assets.config.blank_id);
        FastConformerCtcDecodedTokens result;
        result.encoder_ms = encoder_ms;
        result.encoder_frames = valid_encoded;
        int32_t previous = blank;
        for (int64_t t = 0; t < valid_encoded; ++t) {
            const auto token = labels[static_cast<size_t>(t)];
            if (token != blank && token != previous) {
                result.ids.push_back(token);
                result.frames.push_back(t);
            }
            previous = token;
        }
        result.inference_ms = debug::elapsed_ms(started, Clock::now());
        return result;
    }
};

FastConformerCtcRuntime::FastConformerCtcRuntime(
    const FastConformerCtcAssets & assets, const FastConformerCtcWeights & weights, core::ExecutionContext & execution)
    : impl_(std::make_unique<Impl>(assets, weights, execution)) {}

FastConformerCtcRuntime::~FastConformerCtcRuntime() = default;

FastConformerCtcDecodedTokens FastConformerCtcRuntime::transcribe(const audio::NemoMelFeatures & features) {
    return impl_->infer(features);
}

}  // namespace engine::models::fastconformer_ctc
