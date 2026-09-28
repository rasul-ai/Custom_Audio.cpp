#include "engine/models/fastconformer_ctc/model.h"

#include "engine/framework/modules/weight_binding.h"

namespace engine::models::fastconformer_ctc {

std::unique_ptr<FastConformerCtcWeights> load_fastconformer_ctc_weights(
    const FastConformerCtcAssets & assets, core::ExecutionContext & execution, assets::TensorStorageType type) {
    namespace binding = modules::binding;
    auto out = std::make_unique<FastConformerCtcWeights>();
    out->store = std::make_unique<core::BackendWeightStore>(
        execution.backend(), execution.backend_type(), "fastconformer_ctc.weights", 4 * 1024 * 1024);
    auto & store = *out->store;
    const auto & source = *assets.source;
    const auto & c = assets.config;

    const auto norm = [&](const std::string & name) {
        return binding::norm_from_named_source(store, source, name + ".weight", name + ".bias");
    };
    const auto linear = [&](const std::string & name) {
        return binding::linear_from_named_source(store, source, name + ".weight", name + ".bias", type);
    };

    out->subsampling.input_conv = binding::conv2d_from_source(store, source, "encoder.pre_encode.conv.0",
        assets::TensorStorageType::F32, c.subsampling_channels, 1, 3, 3, true);
    out->subsampling.stages.resize(2);
    out->subsampling.stages[0].depthwise = binding::conv2d_from_source(store, source, "encoder.pre_encode.conv.2",
        assets::TensorStorageType::F32, c.subsampling_channels, 1, 3, 3, true);
    out->subsampling.stages[0].pointwise = binding::conv2d_from_source(store, source, "encoder.pre_encode.conv.3",
        assets::TensorStorageType::F32, c.subsampling_channels, c.subsampling_channels, 1, 1, true);
    out->subsampling.stages[1].depthwise = binding::conv2d_from_source(store, source, "encoder.pre_encode.conv.5",
        assets::TensorStorageType::F32, c.subsampling_channels, 1, 3, 3, true);
    out->subsampling.stages[1].pointwise = binding::conv2d_from_source(store, source, "encoder.pre_encode.conv.6",
        assets::TensorStorageType::F32, c.subsampling_channels, c.subsampling_channels, 1, 1, true);
    out->subsampling.projection = linear("encoder.pre_encode.out");

    for (int64_t i = 0; i < c.num_layers; ++i) {
        const auto p = "encoder.layers." + std::to_string(i);
        modules::RelativeConformerBlockWeights w;
        w.ffn1_norm = norm(p + ".norm_feed_forward1");
        w.ffn1_fc1 = linear(p + ".feed_forward1.linear1");
        w.ffn1_fc2 = linear(p + ".feed_forward1.linear2");
        w.norm1 = norm(p + ".norm_self_att");
        {
            const auto q = linear(p + ".self_attn.linear_q");
            const auto k = linear(p + ".self_attn.linear_k");
            const auto v = linear(p + ".self_attn.linear_v");
            const auto o = linear(p + ".self_attn.linear_out");
            w.self_attention.attention.q_weight = q.weight; w.self_attention.attention.q_bias = q.bias;
            w.self_attention.attention.k_weight = k.weight; w.self_attention.attention.k_bias = k.bias;
            w.self_attention.attention.v_weight = v.weight; w.self_attention.attention.v_bias = v.bias;
            w.self_attention.attention.out_weight = o.weight; w.self_attention.attention.out_bias = o.bias;
        }
        w.self_attention.pos_weight = store.load_tensor(
            source, p + ".self_attn.linear_pos.weight", type, {c.hidden_size, c.hidden_size});
        const int64_t head_dim = c.hidden_size / c.num_heads;
        w.self_attention.pos_bias_u = store.load_f32_tensor(source, p + ".self_attn.pos_bias_u", {c.num_heads, head_dim});
        w.self_attention.pos_bias_v = store.load_f32_tensor(source, p + ".self_attn.pos_bias_v", {c.num_heads, head_dim});
        w.conv.norm = norm(p + ".norm_conv");
        w.conv.pointwise_in = {
            store.load_tensor_as_shape(source, p + ".conv.pointwise_conv1.weight", type,
                {2 * c.hidden_size, c.hidden_size, 1}, core::TensorShape::from_dims({2 * c.hidden_size, c.hidden_size})),
            store.load_f32_tensor(source, p + ".conv.pointwise_conv1.bias", {2 * c.hidden_size})};
        w.conv.pointwise_out = {
            store.load_tensor_as_shape(source, p + ".conv.pointwise_conv2.weight", type,
                {c.hidden_size, c.hidden_size, 1}, core::TensorShape::from_dims({c.hidden_size, c.hidden_size})),
            store.load_f32_tensor(source, p + ".conv.pointwise_conv2.bias", {c.hidden_size})};
        w.conv.depthwise = binding::depthwise_conv1d_from_source(
            store, source, p + ".conv.depthwise_conv", assets::TensorStorageType::F32, c.hidden_size, c.conv_kernel_size, true);
        w.conv.depthwise_norm = binding::batch_norm_eval_from_source(
            store, source, p + ".conv.batch_norm", c.hidden_size, 1e-5f);
        w.norm2 = norm(p + ".norm_feed_forward2");
        w.ffn2_fc1 = linear(p + ".feed_forward2.linear1");
        w.ffn2_fc2 = linear(p + ".feed_forward2.linear2");
        w.final_norm = norm(p + ".norm_out");
        out->layers.push_back(std::move(w));
    }

    out->ctc_head = {
        store.load_tensor_as_shape(source, "decoder.decoder_layers.0.weight", type,
            {c.vocab_size, c.hidden_size, 1}, core::TensorShape::from_dims({c.vocab_size, c.hidden_size})),
        store.load_f32_tensor(source, "decoder.decoder_layers.0.bias", {c.vocab_size})};

    store.upload();
    return out;
}

}  // namespace engine::models::fastconformer_ctc
