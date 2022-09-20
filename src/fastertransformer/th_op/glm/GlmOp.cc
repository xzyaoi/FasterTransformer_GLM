/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "src/fastertransformer/th_op/glm/GlmOp.h"

namespace th = torch;
namespace torch_ext {

GlmOp::GlmOp(const c10d::ProcessGroupNCCL& p,
                             const int64_t rank,
                             const int64_t head_num,
                             const int64_t size_per_head,
                             const int64_t inter_size,
                             const int64_t layer_num,
                             const int64_t vocab_size,
                             const int64_t rotary_embedding_dim,
                             const int64_t start_id,
                             const int64_t end_id,
                             const int64_t tensor_para_size,
                             const int64_t pipeline_para_size,
                             const std::vector<th::Tensor> weights):
    vocab_size_(vocab_size), st_(weights[0].scalar_type())
{
    const c10d::ProcessGroupNCCL* p_ = &p;

    h = (HackNCCLGroup*)(void*)p_;

    for (auto t : weights) {
        CHECK_INPUT(t, st_);
    }

    switch (st_) {
        case at::ScalarType::Float:
            ftglm = new FTGlm<float>(h,
                                     (size_t)rank,
                                     (size_t)head_num,
                                     (size_t)size_per_head,
                                     (size_t)inter_size,
                                     (size_t)layer_num,
                                     (size_t)vocab_size,
                                     (size_t)rotary_embedding_dim,
                                     start_id,
                                     end_id,
                                     tensor_para_size,
                                     pipeline_para_size,
                                     weights);
            break;
        case at::ScalarType::Half:
            ftglm = new FTGlm<half>(h,
                                    (size_t)rank,
                                    (size_t)head_num,
                                    (size_t)size_per_head,
                                    (size_t)inter_size,
                                    (size_t)layer_num,
                                    (size_t)vocab_size,
                                    (size_t)rotary_embedding_dim,
                                    start_id,
                                    end_id,
                                    tensor_para_size,
                                    pipeline_para_size,
                                    weights);
            break;
        default:
            throw std::runtime_error("Wrong Tensor type.");
    }
}

GlmOp::~GlmOp()
{
    delete ftglm;
}

void GlmOp::init_model(const int64_t output_len_,
                        const int64_t beam_width_,
                        const int64_t top_k_,
                        const double top_p_,
                        const double beam_search_diversity_rate_,
                        const double temperature_,
                        const double len_penalty_,
                        const double repetition_penalty_,
                        const int64_t random_seed_)
{
    output_len = output_len_;
    beam_width = beam_width_;
    top_k = top_k_;
    top_p = top_p_;
    beam_search_diversity_rate = beam_search_diversity_rate_;
    temperature = temperature_;
    len_penalty = len_penalty_;
    repetition_penalty = repetition_penalty_;
    random_seed = random_seed_;

    ftglm->init_model((const size_t)output_len,
                   (const size_t)beam_width,
                   (const size_t)top_k,
                   (const float)top_p,
                   (const float)beam_search_diversity_rate,
                   (const float)temperature,
                   (const float)len_penalty,
                   (const float)repetition_penalty,
                   (const unsigned long long int)random_seed);
}

std::vector<th::Tensor> GlmOp::forward(th::Tensor input_ids,
                                               th::Tensor input_lengths,
                                               const int64_t return_cum_log_probs)
{
    CHECK_TH_CUDA(input_ids);
    CHECK_CONTIGUOUS(input_ids);
    TORCH_CHECK(input_ids.dtype() == torch::kInt32, "input_ids dtype should be int32");
    CHECK_TH_CUDA(input_lengths);
    CHECK_CONTIGUOUS(input_lengths);
    TORCH_CHECK(input_lengths.dtype() == torch::kInt32, "input_lengths dtype should be int32");
    TORCH_CHECK(return_cum_log_probs == 0 || return_cum_log_probs == 1 || return_cum_log_probs == 2,
                "return_cum_log_probs should be"
                " 0 (no return cum_log_probs), "
                " 1 (the cumulative log probs of generated sequences), or"
                " 2 (the cumulative log probs of sequences).")

    const int batch_size = input_ids.size(0);
    const int max_input_length = input_ids.size(1);
    const int total_request_output_len = max_input_length + output_len;
    th::Tensor output_ids = torch::empty({batch_size, beam_width, total_request_output_len},
                                         torch::dtype(torch::kInt32).device(torch::kCUDA).requires_grad(false));
    th::Tensor output_ids_buf = torch::empty({batch_size, beam_width, total_request_output_len},
                                         torch::dtype(torch::kInt32).device(torch::kCUDA).requires_grad(false));
    th::Tensor logits_buf =
        torch::empty({batch_size, beam_width, vocab_size_}, torch::dtype(torch::kFloat32).device(torch::kCUDA).requires_grad(false));
    th::Tensor parent_ids = torch::empty({total_request_output_len, batch_size, beam_width},
                                         torch::dtype(torch::kInt32).device(torch::kCUDA).requires_grad(false));
    th::Tensor sequence_lengths =
        torch::empty({batch_size, beam_width}, torch::dtype(torch::kInt32).device(torch::kCUDA).requires_grad(false));
    th::Tensor cum_log_probs =
        torch::empty({batch_size, beam_width}, torch::dtype(torch::kFloat32).device(torch::kCUDA).requires_grad(false));

    ftglm->forward(input_ids,
                   input_lengths,
                   output_ids,
                   output_ids_buf,
                   logits_buf,
                   parent_ids,
                   sequence_lengths,
                   cum_log_probs,
                   return_cum_log_probs);
    if (return_cum_log_probs > 0) {
        return std::vector<th::Tensor>{output_ids, sequence_lengths, cum_log_probs};
    }
    return std::vector<th::Tensor>{output_ids, sequence_lengths};
}


std::vector<th::Tensor> GlmOp::encode(th::Tensor input_ids,
                                               th::Tensor input_lengths,
                                               th::Tensor output_ids_buf,
                                               th::Tensor logits_buf,
                                               th::Tensor output_ids,
                                               th::Tensor parent_ids,
                                               th::Tensor sequence_lengths,
                                               th::Tensor cum_log_probs,
                                               const int64_t return_cum_log_probs)
{
    CHECK_TH_CUDA(input_ids);
    CHECK_CONTIGUOUS(input_ids);
    TORCH_CHECK(input_ids.dtype() == torch::kInt32, "input_ids dtype should be int32");
    CHECK_TH_CUDA(input_lengths);
    CHECK_CONTIGUOUS(input_lengths);
    TORCH_CHECK(input_lengths.dtype() == torch::kInt32, "input_lengths dtype should be int32");
    TORCH_CHECK(return_cum_log_probs == 0 || return_cum_log_probs == 1 || return_cum_log_probs == 2,
                "return_cum_log_probs should be"
                " 0 (no return cum_log_probs), "
                " 1 (the cumulative log probs of generated sequences), or"
                " 2 (the cumulative log probs of sequences).")


    ftglm->encode(input_ids,
                   input_lengths,
                   output_ids,
                   output_ids_buf,
                   logits_buf,
                   parent_ids,
                   sequence_lengths,
                   cum_log_probs,
                   return_cum_log_probs);

    return std::vector<th::Tensor>{};
}

std::vector<th::Tensor> GlmOp::decode(const int64_t step)
{
    ftglm->decode(step);
    return std::vector<th::Tensor>{};
}


}  // namespace torch_ext


PYBIND11_MODULE(libth_glm, m) {
    pybind11::class_<torch_ext::GlmOp>(m, "Glm")
        .def(pybind11::init<c10d::ProcessGroupNCCL&,
                                int64_t,
                                int64_t,
                                int64_t,
                                int64_t,
                                int64_t,
                                int64_t,
                                int64_t,
                                int64_t,
                                int64_t,
                                int64_t,
                                int64_t,
                                std::vector<th::Tensor>>())
        .def("init_model", &torch_ext::GlmOp::init_model)
        .def("forward", &torch_ext::GlmOp::forward)
        .def("encode", &torch_ext::GlmOp::encode)
        .def("decode", &torch_ext::GlmOp::decode);

}
