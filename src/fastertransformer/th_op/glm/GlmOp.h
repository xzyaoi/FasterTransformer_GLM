/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
 * Copyright (c) 2021, NAVER Corp.  Authored by CLOVA.
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

#include <unistd.h>
#include <torch/csrc/autograd/custom_function.h>
#include <torch/extension.h>
#include <c10d/ProcessGroupNCCL.hpp>
#include "src/fastertransformer/models/glm/Glm.h"
#include "src/fastertransformer/th_op/th_utils.h"

namespace ft = fastertransformer;
namespace th = torch;
namespace torch_ext {

// if want to use nccl_name, maybe only support pytorch >= 1.12
class HackNCCLGroup: public c10d::ProcessGroupNCCL {
public:
    ncclComm_t getcomm(size_t rank, size_t size, const char* nccl_name) {
        ncclUniqueId ncclID;
        if (rank == 0) {
            ncclGetUniqueId(&ncclID);
        }
#if defined(TORCH_VERSION_MAJOR) && (TORCH_VERSION_MAJOR > 1 || \
        (TORCH_VERSION_MAJOR == 1 && TORCH_VERSION_MINOR >= 12))
        broadcastUniqueNCCLID(&ncclID,
                true,
                nccl_name,
                rank);
#elif defined(TORCH_VERSION_MAJOR) && (TORCH_VERSION_MAJOR > 1 || \
        (TORCH_VERSION_MAJOR == 1 && TORCH_VERSION_MINOR >= 8))
        broadcastUniqueNCCLID(&ncclID,
                c10d::OpType::SEND,
                nccl_name,
                rank);
#else
        broadcastUniqueNCCLID(&ncclID);
#endif
        ncclComm_t comm;
        NCCLCHECK(ncclCommInitRank(&comm, size, ncclID, rank));
        return comm;
    }
};


using std::vector;

class IFGlm {
public:
    virtual ~IFGlm() {}
    virtual void forward(th::Tensor& input_ids,
                         th::Tensor& input_lengths,
                         th::Tensor& output_ids,
                         th::Tensor& output_ids_buf,
                         th::Tensor& logits_buf,
                         th::Tensor& parent_ids,
                         th::Tensor& sequence_lengths,
                         th::Tensor& cum_log_probs,
                         const size_t request_output_len,
                         const size_t beam_width,
                         const size_t top_k,
                         const float top_p,
                         const float beam_search_diversity_rate,
                         const float temperature,
                         const float len_penalty,
                         const float repetition_penalty,
                         const unsigned long long int random_seed,
                         const int return_cum_log_probs = 0) = 0;
    virtual void encode(th::Tensor& input_ids,
                        th::Tensor& input_lengths,
                        th::Tensor& output_ids_buf,
                        th::Tensor& logits_buf,
                        th::Tensor& output_ids,
                        th::Tensor& parent_ids,
                        th::Tensor& sequence_lengths,
                        th::Tensor& cum_log_probs,
                        const size_t request_output_len,
                        const size_t beam_width,
                        const size_t top_k,
                        const float top_p,
                        const float beam_search_diversity_rate,
                        const float temperature,
                        const float len_penalty,
                        const float repetition_penalty,
                        const unsigned long long int random_seed,
                        const int return_cum_log_probs = 0) = 0;
    virtual void decode(const size_t step) = 0;
};

template<typename T>
class FTGlm: public IFGlm {
public:
    FTGlm(HackNCCLGroup* h,
          const size_t rank,
          const size_t head_num,
          const size_t size_per_head,
          const size_t inter_size,
          const size_t layer_num,
          const size_t vocab_size,
          const size_t rotary_embedding_dim,
          const int start_id,
          const int end_id,
          const int tensor_para_size,
          const int pipeline_para_size,
          const vector<th::Tensor> weights):
        h_(h),
        rank_(rank),
        head_num_(head_num),
        size_per_head_(size_per_head),
        inter_size_(inter_size),
        layer_num_(layer_num),
        rotary_embedding_dim_(rotary_embedding_dim),
        vocab_size_(vocab_size),
        start_id_(start_id),
        end_id_(end_id),
        tensor_para_size_(tensor_para_size),
        pipeline_para_size_(pipeline_para_size),
        weights_(weights)
    {
        ft::check_cuda_error(cublasLtCreate(&cublasltHandle_));
        cublas_algo_map_ = new ft::cublasAlgoMap("gemm_config.in");
        cublas_wrapper_mutex_ = new std::mutex();

        init_nccl_comm();

        glm_weights_.resizeLayer(layer_num_);

        for (int i = 0; i < (int)layer_num_; i++) {
            glm_weights_.decoder_layer_weights[i]->self_attention_weights.query_weight.kernel =
                get_ptr<T>(weights_[i + 0 * layer_num_]);
            glm_weights_.decoder_layer_weights[i]->self_attention_weights.query_weight.bias =
                get_ptr<T>(weights_[i + 1 * layer_num_]);
            glm_weights_.decoder_layer_weights[i]->self_attention_weights.attention_output_weight.kernel =
                get_ptr<T>(weights_[i + 2 * layer_num_]);
            glm_weights_.decoder_layer_weights[i]->self_attention_weights.attention_output_weight.bias =
                get_ptr<T>(weights_[i + 3 * layer_num_]);
            glm_weights_.decoder_layer_weights[i]->self_attn_layernorm_weights.beta =
                get_ptr<T>(weights_[i + 4 * layer_num_]);
            glm_weights_.decoder_layer_weights[i]->self_attn_layernorm_weights.gamma =
                get_ptr<T>(weights_[i + 5 * layer_num_]);
            glm_weights_.decoder_layer_weights[i]->glu_ffn_weights.intermediate_weight[0].kernel =
                get_ptr<T>(weights_[i + 6 * layer_num_]);
            glm_weights_.decoder_layer_weights[i]->glu_ffn_weights.intermediate_weight[0].bias =
                get_ptr<T>(weights_[i + 7 * layer_num_]);
            glm_weights_.decoder_layer_weights[i]->glu_ffn_weights.intermediate_weight[1].kernel  =
                get_ptr<T>(weights_[i + 8 * layer_num_]);
            glm_weights_.decoder_layer_weights[i]->glu_ffn_weights.intermediate_weight[1].bias =
                get_ptr<T>(weights_[i + 9 * layer_num_]);
            glm_weights_.decoder_layer_weights[i]->glu_ffn_weights.output_weight.kernel  =
                get_ptr<T>(weights_[i + 10 * layer_num_]);
            glm_weights_.decoder_layer_weights[i]->glu_ffn_weights.output_weight.bias =
                get_ptr<T>(weights_[i + 11 * layer_num_]);
            glm_weights_.decoder_layer_weights[i]->glu_ffn_layernorm_weights.beta =
                get_ptr<T>(weights_[i + 12 * layer_num_]);
            glm_weights_.decoder_layer_weights[i]->glu_ffn_layernorm_weights.gamma =
                get_ptr<T>(weights_[i + 13 * layer_num_]);
        }

        glm_weights_.post_decoder_layernorm.gamma = get_ptr<T>(weights_[14 * layer_num_ + 0]);
        glm_weights_.post_decoder_layernorm.beta = get_ptr<T>(weights_[14 * layer_num_ + 1]);
        glm_weights_.pre_decoder_embedding_table = get_ptr<T>(weights_[14 * layer_num_ + 2]);
        glm_weights_.post_decoder_embedding.kernel = get_ptr<T>(weights_[14 * layer_num_ + 2]);

        // std::cout << 111 << std::endl;
        // std::cout << weights_[14 * layer_num_ + 2][0] << std::endl;
        // std::cout << weights_[14 * layer_num_ + 2].cpu().data_ptr<at::Half>()[0] << std::endl;
        // std::cout << glm_weights_.pre_decoder_embedding_table << std::endl;
        // std::cout << 222 << std::endl;

        int device_id = 0;
        ft::check_cuda_error(cudaGetDevice(&device_id));
        ft::check_cuda_error(cudaGetDeviceProperties(&prop_, device_id));
        FT_LOG_INFO("Device %s", prop_.name);

    }

    ~FTGlm() override
    {
        ncclCommDestroy(tensor_para_comm_);
        ncclCommDestroy(pipeline_para_comm_);
        cublasLtDestroy(cublasltHandle_);
        delete cublas_algo_map_;
        delete cublas_wrapper_mutex_;
    }

    void init_nccl_comm()
    {
        int rank = rank_;
        tensor_para_rank_ = rank % tensor_para_size_;
        pipeline_para_rank_ = rank / tensor_para_size_;
        
        char tensor_para_name[32], pipeline_para_name[32];
        sprintf(tensor_para_name, "tp_nccl_comm_%lu", pipeline_para_rank_);
        sprintf(pipeline_para_name, "pp_nccl_comm_%lu", tensor_para_rank_);

        tensor_para_comm_ = h_->getcomm(tensor_para_rank_, tensor_para_size_, tensor_para_name);
        pipeline_para_comm_ = h_->getcomm(pipeline_para_rank_, pipeline_para_size_, pipeline_para_name);
    }

    void forward(th::Tensor& input_ids,
                 th::Tensor& input_lengths,
                 th::Tensor& output_ids,
                 th::Tensor& output_ids_buf,
                 th::Tensor& logits_buf,
                 th::Tensor& parent_ids,
                 th::Tensor& sequence_lengths,
                 th::Tensor& cum_log_probs,
                 const size_t request_output_len,
                 const size_t beam_width,
                 const size_t top_k,
                 const float top_p,
                 const float beam_search_diversity_rate,
                 const float temperature,
                 const float len_penalty,
                 const float repetition_penalty,
                 const unsigned long long int query_random_seed,
                 const int return_cum_log_probs = 0) override
    {
        auto stream = at::cuda::getCurrentCUDAStream().stream();
        cublasHandle_t cublasHandle = at::cuda::getCurrentCUDABlasHandle();
        cublasSetStream(cublasHandle, stream);
        allocator = new ft::Allocator<ft::AllocatorType::TH>();
        cublas_wrapper = new ft::cublasMMWrapper(
            cublasHandle, cublasltHandle_, stream, cublas_algo_map_, cublas_wrapper_mutex_, allocator);

        if (std::is_same<T, half>::value) {
            cublas_wrapper->setGemmConfig(CUDA_R_16F, CUDA_R_16F, CUDA_R_16F, CUDA_R_32F);
        }
        else if (std::is_same<T, float>::value) {
            cublas_wrapper->setFP32GemmConfig();
        }

        const size_t request_batch_size = (size_t)input_ids.size(0);
        const size_t max_input_length = (size_t)input_ids.size(1);
        const int total_output_len = (int)(max_input_length + request_output_len);

        ft::NcclParam tensor_para(tensor_para_rank_, tensor_para_size_, tensor_para_comm_);
        ft::NcclParam pipeline_para(pipeline_para_rank_, pipeline_para_size_, pipeline_para_comm_);

        random_seed = query_random_seed;

        glm = new ft::Glm<T>(0,  // max_batch_size, FT will adjust the buffer automatically.
                          0,  // max_seq_len, FT will adjust the buffer automatically.
                          0,  // max_input_len, FT will adjust the buffer automatically.
                          beam_width,
                          head_num_,
                          size_per_head_,
                          inter_size_,
                          layer_num_,
                          vocab_size_,
                          -rotary_embedding_dim_,
                          start_id_,
                          end_id_,
                          0.0f,
                          top_k,
                          top_p,
                          random_seed,
                          temperature,
                          len_penalty,
                          repetition_penalty,
                          tensor_para,
                          pipeline_para,
                          stream,
                          cublas_wrapper,
                          allocator,
                          false,
                          &prop_);
        
        std::unordered_map<std::string, ft::Tensor> input_tensors = std::unordered_map<std::string, ft::Tensor>{
            {"input_ids",
             ft::Tensor{ft::MEMORY_GPU,
                        ft::TYPE_INT32,
                        std::vector<size_t>{request_batch_size, max_input_length},
                        get_ptr<int>(input_ids)}},
            {"input_lengths",
             ft::Tensor{
                 ft::MEMORY_GPU, ft::TYPE_INT32, std::vector<size_t>{request_batch_size}, get_ptr<int>(input_lengths)}},
            {"max_output_seq_len",
             ft::Tensor{ft::MEMORY_CPU, ft::TYPE_INT32, std::vector<size_t>{1}, &total_output_len}}};

        if (top_k == 0 && top_p == 0.0f) {
            ft::FT_CHECK(beam_width > 1);
            input_tensors.insert(
                {"beam_search_diversity_rate",
                 ft::Tensor{ft::MEMORY_CPU, ft::TYPE_FP32, std::vector<size_t>{1}, &beam_search_diversity_rate}});
        }
        else {
            if (top_p != 0.0f) {
                input_tensors.insert(
                    {"runtime_top_p", ft::Tensor{ft::MEMORY_CPU, ft::TYPE_FP32, std::vector<size_t>{1}, &top_p}});
            }
            if (top_k != 0) {
                input_tensors.insert(
                    {"runtime_top_k", ft::Tensor{ft::MEMORY_CPU, ft::TYPE_INT32, std::vector<size_t>{1}, &top_k}});
            }
        }
        input_tensors.insert(
            {"temperature", ft::Tensor{ft::MEMORY_CPU, ft::TYPE_FP32, std::vector<size_t>{1}, &temperature}});
        input_tensors.insert(
            {"len_penalty", ft::Tensor{ft::MEMORY_CPU, ft::TYPE_FP32, std::vector<size_t>{1}, &len_penalty}});
        input_tensors.insert({"repetition_penalty",
                              ft::Tensor{ft::MEMORY_CPU, ft::TYPE_FP32, std::vector<size_t>{1}, &repetition_penalty}});
        input_tensors.insert(
            {"random_seed", ft::Tensor{ft::MEMORY_CPU, ft::TYPE_UINT64, std::vector<size_t>{1}, &random_seed}});
        
        bool return_context_cum_log_probs = false;
        if (return_cum_log_probs == 2) {
            return_context_cum_log_probs = true;
            input_tensors.insert(
                {"is_return_context_cum_log_probs",
                 ft::Tensor{ft::MEMORY_CPU, ft::TYPE_BOOL, std::vector<size_t>{1}, &return_context_cum_log_probs}});
        }

        std::unordered_map<std::string, ft::Tensor> output_tensors = std::unordered_map<std::string, ft::Tensor>{
            {"output_ids",
             ft::Tensor{ft::MEMORY_GPU,
                        ft::TYPE_INT32,
                        std::vector<size_t>{request_batch_size, beam_width, (size_t)total_output_len},
                        get_ptr<int>(output_ids)}},
            {"output_ids_buf",
             ft::Tensor{ft::MEMORY_GPU,
                        ft::TYPE_INT32,
                        std::vector<size_t>{request_batch_size, beam_width, (size_t)total_output_len},
                        get_ptr<int>(output_ids_buf)}},
            {"logits_buf",
             ft::Tensor{ft::MEMORY_GPU,
                        ft::TYPE_FP32,
                        std::vector<size_t>{request_batch_size, beam_width, vocab_size_},
                        get_ptr<float>(logits_buf)}},
            {"parent_ids",
             ft::Tensor{ft::MEMORY_GPU,
                        ft::TYPE_INT32,
                        std::vector<size_t>{(size_t)total_output_len, request_batch_size, beam_width},
                        get_ptr<int>(parent_ids)}},
            {"sequence_length",
             ft::Tensor{ft::MEMORY_GPU,
                        ft::TYPE_INT32,
                        std::vector<size_t>{request_batch_size, beam_width},
                        get_ptr<int>(sequence_lengths)}}};

        if (return_cum_log_probs > 0) {
            output_tensors.insert({"cum_log_probs",
                                   ft::Tensor{ft::MEMORY_GPU,
                                              ft::TYPE_FP32,
                                              std::vector<size_t>{request_batch_size, beam_width},
                                              get_ptr<float>(cum_log_probs)}});
        }

        try {
            glm->forward(&output_tensors, &input_tensors, &glm_weights_);
        }
        catch (std::runtime_error& error) {
            std::cout << error.what() << std::endl;
            ft::FT_CHECK(false);
        }
        catch (...) {
            std::cout << "Runtime error" << std::endl;
            ft::FT_CHECK(false);
        }

    }

    void encode(th::Tensor& input_ids,
                 th::Tensor& input_lengths,
                 th::Tensor& output_ids,
                 th::Tensor& output_ids_buf,
                 th::Tensor& logits_buf,
                 th::Tensor& parent_ids,
                 th::Tensor& sequence_lengths,
                 th::Tensor& cum_log_probs,
                 const size_t request_output_len,
                 const size_t beam_width,
                 const size_t top_k,
                 const float top_p,
                 const float beam_search_diversity_rate,
                 const float temperature,
                 const float len_penalty,
                 const float repetition_penalty,
                 const unsigned long long int query_random_seed,
                 const int return_cum_log_probs = 0) override
    {
        auto stream = at::cuda::getCurrentCUDAStream().stream();
        cublasHandle_t cublasHandle = at::cuda::getCurrentCUDABlasHandle();
        cublasSetStream(cublasHandle, stream);
        allocator = new ft::Allocator<ft::AllocatorType::TH>();
        cublas_wrapper = new ft::cublasMMWrapper(
            cublasHandle, cublasltHandle_, stream, cublas_algo_map_, cublas_wrapper_mutex_, allocator);

        if (std::is_same<T, half>::value) {
            cublas_wrapper->setGemmConfig(CUDA_R_16F, CUDA_R_16F, CUDA_R_16F, CUDA_R_32F);
        }
        else if (std::is_same<T, float>::value) {
            cublas_wrapper->setFP32GemmConfig();
        }

        request_batch_size = (size_t)input_ids.size(0);
        max_input_length = (size_t)input_ids.size(1);
        total_output_len = (int)(max_input_length + request_output_len);

        ft::NcclParam tensor_para(tensor_para_rank_, tensor_para_size_, tensor_para_comm_);
        ft::NcclParam pipeline_para(pipeline_para_rank_, pipeline_para_size_, pipeline_para_comm_);

        random_seed = query_random_seed;

        glm = new ft::Glm<T>(0,  // max_batch_size, FT will adjust the buffer automatically.
                          0,  // max_seq_len, FT will adjust the buffer automatically.
                          0,  // max_input_len, FT will adjust the buffer automatically.
                          beam_width,
                          head_num_,
                          size_per_head_,
                          inter_size_,
                          layer_num_,
                          vocab_size_,
                          -rotary_embedding_dim_,
                          start_id_,
                          end_id_,
                          0.0f,
                          top_k,
                          top_p,
                          random_seed,
                          temperature,
                          len_penalty,
                          repetition_penalty,
                          tensor_para,
                          pipeline_para,
                          stream,
                          cublas_wrapper,
                          allocator,
                          false,
                          &prop_);
        
        input_tensors = std::unordered_map<std::string, ft::Tensor>{
            {"input_ids",
             ft::Tensor{ft::MEMORY_GPU,
                        ft::TYPE_INT32,
                        std::vector<size_t>{request_batch_size, max_input_length},
                        get_ptr<int>(input_ids)}},
            {"input_lengths",
             ft::Tensor{
                 ft::MEMORY_GPU, ft::TYPE_INT32, std::vector<size_t>{request_batch_size}, get_ptr<int>(input_lengths)}},
            {"max_output_seq_len",
             ft::Tensor{ft::MEMORY_CPU, ft::TYPE_INT32, std::vector<size_t>{1}, &total_output_len}}};

        if (top_k == 0 && top_p == 0.0f) {
            ft::FT_CHECK(beam_width > 1);
            input_tensors.insert(
                {"beam_search_diversity_rate",
                 ft::Tensor{ft::MEMORY_CPU, ft::TYPE_FP32, std::vector<size_t>{1}, &beam_search_diversity_rate}});
        }
        else {
            if (top_p != 0.0f) {
                input_tensors.insert(
                    {"runtime_top_p", ft::Tensor{ft::MEMORY_CPU, ft::TYPE_FP32, std::vector<size_t>{1}, &top_p}});
            }
            if (top_k != 0) {
                input_tensors.insert(
                    {"runtime_top_k", ft::Tensor{ft::MEMORY_CPU, ft::TYPE_INT32, std::vector<size_t>{1}, &top_k}});
            }
        }
        input_tensors.insert(
            {"temperature", ft::Tensor{ft::MEMORY_CPU, ft::TYPE_FP32, std::vector<size_t>{1}, &temperature}});
        input_tensors.insert(
            {"len_penalty", ft::Tensor{ft::MEMORY_CPU, ft::TYPE_FP32, std::vector<size_t>{1}, &len_penalty}});
        input_tensors.insert({"repetition_penalty",
                              ft::Tensor{ft::MEMORY_CPU, ft::TYPE_FP32, std::vector<size_t>{1}, &repetition_penalty}});
        input_tensors.insert(
            {"random_seed", ft::Tensor{ft::MEMORY_CPU, ft::TYPE_UINT64, std::vector<size_t>{1}, &random_seed}});
        
        return_context_cum_log_probs = false;
        if (return_cum_log_probs == 2) {
            return_context_cum_log_probs = true;
            input_tensors.insert(
                {"is_return_context_cum_log_probs",
                 ft::Tensor{ft::MEMORY_CPU, ft::TYPE_BOOL, std::vector<size_t>{1}, &return_context_cum_log_probs}});
        }

        output_tensors = std::unordered_map<std::string, ft::Tensor>{
            {"output_ids",
             ft::Tensor{ft::MEMORY_GPU,
                        ft::TYPE_INT32,
                        std::vector<size_t>{request_batch_size, beam_width, (size_t)total_output_len},
                        get_ptr<int>(output_ids)}},
            {"output_ids_buf",
             ft::Tensor{ft::MEMORY_GPU,
                        ft::TYPE_INT32,
                        std::vector<size_t>{request_batch_size, beam_width, (size_t)total_output_len},
                        get_ptr<int>(output_ids_buf)}},
            {"logits_buf",
             ft::Tensor{ft::MEMORY_GPU,
                        ft::TYPE_FP32,
                        std::vector<size_t>{request_batch_size, beam_width, vocab_size_},
                        get_ptr<float>(logits_buf)}},
            {"parent_ids",
             ft::Tensor{ft::MEMORY_GPU,
                        ft::TYPE_INT32,
                        std::vector<size_t>{(size_t)total_output_len, request_batch_size, beam_width},
                        get_ptr<int>(parent_ids)}},
            {"sequence_length",
             ft::Tensor{ft::MEMORY_GPU,
                        ft::TYPE_INT32,
                        std::vector<size_t>{request_batch_size, beam_width},
                        get_ptr<int>(sequence_lengths)}}};

        if (return_cum_log_probs > 0) {
            output_tensors.insert({"cum_log_probs",
                                   ft::Tensor{ft::MEMORY_GPU,
                                              ft::TYPE_FP32,
                                              std::vector<size_t>{request_batch_size, beam_width},
                                              get_ptr<float>(cum_log_probs)}});
        }

        try {
            glm->encode(&output_tensors, &input_tensors, &glm_weights_);
        }
        catch (std::runtime_error& error) {
            std::cout << error.what() << std::endl;
            ft::FT_CHECK(false);
        }
        catch (...) {
            std::cout << "Runtime error" << std::endl;
            ft::FT_CHECK(false);
        }

    }

    void decode(const size_t step) override
    {
        try {
            glm->decode(&output_tensors, &input_tensors, &glm_weights_, step, false);
        }
        catch (std::runtime_error& error) {
            std::cout << error.what() << std::endl;
            ft::FT_CHECK(false);
        }
        catch (...) {
            std::cout << "Runtime error" << std::endl;
            ft::FT_CHECK(false);
        }

    }
    

private:
    const size_t head_num_;
    const size_t size_per_head_;
    const size_t inter_size_;
    const size_t layer_num_;
    const size_t vocab_size_;
    const size_t rotary_embedding_dim_;
    const int start_id_;
    const int end_id_;

    size_t tensor_para_size_;
    size_t pipeline_para_size_;

    std::vector<th::Tensor> weights_;

    size_t tensor_para_rank_;
    ncclComm_t tensor_para_comm_;
    size_t pipeline_para_rank_;
    ncclComm_t pipeline_para_comm_;

    cublasLtHandle_t cublasltHandle_;
    std::mutex* cublas_wrapper_mutex_;
    ft::cublasAlgoMap* cublas_algo_map_;
    struct cudaDeviceProp prop_;
    ft::GlmWeight<T> glm_weights_;
    int world_size_ = 1;
    int rank_ = 0;
    HackNCCLGroup* h_;

    ft::cublasMMWrapper* cublas_wrapper;
    ft::Allocator<ft::AllocatorType::TH>* allocator;



    ft::Glm<T>* glm;
    size_t request_batch_size;
    size_t max_input_length;
    int total_output_len;
    unsigned long long int random_seed;

    std::unordered_map<std::string, ft::Tensor> input_tensors;
    std::unordered_map<std::string, ft::Tensor> output_tensors;
    bool return_context_cum_log_probs;
};


class GlmOp: public th::jit::CustomClassHolder {
public:
    GlmOp(const c10d::ProcessGroupNCCL& p,
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
                  const vector<th::Tensor> weights);

    ~GlmOp();

    vector<th::Tensor> forward(th::Tensor input_ids,
                               th::Tensor input_lengths,
                               const int64_t output_len,
                               const int64_t beam_width,
                               const int64_t top_k,
                               const double top_p,
                               const double beam_search_diversity_rate,
                               const double temperature,
                               const double len_penalty,
                               const double repetition_penalty,
                               const int64_t random_seed,
                               const int64_t return_cum_log_probs);
    
    std::vector<th::Tensor> encode(th::Tensor input_ids,
                                    th::Tensor input_lengths,
                                    th::Tensor output_ids_buf,
                                    th::Tensor logits_buf,
                                    th::Tensor output_ids,
                                    th::Tensor parent_ids,
                                    th::Tensor sequence_lengths,
                                    th::Tensor cum_log_probs,
                                    const int64_t output_len,
                                    const int64_t beam_width,
                                    const int64_t top_k,
                                    const double top_p,
                                    const double beam_search_diversity_rate,
                                    const double temperature,
                                    const double len_penalty,
                                    const double repetition_penalty,
                                    const int64_t random_seed,
                                    const int64_t return_cum_log_probs);
    
    std::vector<th::Tensor> decode(const int64_t step);

private:
    const at::ScalarType st_;
    const int64_t vocab_size_;
    IFGlm* ftglm;
    HackNCCLGroup* h;
    std::vector<th::Tensor> weights;
};

}  // namespace torch_ext
