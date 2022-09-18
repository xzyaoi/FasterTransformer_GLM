/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.  All rights reserved.
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

#include "src/fastertransformer/layers/GluFfnLayer.h"

namespace fastertransformer {

template<typename T>
void GluFfnLayer<T>::forward(std::vector<fastertransformer::Tensor>* output_tensors,
                          const std::vector<fastertransformer::Tensor>* input_tensors,
                          const GluFfnWeight<T>* glu_ffn_weights)
{
    // input tensors:
    //      ffn_input [token_num, hidden_dimension],

    // output tensors:
    //      ffn_output [token_num, hidden_dimension],

    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    FT_CHECK(input_tensors->size() == 1);
    FT_CHECK(output_tensors->size() == 1);
    // FT_CHECK(isValidTokenNum(input_tensors->at(0).shape[0]));
    allocateBuffer(input_tensors->at(0).shape[0]);

    const int m = input_tensors->at(0).shape[0];
    T* output_tensor = (T*)output_tensors->at(0).data;
    const T* input_tensor = (const T*)input_tensors->at(0).data;

    for(int inter_buf_id = 0; inter_buf_id < 2; inter_buf_id++) {
#ifdef SPARSITY_ENABLED
        int m_tmp = input_tensors->at(0).shape[0];
        if (m_tmp % 8 != 0) {
            m_tmp = (m_tmp / 8 + 1) * 8;
        }
        const int m_padded = m_tmp;
        if (sparse_ && cublas_wrapper_->isUseSparse(1, inter_size_, m, hidden_units_)) {
            cublas_wrapper_->SpGemm(CUBLAS_OP_N,
                                    CUBLAS_OP_N,
                                    inter_size_,
                                    m_padded,
                                    hidden_units_,
                                    glu_ffn_weights->intermediate_weight[inter_buf_id].sp_kernel,
                                    input_tensor,
                                    inter_buf_[inter_buf_id]);
        }
        else {
#endif
            if (int8_mode_ == 1 && m <= 2) {
                FT_CHECK(glu_ffn_weights->intermediate_weight[inter_buf_id].int8_kernel != NULL
                        && glu_ffn_weights->intermediate_weight[inter_buf_id].scale != NULL);
                int8WeightPerChannelLdkMultiplicationLauncher(glu_ffn_weights->intermediate_weight[inter_buf_id].int8_kernel,
                                                            input_tensor,
                                                            glu_ffn_weights->intermediate_weight[inter_buf_id].scale,
                                                            inter_buf_[inter_buf_id],
                                                            m,
                                                            inter_size_,
                                                            hidden_units_,
                                                            stream_);
            }
            else {
                if (int8_mode_ == 1) {
                    printf("[WARNING][GluFfnLayer<T>::forward] int8 gpt doesn't support m > 2, run fp gpt instead.\n");
                }
                if (glu_ffn_weights->intermediate_weight[inter_buf_id].kernel) {
                    cublas_wrapper_->Gemm(CUBLAS_OP_N,
                                    CUBLAS_OP_N,
                                    inter_size_,
                                    m,
                                    hidden_units_,
                                    glu_ffn_weights->intermediate_weight[inter_buf_id].kernel,
                                    inter_size_,
                                    input_tensor,
                                    hidden_units_,
                                    inter_buf_[inter_buf_id],
                                    inter_size_);
                } else if(m > 2) {
                    FT_CHECK(glu_ffn_weights->intermediate_weight[inter_buf_id].int8_kernel != NULL
                            && glu_ffn_weights->intermediate_weight[inter_buf_id].scale != NULL);
                    invokeInt4WeightExtraction(glu_ffn_weights->intermediate_weight[inter_buf_id].int8_kernel,
                                                glu_ffn_weights->intermediate_weight[inter_buf_id].scale,
                                                weights_buf_,
                                                inter_size_,
                                                hidden_units_ / 2,
                                                stream_);
                    sync_check_cuda_error();
                    cublas_wrapper_->Gemm(CUBLAS_OP_N,
                                    CUBLAS_OP_N,
                                    inter_size_,
                                    m,
                                    hidden_units_,
                                    weights_buf_,
                                    inter_size_,
                                    input_tensor,
                                    hidden_units_,
                                    inter_buf_[inter_buf_id],
                                    inter_size_);
                } else {
                    int4WeightPerChannelLdkMultiplicationLauncher(glu_ffn_weights->intermediate_weight[inter_buf_id].int8_kernel,
                                                            input_tensor,
                                                            glu_ffn_weights->intermediate_weight[inter_buf_id].scale,
                                                            inter_buf_[inter_buf_id],
                                                            m,
                                                            inter_size_,
                                                            hidden_units_ / 2,
                                                            stream_);
                }
                
            }
#ifdef SPARSITY_ENABLED
        }
#endif
    }

    invokeAddBiasActivation(m, glu_ffn_weights);
    sync_check_cuda_error();

#ifdef SPARSITY_ENABLED
    if (sparse_ && cublas_wrapper_->isUseSparse(1, hidden_units_, m, inter_size_)) {
        cublas_wrapper_->SpGemm(CUBLAS_OP_N,
                                CUBLAS_OP_N,
                                hidden_units_,
                                m_padded,
                                inter_size_,
                                glu_ffn_weights->output_weight.sp_kernel,
                                inter_buf_[0],
                                output_tensor);
    }
    else {
#endif
        if (int8_mode_ == 1 && m <= 2) {
            FT_CHECK(glu_ffn_weights->output_weight.int8_kernel != NULL && glu_ffn_weights->output_weight.scale != NULL);
            int8WeightPerChannelLdkMultiplicationLauncher(glu_ffn_weights->output_weight.int8_kernel,
                                                          inter_buf_[0],
                                                          glu_ffn_weights->output_weight.scale,
                                                          output_tensor,
                                                          m,
                                                          hidden_units_,
                                                          inter_size_,
                                                          stream_);
        }
        else {
            if(glu_ffn_weights->output_weight.kernel) {
                cublas_wrapper_->Gemm(CUBLAS_OP_N,
                                  CUBLAS_OP_N,
                                  hidden_units_,
                                  m,
                                  inter_size_,
                                  glu_ffn_weights->output_weight.kernel,
                                  hidden_units_,
                                  inter_buf_[0],
                                  inter_size_,
                                  output_tensor,
                                  hidden_units_);
            } else {
                FT_CHECK(glu_ffn_weights->output_weight.int8_kernel != NULL
                            && glu_ffn_weights->output_weight.scale != NULL);
                    invokeInt4WeightExtraction(glu_ffn_weights->output_weight.int8_kernel,
                                                glu_ffn_weights->output_weight.scale,
                                                weights_buf_,
                                                hidden_units_,
                                                inter_size_ / 2,
                                                stream_);
                    sync_check_cuda_error();
                    cublas_wrapper_->Gemm(CUBLAS_OP_N,
                                  CUBLAS_OP_N,
                                  hidden_units_,
                                  m,
                                  inter_size_,
                                  weights_buf_,
                                  hidden_units_,
                                  inter_buf_[0],
                                  inter_size_,
                                  output_tensor,
                                  hidden_units_);
            }
            
        }
#ifdef SPARSITY_ENABLED
    }
#endif
    sync_check_cuda_error();
    if (is_free_buffer_after_forward_ == true) {
        freeBuffer();
    }
    sync_check_cuda_error();
}

template<typename T>
GluFfnLayer<T>::GluFfnLayer(size_t max_batch_size,
                      size_t max_seq_len,
                      size_t head_num,
                      size_t size_per_head,
                      size_t inter_size,
                      cudaStream_t stream,
                      cublasMMWrapper* cublas_wrapper,
                      IAllocator* allocator,
                      bool is_free_buffer_after_forward,
                      bool sparse,
                      int int8_mode):
    BaseLayer(stream, cublas_wrapper, allocator, is_free_buffer_after_forward, nullptr, sparse),
    max_token_num_(max_batch_size * max_seq_len),
    head_num_(head_num),
    size_per_head_(size_per_head),
    hidden_units_(head_num * size_per_head),
    inter_size_(inter_size),
    int8_mode_(int8_mode)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
}

template<typename T>
GluFfnLayer<T>::GluFfnLayer(GluFfnLayer<T> const& glu_ffn_layer):
    BaseLayer(glu_ffn_layer.stream_,
              glu_ffn_layer.cublas_wrapper_,
              glu_ffn_layer.allocator_,
              glu_ffn_layer.is_free_buffer_after_forward_,
              glu_ffn_layer.cuda_device_prop_,
              glu_ffn_layer.sparse_),
    max_token_num_(glu_ffn_layer.max_token_num_),
    head_num_(glu_ffn_layer.head_num_),
    size_per_head_(glu_ffn_layer.size_per_head_),
    hidden_units_(glu_ffn_layer.hidden_units_),
    inter_size_(glu_ffn_layer.inter_size_),
    int8_mode_(glu_ffn_layer.int8_mode_)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
}

template<typename T>
GluFfnLayer<T>::~GluFfnLayer()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    cublas_wrapper_ = nullptr;
    freeBuffer();
}

template<typename T>
void GluFfnLayer<T>::allocateBuffer()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (is_allocate_buffer_ == false) {
        inter_buf_[0] = (T*)allocator_->malloc(sizeof(T) * max_token_num_ * inter_size_, false);
        inter_buf_[1] = (T*)allocator_->malloc(sizeof(T) * max_token_num_ * inter_size_, false);
        weights_buf_ = (T*)allocator_->malloc(sizeof(T) * hidden_units_ * inter_size_, false);
        is_allocate_buffer_ = true;
    }
}

template<typename T>
void GluFfnLayer<T>::allocateBuffer(size_t token_num)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    inter_buf_[0] = (T*)allocator_->reMalloc(inter_buf_[0], sizeof(T) * token_num * inter_size_, false);
    inter_buf_[1] = (T*)allocator_->reMalloc(inter_buf_[1], sizeof(T) * token_num * inter_size_, false);
    weights_buf_ = (T*)allocator_->reMalloc(weights_buf_, sizeof(T) * hidden_units_ * inter_size_, false);
    is_allocate_buffer_ = true;
}

template<typename T>
void GluFfnLayer<T>::freeBuffer()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (is_allocate_buffer_) {
        allocator_->free(inter_buf_[0]);
        allocator_->free(inter_buf_[1]);
        allocator_->free(weights_buf_);
        is_allocate_buffer_ = false;
    }
}

template<typename T>
bool GluFfnLayer<T>::isValidTokenNum(size_t token_num)
{
    if (max_token_num_ < token_num) {
        max_token_num_ = token_num;
    }
    return true;
}

template class GluFfnLayer<float>;
template class GluFfnLayer<half>;
#ifdef ENABLE_BF16
template class GluFfnLayer<__nv_bfloat16>;
#endif

template<typename T>
GeluGluFfnLayer<T>::GeluGluFfnLayer(size_t max_batch_size,
                              size_t max_seq_len,
                              size_t head_num,
                              size_t size_per_head,
                              size_t inter_size,
                              cudaStream_t stream,
                              cublasMMWrapper* cublas_wrapper,
                              IAllocator* allocator,
                              bool is_free_buffer_after_forward,
                              bool sparse,
                              int int8_mode):
    GluFfnLayer<T>(max_batch_size,
                max_seq_len,
                head_num,
                size_per_head,
                inter_size,
                stream,
                cublas_wrapper,
                allocator,
                is_free_buffer_after_forward,
                sparse,
                int8_mode)
{
}

template<typename T>
GeluGluFfnLayer<T>::GeluGluFfnLayer(GeluGluFfnLayer<T> const& gelu_glu_ffn_layer): GluFfnLayer<T>(gelu_glu_ffn_layer)
{
}

template<typename T>
void GeluGluFfnLayer<T>::invokeAddBiasActivation(const int m, const GluFfnWeight<T>* glu_ffn_weights)
{
    invokeAddBias<T>(inter_buf_[0], glu_ffn_weights->intermediate_weight[0].bias, m, inter_size_, stream_);
    invokeAddBiasGelu<T>(inter_buf_[1], glu_ffn_weights->intermediate_weight[1].bias, m, inter_size_, stream_);
    invokeDot<T>(inter_buf_[0], inter_buf_[1], m, inter_size_, stream_);
}

template class GeluGluFfnLayer<float>;
template class GeluGluFfnLayer<half>;
#ifdef ENABLE_BF16
template class GeluGluFfnLayer<__nv_bfloat16>;
#endif

template<typename T>
ReluGluFfnLayer<T>::ReluGluFfnLayer(size_t max_batch_size,
                              size_t max_seq_len,
                              size_t head_num,
                              size_t size_per_head,
                              size_t inter_size,
                              cudaStream_t stream,
                              cublasMMWrapper* cublas_wrapper,
                              IAllocator* allocator,
                              bool is_free_buffer_after_forward,
                              bool sparse):
    GluFfnLayer<T>(max_batch_size,
                max_seq_len,
                head_num,
                size_per_head,
                inter_size,
                stream,
                cublas_wrapper,
                allocator,
                is_free_buffer_after_forward,
                sparse)
{
}

template<typename T>
ReluGluFfnLayer<T>::ReluGluFfnLayer(ReluGluFfnLayer<T> const& relu_glu_ffn_layer): GluFfnLayer<T>(relu_glu_ffn_layer)
{
}

template<typename T>
void ReluGluFfnLayer<T>::invokeAddBiasActivation(const int m, const GluFfnWeight<T>* glu_ffn_weights)
{
    invokeAddBias<T>(inter_buf_[0], glu_ffn_weights->intermediate_weight[0].bias, m, inter_size_, stream_);
    invokeAddBiasRelu<T>(inter_buf_[1], glu_ffn_weights->intermediate_weight[1].bias, m, inter_size_, stream_);
    invokeDot<T>(inter_buf_[0], inter_buf_[1], m, inter_size_, stream_);
}

template class ReluGluFfnLayer<float>;
template class ReluGluFfnLayer<half>;
#ifdef ENABLE_BF16
template class ReluGluFfnLayer<__nv_bfloat16>;
#endif

}  // namespace fastertransformer
