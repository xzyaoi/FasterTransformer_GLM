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

#pragma once

#include "src/fastertransformer/kernels/activation_kernels.h"
#include "src/fastertransformer/kernels/matrix_vector_multiplication.h"
#include "src/fastertransformer/layers/BaseLayer.h"
#include "src/fastertransformer/layers/GluFfnWeight.h"
#include "src/fastertransformer/utils/memory_utils.h"
#include <vector>

namespace fastertransformer {

template<typename T>
class GluFfnLayer: public BaseLayer {
private:
    // buffer handling
    size_t max_token_num_ = 0;

    // meta data
    size_t head_num_;
    size_t size_per_head_;

    // int8_mode_ == 1 for weight quantized only gemm for GPT
    int int8_mode_ = 0;

    // calculated data
    size_t hidden_units_;

    void allocateBuffer() override;
    void freeBuffer() override;
    bool isValidTokenNum(size_t token_num);
    void allocateBuffer(size_t token_num);

protected:
    T* inter_buf_[2] = {nullptr, nullptr};
    size_t inter_size_;
    virtual void invokeAddBiasActivation(const int m, const GluFfnWeight<T>* glu_ffn_weights) = 0;

public:
    GluFfnLayer(size_t max_batch_size,
             size_t max_seq_len,
             size_t head_num,
             size_t size_per_head,
             size_t inter_size,
             cudaStream_t stream,
             cublasMMWrapper* cublas_wrapper,
             IAllocator* allocator,
             bool is_free_buffer_after_forward,
             bool sparse = false,
             int int8_mode = 0);

    GluFfnLayer(GluFfnLayer<T> const& glu_ffn_layer);

    virtual ~GluFfnLayer();

    virtual void forward(std::vector<fastertransformer::Tensor>* output_tensors,
                         const std::vector<fastertransformer::Tensor>* input_tensors,
                         const GluFfnWeight<T>* glu_ffn_weights);
};

template<typename T>
class GeluGluFfnLayer: public GluFfnLayer<T> {
public:
    GeluGluFfnLayer(size_t max_batch_size,
                 size_t max_seq_len,
                 size_t head_num,
                 size_t size_per_head,
                 size_t inter_size,
                 cudaStream_t stream,
                 cublasMMWrapper* cublas_wrapper,
                 IAllocator* allocator,
                 bool is_free_buffer_after_forward,
                 bool sparse = false,
                 int int8_mode = 0);

    GeluGluFfnLayer(GeluGluFfnLayer<T> const& glu_ffn_layer);

    virtual ~GeluGluFfnLayer() = default;

protected:
    using GluFfnLayer<T>::stream_;

private:
    using GluFfnLayer<T>::inter_buf_;
    using GluFfnLayer<T>::inter_size_;
    void invokeAddBiasActivation(const int m, const GluFfnWeight<T>* glu_ffn_weights) override;
};

template<typename T>
class ReluGluFfnLayer: public GluFfnLayer<T> {
public:
    ReluGluFfnLayer(size_t max_batch_size,
                 size_t max_seq_len,
                 size_t head_num,
                 size_t size_per_head,
                 size_t inter_size,
                 cudaStream_t stream,
                 cublasMMWrapper* cublas_wrapper,
                 IAllocator* allocator,
                 bool is_free_buffer_after_forward,
                 bool sparse = false);

    ReluGluFfnLayer(ReluGluFfnLayer<T> const& glu_ffn_layer);

    virtual ~ReluGluFfnLayer() = default;

protected:
    using GluFfnLayer<T>::stream_;

private:
    using GluFfnLayer<T>::inter_buf_;
    using GluFfnLayer<T>::inter_size_;
    void invokeAddBiasActivation(const int m, const GluFfnWeight<T>* glu_ffn_weights) override;
};

}  // namespace fastertransformer
