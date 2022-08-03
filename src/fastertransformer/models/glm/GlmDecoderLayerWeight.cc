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

#include "src/fastertransformer/utils/mpi_utils.h"
#include "src/fastertransformer/models/glm/GlmDecoderLayerWeight.h"
#include "src/fastertransformer/utils/memory_utils.h"

namespace fastertransformer {

template<typename T>
GlmDecoderLayerWeight<T>::GlmDecoderLayerWeight(const int hidden_units,
                                                  const int inter_size,
                                                  const int tensor_para_size,
                                                  const int tensor_para_rank):
    hidden_units_(hidden_units),
    inter_size_(inter_size),
    tensor_para_size_(tensor_para_size),
    tensor_para_rank_(tensor_para_rank)
{
    // int rank;
    //     MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    //     if(rank==0) {
    //         size_t free, total;
    //         cudaMemGetInfo( &free, &total );
    // printf("GlmDecoderLayerWeight memory: free=%f, total=%f\n", double(free)/1024/1024, double(total)/1024/1024);
    //     }
        
    mallocWeights();
    setWeightPtr();
    // if(rank==0) {
    //         size_t free, total;
    //         cudaMemGetInfo( &free, &total );
    // printf("After GlmDecoderLayerWeight memory: free=%f, total=%f\n", double(free)/1024/1024, double(total)/1024/1024);
    //     }
    
}

template<typename T>
GlmDecoderLayerWeight<T>::~GlmDecoderLayerWeight()
{
    if (is_maintain_buffer == true) {
        for (int i = 0; i < 14; i++) {
            deviceFree(weights_ptr[i]);
        }

        self_attention_weights.query_weight.kernel = nullptr;
        self_attention_weights.query_weight.bias = nullptr;
        self_attention_weights.attention_output_weight.kernel = nullptr;
        self_attention_weights.attention_output_weight.bias = nullptr;
        self_attn_layernorm_weights.beta = nullptr;
        self_attn_layernorm_weights.gamma = nullptr;
        glu_ffn_weights.intermediate_weight[0].kernel = nullptr;
        glu_ffn_weights.intermediate_weight[0].bias = nullptr;
        glu_ffn_weights.intermediate_weight[1].kernel = nullptr;
        glu_ffn_weights.intermediate_weight[1].bias = nullptr;
        glu_ffn_weights.output_weight.kernel = nullptr;
        glu_ffn_weights.output_weight.bias = nullptr;
        glu_ffn_layernorm_weights.beta = nullptr;
        glu_ffn_layernorm_weights.gamma = nullptr;
        is_maintain_buffer = false;
    }
}

template<typename T>
GlmDecoderLayerWeight<T>::GlmDecoderLayerWeight(const GlmDecoderLayerWeight& other):
    hidden_units_(other.hidden_units_),
    inter_size_(other.inter_size_),
    tensor_para_size_(other.tensor_para_size_),
    tensor_para_rank_(other.tensor_para_rank_)
{
    mallocWeights();

    cudaD2Dcpy(weights_ptr[0], other.weights_ptr[0], hidden_units_ * 3 * hidden_units_ / tensor_para_size_);
    cudaD2Dcpy(weights_ptr[1], other.weights_ptr[1], 3 * hidden_units_ / tensor_para_size_);
    cudaD2Dcpy(weights_ptr[2], other.weights_ptr[2], hidden_units_ / tensor_para_size_ * hidden_units_);
    cudaD2Dcpy(weights_ptr[3], other.weights_ptr[3], hidden_units_);
    cudaD2Dcpy(weights_ptr[4], other.weights_ptr[4], hidden_units_);
    cudaD2Dcpy(weights_ptr[5], other.weights_ptr[5], hidden_units_);
    cudaD2Dcpy(weights_ptr[6], other.weights_ptr[6], hidden_units_ * inter_size_ / tensor_para_size_);
    cudaD2Dcpy(weights_ptr[7], other.weights_ptr[7], inter_size_ / tensor_para_size_);
    cudaD2Dcpy(weights_ptr[8], other.weights_ptr[8], hidden_units_ * inter_size_ / tensor_para_size_);
    cudaD2Dcpy(weights_ptr[9], other.weights_ptr[9], inter_size_ / tensor_para_size_);
    cudaD2Dcpy(weights_ptr[10], other.weights_ptr[10], inter_size_ / tensor_para_size_ * hidden_units_);
    cudaD2Dcpy(weights_ptr[11], other.weights_ptr[11], hidden_units_);
    cudaD2Dcpy(weights_ptr[12], other.weights_ptr[12], hidden_units_);
    cudaD2Dcpy(weights_ptr[13], other.weights_ptr[13], hidden_units_);

    setWeightPtr();
}

template<typename T>
GlmDecoderLayerWeight<T>& GlmDecoderLayerWeight<T>::operator=(const GlmDecoderLayerWeight& other)
{
    hidden_units_ = other.hidden_units_;
    inter_size_ = other.inter_size_;
    tensor_para_size_ = other.tensor_para_size_;
    tensor_para_rank_ = other.tensor_para_rank_;

    mallocWeights();

    cudaD2Dcpy(weights_ptr[0], other.weights_ptr[0], hidden_units_ * 3 * hidden_units_ / tensor_para_size_);
    cudaD2Dcpy(weights_ptr[1], other.weights_ptr[1], 3 * hidden_units_ / tensor_para_size_);
    cudaD2Dcpy(weights_ptr[2], other.weights_ptr[2], hidden_units_ / tensor_para_size_ * hidden_units_);
    cudaD2Dcpy(weights_ptr[3], other.weights_ptr[3], hidden_units_);
    cudaD2Dcpy(weights_ptr[4], other.weights_ptr[4], hidden_units_);
    cudaD2Dcpy(weights_ptr[5], other.weights_ptr[5], hidden_units_);
    cudaD2Dcpy(weights_ptr[6], other.weights_ptr[6], hidden_units_ * inter_size_ / tensor_para_size_);
    cudaD2Dcpy(weights_ptr[7], other.weights_ptr[7], inter_size_ / tensor_para_size_);
    cudaD2Dcpy(weights_ptr[8], other.weights_ptr[8], hidden_units_ * inter_size_ / tensor_para_size_);
    cudaD2Dcpy(weights_ptr[9], other.weights_ptr[9], inter_size_ / tensor_para_size_);
    cudaD2Dcpy(weights_ptr[10], other.weights_ptr[10], inter_size_ / tensor_para_size_ * hidden_units_);
    cudaD2Dcpy(weights_ptr[11], other.weights_ptr[11], hidden_units_);
    cudaD2Dcpy(weights_ptr[12], other.weights_ptr[12], hidden_units_);
    cudaD2Dcpy(weights_ptr[13], other.weights_ptr[13], hidden_units_);

    setWeightPtr();
    return *this;
}

template<typename T>
void GlmDecoderLayerWeight<T>::loadModel(std::string dir_path, FtCudaDataType model_file_type)
{
    FT_CHECK(is_maintain_buffer == true);
    const std::string rank_spec = std::to_string(tensor_para_rank_);
    
    // GPT-J does not have bias for QKV
    // cudaMemset(weights_ptr[3], 0, sizeof(T) * 3 * hidden_units_ / tensor_para_size_);
    loadWeightFromBin<T>(weights_ptr[0], {hidden_units_, 3 * hidden_units_ / tensor_para_size_}, dir_path + ".attention.query_key_value.weight." + rank_spec + ".bin", model_file_type);
    loadWeightFromBin<T>(weights_ptr[1], {3, hidden_units_ / tensor_para_size_}, dir_path + ".attention.query_key_value.bias." + rank_spec + ".bin", model_file_type);
    loadWeightFromBin<T>(weights_ptr[2],
                         {hidden_units_ / tensor_para_size_, hidden_units_},
                         dir_path + ".attention.dense.weight." + rank_spec + ".bin",
                         model_file_type);
    loadWeightFromBin<T>(weights_ptr[3],
                         {hidden_units_},
                         dir_path + ".attention.dense.bias." + rank_spec + ".bin",
                         model_file_type);
    loadWeightFromBin<T>(weights_ptr[4],
                         {hidden_units_},
                         dir_path + ".input_layernorm.bias." + rank_spec + ".bin",
                         model_file_type);
    loadWeightFromBin<T>(weights_ptr[5],
                         {hidden_units_},
                         dir_path + ".input_layernorm.weight." + rank_spec + ".bin",
                         model_file_type);
    loadWeightFromBin<T>(weights_ptr[6],
                         {hidden_units_, inter_size_ / tensor_para_size_},
                         dir_path + ".mlp.dense_h_to_4h.weight." + rank_spec + ".1.bin",
                         model_file_type);
    loadWeightFromBin<T>(weights_ptr[7],
                         {inter_size_ / tensor_para_size_},
                         dir_path + ".mlp.dense_h_to_4h.bias." + rank_spec + ".1.bin",
                         model_file_type);
    loadWeightFromBin<T>(weights_ptr[8],
                         {hidden_units_, inter_size_ / tensor_para_size_},
                         dir_path + ".mlp.dense_h_to_4h.weight." + rank_spec + ".2.bin",
                         model_file_type);
    loadWeightFromBin<T>(weights_ptr[9],
                         {inter_size_ / tensor_para_size_},
                         dir_path + ".mlp.dense_h_to_4h.bias." + rank_spec + ".2.bin",
                         model_file_type);
    loadWeightFromBin<T>(weights_ptr[10],
                         {inter_size_ / tensor_para_size_, hidden_units_},
                         dir_path + ".mlp.dense_4h_to_h.weight." + rank_spec + ".bin",
                         model_file_type);
    loadWeightFromBin<T>(weights_ptr[11],
                         {hidden_units_},
                         dir_path + ".mlp.dense_4h_to_h.bias." + rank_spec + ".bin",
                         model_file_type);
    loadWeightFromBin<T>(weights_ptr[12],
                         {hidden_units_},
                         dir_path + ".post_attention_layernorm.bias." + rank_spec + ".bin",
                         model_file_type);
    loadWeightFromBin<T>(weights_ptr[13],
                         {hidden_units_},
                         dir_path + ".post_attention_layernorm.weight." + rank_spec + ".bin",
                         model_file_type);
}

template<typename T>
void GlmDecoderLayerWeight<T>::setWeightPtr()
{
    // QKV
    self_attention_weights.query_weight.kernel = weights_ptr[0]; // hidden_units_ * 3 * hidden_units_ / tensor_para_size_
    self_attention_weights.query_weight.bias = weights_ptr[1];   // 3 * hidden_units_ / tensor_para_size_
    self_attention_weights.attention_output_weight.kernel = weights_ptr[2]; // hidden_units_ / tensor_para_size_ * hidden_units_
    self_attention_weights.attention_output_weight.bias = weights_ptr[3];   // hidden_units_
    self_attn_layernorm_weights.beta = weights_ptr[4];               // hidden_units_
    self_attn_layernorm_weights.gamma = weights_ptr[5];              // hidden_units_
    glu_ffn_weights.intermediate_weight[0].kernel = weights_ptr[6];  // hidden_units_ * inter_size_ / tensor_para_size_
    glu_ffn_weights.intermediate_weight[0].bias = weights_ptr[7];    // inter_size_ / tensor_para_size_
    glu_ffn_weights.intermediate_weight[1].kernel = weights_ptr[8];  // hidden_units_ * inter_size_ / tensor_para_size_
    glu_ffn_weights.intermediate_weight[1].bias = weights_ptr[9];    // inter_size_ / tensor_para_size_
    glu_ffn_weights.output_weight.kernel = weights_ptr[10];          // inter_size_ / tensor_para_size_ * hidden_units_
    glu_ffn_weights.output_weight.bias = weights_ptr[11];            // hidden_units_
    glu_ffn_layernorm_weights.beta = weights_ptr[12];                // hidden_units_
    glu_ffn_layernorm_weights.gamma = weights_ptr[13];               // hidden_units_

    is_maintain_buffer = true;
}

template<typename T>
void GlmDecoderLayerWeight<T>::mallocWeights()
{
    deviceMalloc(&weights_ptr[0], hidden_units_ * 3 * hidden_units_ / tensor_para_size_);
    deviceMalloc(&weights_ptr[1], 3 * hidden_units_ / tensor_para_size_);
    deviceMalloc(&weights_ptr[2], hidden_units_ / tensor_para_size_ * hidden_units_);
    deviceMalloc(&weights_ptr[3], hidden_units_);
    deviceMalloc(&weights_ptr[4], hidden_units_);
    deviceMalloc(&weights_ptr[5], hidden_units_);
    deviceMalloc(&weights_ptr[6], hidden_units_ * inter_size_ / tensor_para_size_);
    deviceMalloc(&weights_ptr[7], inter_size_ / tensor_para_size_);
    deviceMalloc(&weights_ptr[8], hidden_units_ * inter_size_ / tensor_para_size_);
    deviceMalloc(&weights_ptr[9], inter_size_ / tensor_para_size_);
    deviceMalloc(&weights_ptr[10], inter_size_ / tensor_para_size_ * hidden_units_);
    deviceMalloc(&weights_ptr[11], hidden_units_);
    deviceMalloc(&weights_ptr[12], hidden_units_);
    deviceMalloc(&weights_ptr[13], hidden_units_);
}

template struct GlmDecoderLayerWeight<float>;
template struct GlmDecoderLayerWeight<half>;

}  // namespace fastertransformer
