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

#include "src/fastertransformer/models/glm/GlmWeight.h"

namespace fastertransformer {

template<typename T>
GlmWeight<T>::GlmWeight(const int hidden_units,
                          const int inter_size,
                          const int vocab_size,
                          const int num_layer,
                          const int max_seq_len,
                          const int tensor_para_size,
                          const int tensor_para_rank,
                          const int layer_para_size,
                          const int layer_para_rank):
    hidden_units_(hidden_units),
    inter_size_(inter_size),
    vocab_size_(vocab_size),
    num_layer_(num_layer),
    max_seq_len_(max_seq_len),
    tensor_para_size_(tensor_para_size),
    tensor_para_rank_(tensor_para_rank),
    layer_para_size_(layer_para_size),
    layer_para_rank_(layer_para_rank)
{
    decoder_layer_weights.reserve(num_layer_);
    for (int l = 0; l < num_layer_; l++) {
        if (isValidLayerParallelId(l)) {
            decoder_layer_weights.push_back(new 
                GlmDecoderLayerWeight<T>(hidden_units_, inter_size_, tensor_para_size_, tensor_para_rank_));
        }
        else {
            // Layer-parallelism: allocate empty layer because
            // this rank does not compute it:
            decoder_layer_weights.push_back(new GlmDecoderLayerWeight<T>(0, 0));
        }
    }

    mallocWeights();
    setWeightPtr();
}

template<typename T>
GlmWeight<T>::~GlmWeight()
{
    if (is_maintain_buffer == true) {
        for (int i = 0; i < 4; i++) {
            deviceFree(weights_ptr[i]);
        }

        pre_decoder_embedding_table = nullptr;
        post_decoder_layernorm.beta = nullptr;
        post_decoder_layernorm.gamma = nullptr;
        post_decoder_embedding.kernel = nullptr;
        post_decoder_embedding.bias = nullptr;
        is_maintain_buffer = false;
    }

    for (int i = 0; i < num_layer_; i++) {
        delete decoder_layer_weights[i];
    }
}

template<typename T>
GlmWeight<T>::GlmWeight(const GlmWeight& other):
    hidden_units_(other.hidden_units_),
    inter_size_(other.inter_size_),
    vocab_size_(other.vocab_size_),
    num_layer_(other.num_layer_),
    max_seq_len_(other.max_seq_len_),
    tensor_para_size_(other.tensor_para_size_),
    tensor_para_rank_(other.tensor_para_rank_),
    layer_para_size_(other.layer_para_size_),
    layer_para_rank_(other.layer_para_rank_)
{
    mallocWeights();
    cudaD2Dcpy(weights_ptr[0], other.weights_ptr[0], vocab_size_ * hidden_units_);
    cudaD2Dcpy(weights_ptr[1], other.weights_ptr[1], hidden_units_);
    cudaD2Dcpy(weights_ptr[2], other.weights_ptr[2], hidden_units_);
    cudaD2Dcpy(weights_ptr[3], other.weights_ptr[3], hidden_units_ * vocab_size_);
    setWeightPtr();

    decoder_layer_weights.clear();
    for (int l = 0; l < num_layer_; l++) {
        decoder_layer_weights.push_back(other.decoder_layer_weights[l]);
    }
}

template<typename T>
GlmWeight<T>& GlmWeight<T>::operator=(const GlmWeight& other)
{
    hidden_units_ = other.hidden_units_;
    inter_size_ = other.inter_size_;
    vocab_size_ = other.vocab_size_;
    num_layer_ = other.num_layer_;
    max_seq_len_ = other.max_seq_len_;
    tensor_para_size_ = other.tensor_para_size_;
    tensor_para_rank_ = other.tensor_para_rank_;
    layer_para_size_ = other.layer_para_size_;
    layer_para_rank_ = other.layer_para_rank_;

    mallocWeights();
    cudaD2Dcpy(weights_ptr[0], other.weights_ptr[0], vocab_size_ * hidden_units_);
    cudaD2Dcpy(weights_ptr[1], other.weights_ptr[1], hidden_units_);
    cudaD2Dcpy(weights_ptr[2], other.weights_ptr[2], hidden_units_);
    cudaD2Dcpy(weights_ptr[3], other.weights_ptr[3], hidden_units_ * vocab_size_);
    setWeightPtr();

    decoder_layer_weights.clear();
    for (int l = 0; l < num_layer_; l++) {
        decoder_layer_weights.push_back(other.decoder_layer_weights[l]);
    }
    return *this;
}

template<typename T>
void GlmWeight<T>::resizeLayer(const int num_layer)
{
    num_layer_ = num_layer;
    for (int l = 0; l < num_layer_; l++) {
        decoder_layer_weights.push_back(new GlmDecoderLayerWeight<T>());
    }
}

template<typename T>
void GlmWeight<T>::setWeightPtr()
{
    pre_decoder_embedding_table = weights_ptr[0];
    post_decoder_layernorm.beta = weights_ptr[1];
    post_decoder_layernorm.gamma = weights_ptr[2];
    post_decoder_embedding.kernel = weights_ptr[3];
    post_decoder_embedding.bias = nullptr;
}

template<typename T>
void GlmWeight<T>::mallocWeights()
{
    deviceMalloc(&weights_ptr[0], vocab_size_ * hidden_units_);
    deviceMalloc(&weights_ptr[1], hidden_units_);
    deviceMalloc(&weights_ptr[2], hidden_units_);
    deviceMalloc(&weights_ptr[3], hidden_units_ * vocab_size_);
    is_maintain_buffer = true;
}

template<typename T>
void GlmWeight<T>::loadModel(std::string dir_path)
{
    // FtCudaDataType model_file_type = getModelFileType(dir_path + "/config.ini");
    FtCudaDataType model_file_type = FtCudaDataType::FP16;  // only support FP32 now
    FT_CHECK(is_maintain_buffer == true);

    const std::string rank_spec = std::to_string(tensor_para_rank_);

    loadWeightFromBin<T>(weights_ptr[0], {vocab_size_ * hidden_units_}, dir_path + "/model.wte.bin", model_file_type);
    loadWeightFromBin<T>(
        weights_ptr[1], {hidden_units_}, dir_path + "/model.final_layernorm.bias." + rank_spec + ".bin", model_file_type);
    loadWeightFromBin<T>(
        weights_ptr[2], {hidden_units_}, dir_path + "/model.final_layernorm.weight." + rank_spec + ".bin", model_file_type);
    loadWeightFromBin<T>(
        weights_ptr[3], {vocab_size_ * hidden_units_}, dir_path + "/model.wte.bin", model_file_type);

    for (int l = 0; l < num_layer_; l++) {
        decoder_layer_weights[l]->loadModel(dir_path + "/model.layers." + std::to_string(l), model_file_type);
    }
}

template<typename T>
bool GlmWeight<T>::isValidLayerParallelId(int l)
{
    int local_num_layer = (int)(ceil(num_layer_ * 1.0f / layer_para_size_));
    return l < num_layer_ && (l >= local_num_layer * layer_para_rank_)
           && (l < local_num_layer * (layer_para_rank_ + 1));
}

template struct GlmWeight<float>;
template struct GlmWeight<half>;

}  // namespace fastertransformer
