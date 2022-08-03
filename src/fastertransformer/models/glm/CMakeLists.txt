# Copyright (c) 2019-2022, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.8)

add_library(GlmDecoderLayerWeight STATIC GlmDecoderLayerWeight.cc)
set_property(TARGET GlmDecoderLayerWeight PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET GlmDecoderLayerWeight PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(GlmDecoderLayerWeight PUBLIC memory_utils)

add_library(GlmDecoder STATIC GlmDecoder.cc)
set_property(TARGET GlmDecoder PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET GlmDecoder PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(GlmDecoder PUBLIC -lcudart cublasMMWrapper
                      TensorParallelGlmDecoderSelfAttentionLayer
                      TensorParallelGeluGluFfnLayer
                      layernorm_kernels
                      add_residual_kernels
                      GlmDecoderLayerWeight
                      nccl_utils)

add_library(GlmContextDecoder STATIC GlmContextDecoder.cc)
set_property(TARGET GlmContextDecoder PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET GlmContextDecoder PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(GlmContextDecoder PUBLIC -lcudart cublasMMWrapper
                      TensorParallelGlmContextAttentionLayer
                      TensorParallelGeluGluFfnLayer
                      layernorm_kernels
                      add_residual_kernels
                      gpt_kernels
                      nccl_utils)

add_library(GlmWeight STATIC GlmWeight.cc)
set_property(TARGET GlmWeight PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET GlmWeight PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(GlmWeight PUBLIC GlmDecoderLayerWeight)

add_library(Glm STATIC Glm.cc)
set_property(TARGET Glm PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET Glm PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(Glm PUBLIC -lcudart
                      GlmDecoder
                      GlmContextDecoder
                      decoding_kernels
                      gpt_kernels
                      DynamicDecodeLayer
                      BaseBeamSearchLayer
                      bert_preprocess_kernels
                      GlmWeight)
