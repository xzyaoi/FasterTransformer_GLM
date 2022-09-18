# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
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

from __future__ import print_function

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import torch.distributed as dist

def get_torch_default_comm():
    r"""
    The NCCL communicator is needed so that Fast MoE can perform customized
    communication operators in the C code. However, it is not a publicly
    available variable. Therefore, a hacking class of the `ProcessGroupNCCL`
    in Fast MoE's C code takes the `_default_pg` and tries to dig the
    communicator out from the object. As PyTorch's private interface varies from
    time to time, different hacking techniques are tried one-by-one to be
    compatible with various versions of PyTorch.
    """
    try:
        comm = dist.distributed_c10d._get_default_group()
        return comm
    except Exception as _:
        pass
    try:
        comm = dist.distributed_c10d._default_pg
        if comm is not None:
            return comm
    except Exception as _:
        pass
    raise RuntimeError("Unsupported PyTorch version")

class GlmWeights(object):
    def __init__(self, head_num, size_per_head, layer_num, vocab_size, max_seq_len, tensor_para_size, pipeline_para_size):
        assert(head_num % tensor_para_size == 0)
        self.head_num = head_num
        self.size_per_head = size_per_head
        self.layer_num = layer_num
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.tensor_para_size = tensor_para_size
        self.pipeline_para_size = pipeline_para_size
        self.layers_per_device = layer_num // pipeline_para_size

        local_head_num = head_num // tensor_para_size
        global_head_num = head_num
        local_hidden_units = local_head_num * size_per_head
        global_hidden_units = global_head_num * size_per_head
        local_inter_size = local_hidden_units * 8 // 3

        self.local_head_num = local_head_num
        self.global_head_num = global_head_num
        self.local_hidden_units = local_hidden_units
        self.global_hidden_units = global_hidden_units
        self.local_inter_size = local_inter_size

        self.w = []
        self.int8_w = []
        self.int8_scale = []
        # Transformer blocks

        self.w.extend([torch.zeros(3 * local_hidden_units)] * layer_num)                                   # attention.query_key_value.bias
        self.w.extend([torch.zeros(global_hidden_units)] * layer_num)                                   # attention.dense.bias
        self.w.extend([torch.zeros(global_hidden_units)] * layer_num)                                   # input_layernorm.bias
        self.w.extend([torch.zeros(global_hidden_units)] * layer_num)                                   # input_layernorm.weight
        self.w.extend([torch.zeros(local_inter_size)] * layer_num)                                   # mlp.dense_h_to_4h.bias.1
        self.w.extend([torch.zeros(local_inter_size)] * layer_num)                                   # mlp.dense_h_to_4h.bias.2
        self.w.extend([torch.zeros(global_hidden_units)] * layer_num)                                   # mlp.dense_4h_to_h.bias
        self.w.extend([torch.zeros(global_hidden_units)] * layer_num)                                   # post_attention_layernorm.bias
        self.w.extend([torch.zeros(global_hidden_units)] * layer_num)                                   # post_attention_layernorm.weight
        
        # self.w.extend([torch.zeros(global_hidden_units * 3 * local_hidden_units)] * layer_num)             # attention.query_key_value.weight
        # self.w.extend([torch.zeros(local_hidden_units * global_hidden_units)] * layer_num)                                   # attention.dense.weight
        # self.w.extend([torch.zeros(global_hidden_units * local_inter_size)] * layer_num)                                   # mlp.dense_h_to_4h.weight.1
        # self.w.extend([torch.zeros(global_hidden_units * local_inter_size)] * layer_num)                                   # mlp.dense_h_to_4h.weight.2
        # self.w.extend([torch.zeros(local_inter_size * global_hidden_units)] * layer_num)                                   # mlp.dense_4h_to_h.weight
        
        self.int8_w.extend([torch.zeros(global_hidden_units * 3 * local_hidden_units // 2, dtype = torch.int8)] * layer_num)             # attention.query_key_value.weight
        self.int8_w.extend([torch.zeros(local_hidden_units * global_hidden_units // 2, dtype = torch.int8)] * layer_num)                                   # attention.dense.weight
        self.int8_w.extend([torch.zeros(global_hidden_units * local_inter_size // 2, dtype = torch.int8)] * layer_num)                                   # mlp.dense_h_to_4h.weight.1
        self.int8_w.extend([torch.zeros(global_hidden_units * local_inter_size // 2, dtype = torch.int8)] * layer_num)                                   # mlp.dense_h_to_4h.weight.2
        self.int8_w.extend([torch.zeros(local_inter_size * global_hidden_units // 2, dtype = torch.int8)] * layer_num)                                   # mlp.dense_4h_to_h.weight
        
        # scale
        self.int8_scale.extend([torch.zeros(3 * local_hidden_units, dtype = torch.float32)] * layer_num)
        self.int8_scale.extend([torch.zeros(global_hidden_units, dtype = torch.float32)] * layer_num)
        self.int8_scale.extend([torch.zeros(local_inter_size, dtype = torch.float32)] * layer_num)
        self.int8_scale.extend([torch.zeros(local_inter_size, dtype = torch.float32)] * layer_num)
        self.int8_scale.extend([torch.zeros(global_hidden_units, dtype = torch.float32)] * layer_num)
        
        # After Transformer blocks
        self.w.append(torch.zeros(global_hidden_units))   # layernorm_gamma final_layernorm.weight
        self.w.append(torch.zeros(global_hidden_units))   # layernorm_beta  final_layernorm.bias
        self.w.append(torch.zeros(vocab_size * global_hidden_units))   # embedding_table model.wte
        # self.w.append(torch.zeros(vocab_size, global_hidden_units))   # embedding_kernel model.wte

        # Initialization
        # self._map(lambda w: torch.nn.init.normal_(w, mean=0., std=1.))


    def __getitem__(self, idx):
        return self.w[idx]

    def __setitem__(self, idx, val):
        self.w[idx] = val

    def __len__(self):
        return len(self.w)

    def _map(self, func, w = None):
        if not w:
            w = self.w
        for i in range(len(w)):
            if isinstance(w[i], list):
                for j in range(len(w[i])):
                    w[i][j] = func(w[i][j])
            else:
                w[i] = func(w[i])

    def load(self, ckpt_path, tensor_para_rank, pipeline_para_rank):
        if not os.path.exists(ckpt_path):
            return False
        w = []

        # Load
        w.extend([torch.from_numpy(np.fromfile(ckpt_path + "/model.layers.{}.attention.query_key_value.weight.{}.bin".format(i, tensor_para_rank), dtype=np.half)) for i in range(self.layer_num)])

        w.extend([torch.from_numpy(np.fromfile(ckpt_path + "/model.layers.{}.attention.query_key_value.bias.{}.bin".format(i, tensor_para_rank), dtype=np.half)) for i in range(self.layer_num)])
        w.extend([torch.from_numpy(np.fromfile(ckpt_path + "/model.layers.{}.attention.dense.bias.{}.bin".format(i, tensor_para_rank), dtype=np.half)) for i in range(self.layer_num)])
        w.extend([torch.from_numpy(np.fromfile(ckpt_path + "/model.layers.{}.input_layernorm.bias.{}.bin".format(i, tensor_para_rank), dtype=np.half)) for i in range(self.layer_num)])
        w.extend([torch.from_numpy(np.fromfile(ckpt_path + "/model.layers.{}.input_layernorm.weight.{}.bin".format(i, tensor_para_rank), dtype=np.half)) for i in range(self.layer_num)])
        w.extend([torch.from_numpy(np.fromfile(ckpt_path + "/model.layers.{}.mlp.dense_h_to_4h.bias.{}.1.bin".format(i, tensor_para_rank), dtype=np.half)) for i in range(self.layer_num)])
        w.extend([torch.from_numpy(np.fromfile(ckpt_path + "/model.layers.{}.mlp.dense_h_to_4h.bias.{}.2.bin".format(i, tensor_para_rank), dtype=np.half)) for i in range(self.layer_num)])
        w.extend([torch.from_numpy(np.fromfile(ckpt_path + "/model.layers.{}.mlp.dense_4h_to_h.bias.{}.bin".format(i, tensor_para_rank), dtype=np.half)) for i in range(self.layer_num)])
        w.extend([torch.from_numpy(np.fromfile(ckpt_path + "/model.layers.{}.post_attention_layernorm.bias.{}.bin".format(i, tensor_para_rank), dtype=np.half)) for i in range(self.layer_num)])
        w.extend([torch.from_numpy(np.fromfile(ckpt_path + "/model.layers.{}.post_attention_layernorm.weight.{}.bin".format(i, tensor_para_rank), dtype=np.half)) for i in range(self.layer_num)])

        w.extend([torch.from_numpy(np.fromfile(ckpt_path + "/model.layers.{}.attention.query_key_value.weight.{}.bin".format(i, tensor_para_rank), dtype=np.half)) for i in range(self.layer_num)])
        w.extend([torch.from_numpy(np.fromfile(ckpt_path + "/model.layers.{}.attention.dense.weight.{}.bin".format(i, tensor_para_rank), dtype=np.half)) for i in range(self.layer_num)])
        w.extend([torch.from_numpy(np.fromfile(ckpt_path + "/model.layers.{}.mlp.dense_h_to_4h.weight.{}.1.bin".format(i, tensor_para_rank), dtype=np.half)) for i in range(self.layer_num)])
        w.extend([torch.from_numpy(np.fromfile(ckpt_path + "/model.layers.{}.mlp.dense_h_to_4h.weight.{}.2.bin".format(i, tensor_para_rank), dtype=np.half)) for i in range(self.layer_num)])
        w.extend([torch.from_numpy(np.fromfile(ckpt_path + "/model.layers.{}.mlp.dense_4h_to_h.weight.{}.bin".format(i, tensor_para_rank), dtype=np.half)) for i in range(self.layer_num)])

        # scale
        w.extend([torch.zeros(3 * local_hidden_units)] * layer_num)
        w.extend([torch.zeros(global_hidden_units)] * layer_num)
        w.extend([torch.zeros(local_inter_size)] * layer_num)
        w.extend([torch.zeros(local_inter_size)] * layer_num)
        w.extend([torch.zeros(global_hidden_units)] * layer_num)


        w.append(torch.from_numpy(np.fromfile(ckpt_path + "/model.final_layernorm.weight.{}.bin".format(tensor_para_rank), dtype=np.half)))
        w.append(torch.from_numpy(np.fromfile(ckpt_path + "/model.final_layernorm.bias.{}.bin".format(tensor_para_rank), dtype=np.half)))
        w.append(torch.from_numpy(np.fromfile(ckpt_path + "/model.wte.bin", dtype=np.half)))
        # w.append(torch.from_numpy(np.fromfile(ckpt_path + "/model.wte.bin", dtype=np.half)))

        if self.rank == 0:
            import pdb
            pdb.set_trace()
        torch.distributed.barrier()

        # Reshape
        try:
            for i in range(len(w)):
                if w[i].nelement() > 0:
                    self.w[i] = w[i].reshape(self.w[i].shape)

        except RuntimeError:
            raise RuntimeError(
                "head_num, size_per_head, vocab_size, and max_seq_len must be the same as the ones during training.")

        #transpose calibrate quantize the kernel
        layer_num = self.layer_num

        return True


class Glm(nn.Module):
    def __init__(self,
                 head_num, size_per_head,
                 vocab_size, rotary_embedding_dim, start_id, end_id, layer_num,
                 max_seq_len,
                 tensor_para_size, pipeline_para_size,
                 lib_path,
                 world_size,
                 rank,
                 dtype="fp16"):
        super().__init__()
        self.head_num = head_num
        self.size_per_head = size_per_head
        self.vocab_size = vocab_size
        self.rotary_embedding_dim = rotary_embedding_dim
        self.start_id = start_id
        self.end_id = end_id
        self.layer_num = layer_num
        self.tensor_para_size = tensor_para_size
        self.pipeline_para_size = pipeline_para_size
        self.max_seq_len = max_seq_len
        self.use_sparse_gemm = False
        self.build_model = False

        assert torch.cuda.is_available(), "CUDA is required for this model."
 
        assert head_num % tensor_para_size == 0, "head_num must be a multiple of tensor_para_size."
        assert layer_num % pipeline_para_size == 0, "layer_num must be a multiple of pipeline_para_size."

        # Load the C++ model.
        sys.path.append(os.path.abspath(lib_path))
        import libth_glm
        self.Glm = libth_glm.Glm

        # Prepare weights
        self.weights = GlmWeights(head_num, size_per_head, layer_num, vocab_size,
                                  max_seq_len, tensor_para_size, pipeline_para_size)

        # Prepare for tensor/pipeline parallel
        
        self.rank = rank
        self.device_count = torch.cuda.device_count()
        self.device = self.rank % self.device_count
        torch.cuda.set_device(self.device)

        assert world_size == tensor_para_size * pipeline_para_size, "tensor_para_size * pipeline_para_size must be equal to world_size."

        self.tensor_para_rank = self.rank % self.tensor_para_size
        self.pipeline_para_rank = self.rank // self.tensor_para_size

        # Create and copy model to the device.
        # if dtype == "fp16":
        #     self.half()
        # else:
        #     self.cuda()

    def load(self, ckpt_path):
        is_load = self.weights.load(ckpt_path, tensor_para_rank=self.tensor_para_rank,
                                    pipeline_para_rank=self.pipeline_para_rank)
        # self.cuda()
        return is_load

    def half(self):
        self.weights._map(lambda w: w.half())   
        # self.cuda()

    def bfloat16(self):
        self.weights._map(lambda w: w.bfloat16())
        # self.cuda()

    def sparse(self):
        if not self.use_sparse_gemm:
            self.use_sparse_gemm = True
            # self.cuda()

    def cuda(self):
        # print(self.device)
        # if int(self.device) == 3:
        self.weights._map(lambda w: w.cuda(self.device))
        self.weights._map(lambda w: w.cuda(self.device), self.weights.int8_w)
        self.weights._map(lambda w: w.cuda(self.device), self.weights.int8_scale)        
        # import time
        # print(self.device)
        # time.sleep(200000)
        # exit()

        if self.build_model:
            pass
            # del self.model
            # self.build_model = False
        else:
            
            self.model = self.Glm(get_torch_default_comm(), self.rank, self.head_num, self.size_per_head, self.head_num * self.size_per_head * 8 // 3,
                                                            self.layer_num, self.vocab_size, self.rotary_embedding_dim, self.start_id, self.end_id,
                                                            self.tensor_para_size, self.pipeline_para_size, self.weights.w, self.weights.int8_w, self.weights.int8_scale)
        self.build_model = True

    def forward(self,
                start_ids,
                start_lengths,
                output_len,
                beam_width=1,
                top_k=1,
                top_p=0.0,
                beam_search_diversity_rate=0.0,
                temperature=1.0,
                len_penalty=1.0,
                repetition_penalty=1.0,
                random_seed=0,
                return_output_length=False,
                return_cum_log_probs=0):
        if not self.build_model:
            self.cuda()
        input_len = start_ids.size(1)
        assert input_len > 0, "input len must be larger than zero. For an unconditional case, use start_id as the first token."
        
        # Inputs to device
        start_ids = start_ids.cuda(self.device)
        start_lengths = start_lengths.cuda(self.device)
        # outputs: output_ids, output_lengths, output_cum_log_probs (optional)

        # output_ids_buf = torch.zeros([start_ids.shape[0],beam_width,input_len + output_len],dtype=torch.int32).cuda()
        # logits_buf = torch.zeros([start_ids.shape[0],beam_width,self.vocab_size],dtype=torch.float32).cuda()
        
        # self.model.encode(start_ids,
        #                     start_lengths,
        #                     output_ids_buf,
        #                     logits_buf,
        #                     output_len,
        #                     beam_width,
        #                     top_k,
        #                     top_p,
        #                     beam_search_diversity_rate,
        #                     temperature,
        #                     len_penalty,
        #                     repetition_penalty,
        #                     random_seed,
        #                     return_cum_log_probs)


        outputs = self.model.forward(start_ids,
                                     start_lengths,
                                     output_len,
                                     beam_width,
                                     top_k,
                                     top_p,
                                     beam_search_diversity_rate,
                                     temperature,
                                     len_penalty,
                                     repetition_penalty,
                                     random_seed,
                                     return_cum_log_probs)
        
        # self.model.decode(input_len)
        # output_ids_buf[0][0][input_len] += logits_buf.argmax()
        # self.model.decode(input_len+1)

        if return_cum_log_probs == 0:
            output_ids, output_lengths = outputs
        else:
            output_ids, output_lengths, output_cum_log_probs = outputs

        if return_output_length:
            if return_cum_log_probs > 0:
                return output_ids, output_lengths, output_cum_log_probs
            else:
                return output_ids, output_lengths
        else:
            return output_ids

    def set_input_tensor(self, input_tensor):
        """Set input tensor to be used instead of forward()'s input.

        When doing pipeline parallelism the input from the previous
        stage comes from communication, not from the input, so the
        model's forward_step_func won't have it. This function is thus
        used by internal code to bypass the input provided by the
        forward_step_func"""
        self.input_tensor = input_tensor
