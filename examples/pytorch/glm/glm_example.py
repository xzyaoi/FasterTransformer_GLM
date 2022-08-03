# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
# Copyright (c) 2021, NAVER Corp.  Authored by CLOVA.
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

import sys
import ctypes

sys.setdlopenflags(sys.getdlopenflags() | ctypes.RTLD_GLOBAL)

from torch.nn.utils.rnn import pad_sequence
import random
import os
import re
import sys
import argparse
import timeit
import torch
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/../../..")
from examples.pytorch.glm.utils.glm import Glm

sys.setdlopenflags(sys.getdlopenflags() ^ ctypes.RTLD_GLOBAL)
from icetk_glm_130B import _IceTokenizer
tokenizer = _IceTokenizer()

torch.set_printoptions(precision=20)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--layer_num', type=int, default=70,
                        help='number of layers')
    parser.add_argument('--output_len', type=int, default=512,
                        help='output sequence length to generate.')
    parser.add_argument('--head_num', type=int, default=96,
                        help='head number')
    parser.add_argument('--size_per_head', type=int, default=128,
                        help='size per head')
    parser.add_argument('--vocab_size', type=int, default=150528,
                        help='vocab size')
    parser.add_argument('--rotary_embedding_dim', type=int, default=64,
                        help='vocab size')
    parser.add_argument('--beam_width', type=int, default=1,
                        help='beam width for beam search. Using sampling when beam width is 1.')
    parser.add_argument('--top_k', type=int, default=1,
                        help='top k candidate num')
    parser.add_argument('--top_p', type=float, default=0.,
                        help='top p probability threshold')
    parser.add_argument('--temperature', type=float, default=1.,
                        help='temperature')
    parser.add_argument('--len_penalty', type=float, default=1.,
                        help='len_penalty')
    parser.add_argument('--beam_search_diversity_rate', type=float, default=0.,
                        help='beam_search_diversity_rate')
    parser.add_argument('--tensor_para_size', type=int, default=8,
                        help='tensor parallel size')
    parser.add_argument('--pipeline_para_size', type=int, default=1,
                        help='pipeline parallel size')
    parser.add_argument('--ckpt_path', type=str, default='/thudm/workspace/prnake/FasterTransformer/examples/pytorch/glm/output/8-gpu',
                        help='path to the checkpoint file.')
    parser.add_argument('--lib_path', type=str, default='./lib',
                        help='path to the fastertransformer lib folder.')
    parser.add_argument('--start_id', type=int, default=50256,
                        help='start token id.')
    parser.add_argument('--end_id', type=int, default=150005,
                        help='end token id.')
    parser.add_argument('--max_batch_size', type=int, default=8,
                        help='max batch size.')
    parser.add_argument('--repetition_penalty', type=float, default=1.,
                        help='repetition penalty')
    parser.add_argument('--max_seq_len', type=int, default=1024,
                        help='max sequence length for position embedding table.')
    parser.add_argument('--data_type', type=str, choices=['fp32', 'fp16'], default='fp16')
    parser.add_argument('--time', action='store_true',
                        help='whether or not to measure time elapsed.')
    parser.add_argument('--sample_input_file', type=str, default=None,
                        help='path to sample input file. If not set, it runs with no context inputs.')
    parser.add_argument('--sample_output_file', type=str, default=None,
                        help='path to sample output file.')
    parser.add_argument('--is_fix_random_seed', type=bool, default=True,
                        help='is fixing the random seed.')
    parser.add_argument('--return_cum_log_probs', type=int, default=0, choices=[0, 1, 2],
                        help='Whether to compute the cumulative log probsbility of sentences.'
                             ' 0: do not return the cumulative log probs '
                             ' 1: return the cumulative log probs of generated sequences'
                             ' 2: return the cumulative log probs of sequences')
    parser.add_argument('--world_size', type=int, default=8)
    parser.add_argument('--local_rank', type=int, default=None,
                       help='local rank passed from distributed launcher.')

    args = parser.parse_args()

    layer_num = args.layer_num
    output_len = args.output_len
    head_num = args.head_num
    size_per_head = args.size_per_head
    vocab_size = args.vocab_size
    rotary_embedding_dim = args.rotary_embedding_dim
    beam_width = args.beam_width
    top_k = args.top_k
    top_p = args.top_p
    temperature = args.temperature
    len_penalty = args.len_penalty
    beam_search_diversity_rate = args.beam_search_diversity_rate
    tensor_para_size = args.tensor_para_size
    pipeline_para_size = args.pipeline_para_size
    start_id = args.start_id
    end_id = args.end_id
    max_batch_size = args.max_batch_size
    max_seq_len = args.max_seq_len
    repetition_penalty = args.repetition_penalty
    return_cum_log_probs = args.return_cum_log_probs
    return_output_length = return_cum_log_probs > 0

    if args.world_size > 1:
        torch.distributed.init_process_group(
            backend="nccl",
            world_size=args.world_size, rank=args.local_rank)
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
    else:
        rank = 0
        world_size = 1

    if rank == 0:
        print("\n=============== Arguments ===============")
        for arg in vars(args):
            print("{}: {}".format(arg, getattr(args, arg)))
        print("=========================================\n")

    def encode(raw_text):
        # add MASK
        generation_mask = '[gMASK]'
        assert '[MASK]' not in raw_text, 'should not mix [MASK] and [gMASK]'

        mask_pattern = r'\[g?MASK\]'
        text_list = re.split(mask_pattern, raw_text)
        pattern_list = re.compile(mask_pattern).findall(raw_text)
        seq = []
        for i in range(len(pattern_list)):
            pattern = pattern_list[i]
            sub_text = text_list[i]
            seq.extend(tokenizer.tokenize(sub_text))
            seq.append(tokenizer.get_command(pattern))

        seq.extend(tokenizer.tokenize(text_list[-1]))

        if 'MASK]' not in raw_text:
            seq += [tokenizer.get_command(generation_mask)]
            raw_text += ' ' + generation_mask
        if not raw_text.endswith('MASK]'):
            seq = seq + [tokenizer.get_command('eos')]
        seq = seq + [tokenizer.get_command('sop')]
        # seq = seq[:1024]
        if args.local_rank == 0:
            print('raw text: {}\n'.format(raw_text))
            print(seq)
        if len(seq) > args.max_seq_len:
            raise ValueError('text too long.')
        return seq


    # Inputs
    contexts = []
    if args.sample_input_file:  # conditional case
        with open(args.sample_input_file, "r") as f:
            contexts = f.read().splitlines()
            batch_size = min(len(contexts), max_batch_size)
        contexts = contexts[:batch_size]
        start_ids = [torch.IntTensor(encode(c)) for c in contexts]
    else:  # unconditional case
        batch_size = max_batch_size
        contexts = ['<|endoftext|>'] * batch_size
        start_ids = [torch.IntTensor([end_id])] * batch_size

    start_lengths = [len(ids) for ids in start_ids]
    input_len = max(start_lengths)

    start_ids = pad_sequence(start_ids, batch_first=True, padding_value=end_id)
    start_lengths = torch.IntTensor(start_lengths)

    if args.is_fix_random_seed == True:
        random_seed = 0
    else:
        random_seed = random.randint(0, 100000)

    # Prepare model.
    glm = Glm(head_num, size_per_head, vocab_size, rotary_embedding_dim, start_id, end_id,
                      layer_num, max_seq_len, tensor_para_size, pipeline_para_size,
                      lib_path=args.lib_path, world_size=args.world_size, rank=args.local_rank)
    if not glm.load(ckpt_path=args.ckpt_path):
        print("[WARNING] Checkpoint file not found. Model loading is skipped.")
    if args.data_type == 'fp16':
        glm.half()
        
    with torch.no_grad():
        # Generate tokens.
        tokens_batch = glm(start_ids,
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
                        return_output_length,
                        return_cum_log_probs)
        # # only a thread (rank 0) gets the output, while the others are supposed to return None.
        if tokens_batch is not None:

            if return_cum_log_probs > 0:
                tokens_batch, _, cum_log_probs = tokens_batch
                print('[INFO] Log probs of sentences:', cum_log_probs)
            outputs = []
            tokens_batch = tokens_batch.cpu().numpy()
            for i, (context, tokens) in enumerate(zip(contexts, tokens_batch)):
                for beam_id in range(beam_width):
                    token = tokens[beam_id][start_lengths[i]:]  # exclude context input from the output
                    # output = enc.decode(token)
                    output = token
                    outputs.append(output)
                    if args.local_rank == 0:
                        print(f"[INFO] batch {i}, beam {beam_id}: \n[Context]\n{context}\n\n[Output]\n")
                        for k in output:
                            if k:
                                print(tokenizer.IdToToken(int(k)),end=" ")
                        print()
 
        if args.time:
            iterations = 3
            time = timeit.default_timer()
            for i in range(iterations):
                tokens_batch = glm(start_ids,
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
                                return_output_length,
                                return_cum_log_probs)
            time_elapsed = timeit.default_timer() - time
            if args.local_rank == 0:
                print("[INFO] GPT length {} time costs: {:.2f} ms".format(output_len, time_elapsed * 1000 / iterations))

if __name__ == '__main__':
    main()
