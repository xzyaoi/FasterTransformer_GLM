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

torch.manual_seed(42)

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/../../..")
from examples.pytorch.glm.utils.glm import Glm

sys.setdlopenflags(sys.getdlopenflags() ^ ctypes.RTLD_GLOBAL)
from icetk_glm_130B import _IceTokenizer
tokenizer = _IceTokenizer()

torch.set_printoptions(precision=20)

parser = argparse.ArgumentParser()
parser.add_argument('--layer_num', type=int, default=70,
                    help='number of layers')
parser.add_argument('--head_num', type=int, default=96,
                    help='head number')
parser.add_argument('--size_per_head', type=int, default=128,
                    help='size per head')
parser.add_argument('--vocab_size', type=int, default=150528,
                    help='vocab size')
parser.add_argument('--rotary_embedding_dim', type=int, default=64,
                    help='vocab size')
parser.add_argument('--tensor_para_size', type=int, default=8,
                    help='tensor parallel size')
parser.add_argument('--pipeline_para_size', type=int, default=1,
                    help='pipeline parallel size')
parser.add_argument('--ckpt_path', type=str,
                    help='path to the checkpoint file.')
parser.add_argument('--lib_path', type=str, default='./lib',
                    help='path to the fastertransformer lib folder.')
parser.add_argument('--start_id', type=int, default=150004,
                    help='start token id.')
parser.add_argument('--end_id', type=int, default=150001,
                    help='end token id.')
parser.add_argument('--max_seq_len', type=int, default=1024,
                    help='max sequence length for position embedding table.')
parser.add_argument('--data_type', type=str, choices=['fp16', 'int8', 'int4'], default='fp16')
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
head_num = args.head_num
size_per_head = args.size_per_head
vocab_size = args.vocab_size
rotary_embedding_dim = args.rotary_embedding_dim
tensor_para_size = args.tensor_para_size
pipeline_para_size = args.pipeline_para_size
start_id = args.start_id
end_id = args.end_id
max_seq_len = args.max_seq_len

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

# Prepare model.
glm = Glm(head_num, size_per_head, vocab_size, rotary_embedding_dim, start_id, end_id,
                  layer_num, max_seq_len, tensor_para_size, pipeline_para_size,
                  lib_path=args.lib_path, world_size=args.world_size, rank=args.local_rank, tokenizer=tokenizer, dtype = args.data_type)
if not glm.load(ckpt_path=args.ckpt_path):
    print("[WARNING] Checkpoint file not found. Model loading is skipped.")

glm.init_model(512,# output_len,
                1, # beam_width
                1, # top_k,
                0, # top_p,
                0., # beam_search_diversity_rate,
                1.0, # temperature,
                1., # len_penalty,
                1., #repetition_penalty,
                42, # random_seed
                )
    
    # with torch.no_grad():
    #     for _ in range(3):
    #         # Generate tokens.
    #         tokens_batch = glm(start_ids,
    #                             start_lengths,
    #                             mask_positions,
    #                             return_output_length,
    #                             return_cum_log_probs)
    #         # # only a thread (rank 0) gets the output, while the others are supposed to return None.
    #         if args.local_rank == 0:
    #             print(get_res(tokens_batch))  
# if __name__ == '__main__':
#     main()

import json
from flask import Flask, request, jsonify, make_response
import torch.distributed as dist
from threading import Semaphore
from utils.tools import Timeout

# from utils.generation import BeamSearchStrategy

gpu_sem = Semaphore(1)
app = Flask('glm-130b')

# @app.route('/predict')
# def run_server():
#     args = request.args
#     return predict(**args)

def tokenize(contexts):
    def encode(raw_text):
        # add MASK
        generation_mask = "[MASK]" if "[MASK]" in raw_text else "[gMASK]"
        use_gmask = "[MASK]" not in raw_text

        mask_pattern = r"\[g?MASK\]"
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
        # if args.local_rank == 0:
        #     print('raw text: {}\n'.format(raw_text))
        #     print(seq)
        # if len(seq) > args.max_seq_len:
        #     raise ValueError('text too long.')
        return torch.IntTensor(seq), -1 if use_gmask else seq.index(tokenizer.get_command(generation_mask))

    def get_ids(contexts):
        start_ids, mask_positions = zip(*[encode(c) for c in contexts])
        start_lengths = torch.IntTensor([len(ids) for ids in start_ids])
        start_ids = pad_sequence(start_ids, batch_first=True, padding_value=0)
        print(start_lengths, torch.IntTensor([len(ids) for ids in start_ids]))
        return start_ids, start_lengths, torch.IntTensor(mask_positions)
    
    return get_ids(contexts)

@app.route('/tokenize', methods=["POST"])
def get_tokenize():
    config = request.get_data().decode()
    if config is None or config == "":
        return make_response("Config can not be empty.", 500)
    config = json.loads(config)

    contexts = config["text"].splitlines()

    start_ids, start_lengths, mask_positions = tokenize(contexts)

    return make_response(jsonify({
        "start_ids": start_ids.tolist(),
        "start_lengths": start_lengths.tolist(),
        "mask_positions": mask_positions.tolist()
    }),200)

@app.route('/generate', methods=["POST"])
def get_generate():
    config = request.get_data().decode()
    if config is None or config == "":
        return make_response("Config can not be empty.", 500)
    config = json.loads(config)

    contexts = config["text"].splitlines()

    start_ids, start_lengths, mask_positions = tokenize(contexts)

    args = {}
    for i in ["seed", "out_seq_length", "min_gen_length", "sampling_strategy", "num_beams", "length_penalty", "no_repeat_ngram_size", "temperature", "topk", "topp"]:
        if config.get(i):
            args[i] = config.get(i)

    with gpu_sem:
        res = predict(start_ids, start_lengths, mask_positions, **args)

    return make_response(jsonify({"text": res}),200)

if __name__ == "__main__":

#     end_tokens = [tokenizer.get_command("eop"), tokenizer.get_command("eos")]
        
    def get_res(tokens_batch, start_lengths):
        res = []
        try:
            if tokens_batch is not None:
                tokens_batch = tokens_batch.cpu().numpy()
                for i, tokens in enumerate(tokens_batch):
                    # for beam_id in range(beam_width):
                    beam_id = 0
                    token = list(tokens[beam_id][start_lengths[0]:]) # exclude context input from the output
                    print(token, list(tokens[beam_id]))
                    if 20002 in token:
                        token = token[:token.index(20002)]
                    if 150005 in token:
                        token = token[:token.index(150005)]
                    res.append(tokenizer.detokenize(token))
        except:
            pass
        return res

    # def predict(start_ids, start_lengths, mask_positions, return_output_length):

    #     tokens_batch = glm(start_ids,
    #                         start_lengths,
    #                         mask_positions,
    #                         return_output_length,
    #                         0, # return_cum_log_probs
    #                         )

        # global strategy

#         if args.with_id:
#             query_id, raw_text = raw_text.split("\t")

#         answers, answers_with_style, blanks = fill_blanks(raw_text, model, tokenizer, strategy)

        # res = get_res(tokens_batch)
        # if torch.distributed.get_rank() == 0:
        #     print(res)

        # return res

    def predict(start_ids, start_lengths, mask_positions, seed=1234, out_seq_length=200, min_gen_length=20, sampling_strategy='BaseStrategy', 
    num_beams=4, length_penalty=0.9, no_repeat_ngram_size=3, 
    temperature=1, topk=5, topp=0):

        if start_ids.size(1) + out_seq_length > max_seq_len:
            return ["length too long"]
            

#         global strategy

        if torch.distributed.get_rank() == 0:
            print('info', [start_ids, start_lengths, mask_positions, seed, out_seq_length, min_gen_length, sampling_strategy, num_beams, length_penalty, no_repeat_ngram_size, temperature, topk, topp])
            dist.broadcast_object_list([start_ids, start_lengths, mask_positions, seed, out_seq_length, min_gen_length, sampling_strategy, num_beams, length_penalty, no_repeat_ngram_size, temperature, topk, topp], src=0)

        
        torch.manual_seed(seed)


        tokens_batch = glm(start_ids,
                        start_lengths,
                        mask_positions,
                        out_seq_length,
                        1,
                        topk,
                        topp,
                        temperature=temperature)
        res = get_res(tokens_batch, start_lengths)
        if torch.distributed.get_rank() == 0:
            print(res)
        return res

        # return res
#         args.seed = seed
#         args.out_seq_length = out_seq_length
#         args.min_gen_length = min_gen_length
#         args.sampling_strategy = sampling_strategy
#         args.num_beams = num_beams
#         args.length_penalty = length_penalty
#         args.no_repeat_ngram_size = no_repeat_ngram_size
#         args.temperature = temperature
#         args.top_k = topk
#         args.top_p = topp

#         if args.sampling_strategy == "BaseStrategy":
#             strategy = BaseStrategy(temperature=args.temperature, top_k=args.top_k, top_p=args.top_p, end_tokens=end_tokens)
#         elif args.sampling_strategy == "BeamSearchStrategy":
#             strategy = BeamSearchStrategy(
#                 args.num_beams,
#                 length_penalty=args.length_penalty,
#                 consider_end=True,
#                 end_tokens=end_tokens,
#                 no_repeat_ngram_size=args.no_repeat_ngram_size,
#                 min_gen_length=args.min_gen_length,
#             )
#         else:
#             raise ValueError(f"unknown strategy {args.sampling_strategy}")

#         return generate_continually(process, text)

#     # from https://github.com/hanyullai/GLM-130B/commit/a0ad56b76650eee679123fcc26bb92d2b3b49cb2

    if torch.distributed.get_rank() == 0:
        app.run(host="0.0.0.0")
    else:
        while True:
            info = [None, None, None, None, None, None, None, None, None, None, None, None, None]
            dist.broadcast_object_list(info, src=0)

            start_ids, start_lengths, mask_positions, seed, out_seq_length, min_gen_length, sampling_strategy, num_beams, length_penalty, no_repeat_ngram_size, temperature, topk, topp = info

            predict(start_ids, start_lengths, mask_positions, seed, out_seq_length, min_gen_length, sampling_strategy, 
                num_beams, length_penalty, no_repeat_ngram_size, 
                temperature, topk, topp)
