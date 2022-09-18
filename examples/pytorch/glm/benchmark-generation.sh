#! /bin/bash

CUDA_LAUNCH_BLOCKING=1

MPSIZE=4
MAXSEQLEN=10000
MASTER_PORT=$(shuf -n 1 -i 10000-65535)

#SAMPLING ARGS
TEMP=1.0
#If TOPK/TOPP are 0 it defaults to greedy sampling, top-k will also override top-p
TOPK=1
TOPP=0

script_path=$(realpath $0)
script_dir=$(dirname $script_path)


OPTIONS_NCCL="NCCL_DEBUG=VERSION NCCL_IB_DISABLE=0 NCCL_NET_GDR_LEVEL=2 CUDA_LAUNCH_BLOCKING=0"

DISTRIBUTED_ARGS="--nproc_per_node $MPSIZE \
                  --nnodes 1 \
                  --node_rank 0 \
                  --master_addr localhost \
                  --master_port $MASTER_PORT"

CHECKPOINT_PATH="/thudm/workspace/prnake/ft_iter_0049300/8-gpu"

python -m torch.distributed.launch $DISTRIBUTED_ARGS $script_dir/glm_example.py \
       --time \
       --top_k $TOPK \
       --top_p $TOPP \
       --temperature $TEMP \
       --len_penalty 1 \
       --world_size $MPSIZE \
       --tensor_para_size $MPSIZE \
       --pipeline_para_size 1 \
       --data_type fp16 \
       --max_seq_len $MAXSEQLEN \
       --output_len 128 \
       --ckpt_path $CHECKPOINT_PATH \
       --max_batch_size 8 \
       --sample_input_file $script_dir/input.txt
