export HIP_VISIBLE_DEVICES=7
export HF_HOME=/workspace/cache
export TOKENIZERS_PARALLELISM=true
export PYTHONPATH=/workspace/Tool-Star:$PYTHONPATH
# module load cuda/11.8

python run.py \
    --model_path dongguanting/Tool-Star-Qwen-1.5B \
    --dataset_name math \
    --task math \
    --gpu_use 0.8 \
    --max_tokens 31384 \
    --max_input_len 31384 \
    --output_path 0905exp_math_result_mem.json \
    --counts 50 \
    --batch_size 10 \
    --find_nodes "q_a" \
    --use_memory \
    &> ./output_log/0919_q_a.txt
    # --use_memory
    # --use_debug 

