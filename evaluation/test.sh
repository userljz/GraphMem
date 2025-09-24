export HIP_VISIBLE_DEVICES="7"
export HF_HOME=/workspace/cache
export TOKENIZERS_PARALLELISM=true
export PYTHONPATH=/workspace/Tool-Star:$PYTHONPATH
# module load cuda/11.8

date=0924
trail=test
python run.py \
    --model_path dongguanting/Tool-Star-Qwen-1.5B \
    --dataset_name math \
    --task math \
    --gpu_use 0.4 \
    --max_tokens 31384 \
    --max_input_len 31384 \
    --output_path ${date}exp_math_result_mem_q_a_200.json \
    --counts 200 \
    --batch_size 10 \
    --find_nodes "q_a" \
    --use_memory \
    &> ./output_log/${date}_${trail}_q_a_200.txt
    

# python run.py \
#     --model_path dongguanting/Tool-Star-Qwen-1.5B \
#     --dataset_name math \
#     --task math \
#     --gpu_use 0.8 \
#     --max_tokens 31384 \
#     --max_input_len 31384 \
#     --output_path 0919exp_math_result_nomem_200.json \
#     --counts 200 \
#     --batch_size 10 \
#     &> ./output_log/0919_nomem_200.txt

# python run.py \
#     --model_path dongguanting/Tool-Star-Qwen-1.5B \
#     --dataset_name math \
#     --task math \
#     --gpu_use 0.8 \
#     --max_tokens 31384 \
#     --max_input_len 31384 \
#     --output_path 0919exp_math_result_mem_q_200.json \
#     --counts 200 \
#     --batch_size 10 \
#     --find_nodes "q" \
#     --use_memory \
#     &> ./output_log/0919_q_200.txt

