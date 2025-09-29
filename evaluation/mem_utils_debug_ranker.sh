export HIP_VISIBLE_DEVICES="7"
export HF_HOME=/workspace/cache
export TOKENIZERS_PARALLELISM=true
export PYTHONPATH=/workspace/Tool-Star:$PYTHONPATH



python mem_utils_debug_ranker.py