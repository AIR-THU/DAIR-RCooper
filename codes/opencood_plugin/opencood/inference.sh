#!/bin/bash

logs_path="opencood/logs/*"
inference_path="opencood/tools/inference.py"

# cd "$logs_path" || exit

for file in $logs_path; do
    # if [ -f "$file" ]; then
    echo "infering "$file""
    CUDA_VISIBLE_DEVICES=0 python "$inference_path" --model_dir "$file"
    # nohup CUDA_VISIBLE_DEVICES=0 python "$inference_path" --model_dir "$file" >> ${file#*logs/}.log 2>&1 &
    # fi
done