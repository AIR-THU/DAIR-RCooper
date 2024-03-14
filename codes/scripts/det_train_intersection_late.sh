DEVICE_ID=6
MODEL_CONFIG_DIR="configs/intersection_late.yaml"

CUDA_VISIBLE_DEVICES=$DEVICE_ID python v2v4real_plugin/opencood/tools/train.py \
  --hypes_yaml $MODEL_CONFIG_DIR
