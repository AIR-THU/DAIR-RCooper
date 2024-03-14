DEVICE_ID=5
MODEL_CONFIG_DIR="configs/corridor_late.yaml"

CUDA_VISIBLE_DEVICES=$DEVICE_ID python opencood_plugin/opencood/tools/train.py \
  --hypes_yaml $MODEL_CONFIG_DIR
