DEVICE_ID=5
MODEL_CONFIG_DIR="configs/corridor_early.yaml"

CUDA_VISIBLE_DEVICES=$DEVICE_ID python v2v4real_plugin/opencood/tools/train.py \
  --hypes_yaml $MODEL_CONFIG_DIR
