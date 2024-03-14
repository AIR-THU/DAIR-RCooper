DEVICE_ID=0
MODEL_CONFIG_DIR="configs/corridor_cobevt.yaml"

CUDA_VISIBLE_DEVICES=$DEVICE_ID python opencood_plugin/opencood/tools/train.py \
  --hypes_yaml $MODEL_CONFIG_DIR
