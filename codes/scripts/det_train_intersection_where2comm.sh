DEVICE_ID=6
MODEL_CONFIG_DIR="configs/intersection_where2comm.yaml"

CUDA_VISIBLE_DEVICES=$DEVICE_ID python opencood_plugin/opencood/tools/train.py \
  --hypes_yaml $MODEL_CONFIG_DIR
