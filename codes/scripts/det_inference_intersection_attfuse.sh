DEVICE_ID=1
MODEL_DIR="/data/ad_sharing/datasets/RCOOPER_code/codes/ckpts/intersection_attfuse"
DATA_TRAIN_DIR="/data/ad_sharing/datasets/RCOOPER-COOD/data/intersection/train"
DATA_VAL_DIR="/data/ad_sharing/datasets/RCOOPER-COOD/data/intersection/val"

CUDA_VISIBLE_DEVICES=$DEVICE_ID python opencood_plugin/opencood/tools/inference.py \
  --model_dir $MODEL_DIR \
  --data_train_dir $DATA_TRAIN_DIR \
  --data_val_dir $DATA_VAL_DIR \
  --fusion_method intermediate \
  --save_json
