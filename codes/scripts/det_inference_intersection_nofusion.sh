DEVICE_ID=6
MODEL_DIR="/data/ad_sharing/datasets/RCOOPER_code/codes/ckpts/intersection_late"
DATA_TRAIN_DIR="/data/ad_sharing/datasets/RCOOPER-V2V4REAL/data/intersection/train"
DATA_VAL_DIR="/data/ad_sharing/datasets/RCOOPER-V2V4REAL/data/intersection/val"

CUDA_VISIBLE_DEVICES=$DEVICE_ID python v2v4real_plugin/opencood/tools/inference.py \
  --model_dir $MODEL_DIR \
  --data_train_dir $DATA_TRAIN_DIR \
  --data_val_dir $DATA_VAL_DIR \
  --fusion_method nofusion \
  --save_json
