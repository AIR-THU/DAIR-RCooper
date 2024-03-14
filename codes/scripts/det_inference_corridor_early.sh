DEVICE_ID=0
MODEL_DIR="/data/ad_sharing/datasets/RCOOPER_code/codes/ckpts/corridor_early"
DATA_TRAIN_DIR="/data/ad_sharing/datasets/RCOOPER-V2V4REAL/data/corridor/train"
DATA_VAL_DIR="/data/ad_sharing/datasets/RCOOPER-V2V4REAL/data/corridor/val"

CUDA_VISIBLE_DEVICES=$DEVICE_ID python v2v4real_plugin/opencood/tools/inference.py \
  --model_dir $MODEL_DIR \
  --data_train_dir $DATA_TRAIN_DIR \
  --data_val_dir $DATA_VAL_DIR \
  --fusion_method early \
  --save_json
