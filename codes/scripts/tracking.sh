## Modification part

SCENE="intersection"
METHOD="cobevt"
NAME="${SCENE}_${METHOD}"

DATA_SOURCE_PATH="path_to_the_converted_DAIR_dataset/${SCENE}"

DET_OUTPUT="ckpts/${NAME}/json"  # folder of detection results containing json-format detection results
OUTPUT_PATH="ckpts/${NAME}/trk_res"  # folder of tracking results


## Run part

SPLIT_DATA_PATH="${DATA_SOURCE_PATH}/split-data.json"

OUTPUT_PATH_DTC="${OUTPUT_PATH}/trk_output/detection_results_to_kitti"
OUTPUT_PATH_TRACK="${OUTPUT_PATH}/trk_output/tracking_results_to_kitti"
TRACK_EVAL_OUTPUT_PATH="${OUTPUT_PATH}/trk_output/tracking_evaluation_results"

TRACK_EVAL_GT_PATH="../datasets/RCOOPER-KITTI-cooperative/${SCENE}"

### Generate KITTI format GT labels
echo """Generate GT label"""
python ab3dmot_plugin/data_convert/coop_label_dair2kitti.py \
   --source-root $DATA_SOURCE_PATH \
   --target-root "${TRACK_EVAL_GT_PATH}/cooperative" \
   --split-path $SPLIT_DATA_PATH

### Convert detection results to KITTI format
echo """Convert detection results to KITTI format"""
mkdir -p $OUTPUT_PATH_DTC
python ab3dmot_plugin/data_convert/label_det_result2kitti.py \
  --input-dir-path $DET_OUTPUT \
  --output-dir-path $OUTPUT_PATH_DTC \
  --ori-path $DATA_SOURCE_PATH \
  --split-data-path $SPLIT_DATA_PATH

### Run Tracking
echo """Run tracking"""
mkdir -p $OUTPUT_PATH_TRACK
python ab3dmot_plugin/AB3DMOT_plugin/main_tracking.py \
  --split-data-path $SPLIT_DATA_PATH \
  --input-path $OUTPUT_PATH_DTC  \
  --output-path $OUTPUT_PATH_TRACK 

### Evaluate Tracking
echo """Evaluate tracking results"""
mkdir -p $TRACK_EVAL_OUTPUT_PATH
python ab3dmot_plugin/eval_tracking.py \
  --track-eval-gt-path "${TRACK_EVAL_GT_PATH}/cooperative" \
  --track-results-path $OUTPUT_PATH_TRACK \
  --track-eval-output-path $TRACK_EVAL_OUTPUT_PATH
