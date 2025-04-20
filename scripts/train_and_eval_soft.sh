gpu_id=${1}
USE_PIXEL_NORM=${2:-1}
USE_SOFT_VERSION=${3:-0}
feature_layer=${4:-11}

sh mae_train.sh $gpu_id $USE_PIXEL_NORM $USE_SOFT_VERSION $feature_layer
sh mae_eval_linear.sh $gpu_id $USE_PIXEL_NORM $USE_SOFT_VERSION $feature_layer
sh mae_eval_finetune.sh $gpu_id $USE_PIXEL_NORM $USE_SOFT_VERSION $feature_layer