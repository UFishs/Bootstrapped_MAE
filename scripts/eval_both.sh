gpu_id=${1}
bootstrap_k=${2}
feature_layer=${3}
USE_PIXEL_NORM=${4:-1}
use_decoder_feature=${5:-1}

sh Bmae_eval_linear.sh  $gpu_id $bootstrap_k $feature_layer $USE_PIXEL_NORM $use_decoder_feature
sh Bmae_eval_finetune.sh  $gpu_id $bootstrap_k $feature_layer $USE_PIXEL_NORM $use_decoder_feature