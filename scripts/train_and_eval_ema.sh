gpu_id=${1}
ema_decay=${2:-0.999}
ema_warmup=${3:-40}
use_decoder_feature=${4:-1}

sh mae_ema_train.sh  $gpu_id $ema_decay $ema_warmup $use_decoder_feature
sh mae_ema_eval_linear.sh  $gpu_id $ema_decay $ema_warmup $use_decoder_feature
sh mae_ema_eval_finetune.sh  $gpu_id $ema_decay $ema_warmup $use_decoder_feature