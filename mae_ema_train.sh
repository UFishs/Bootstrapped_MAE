gpu_id=${1:-1}
ema_decay=${2:-0.999}
ema_warmup=${3:-40}
use_decoder_feature=${4:-1}

export CUDA_VISIBLE_DEVICES=$gpu_id
echo "Using GPU: $gpu_id"
echo "EMA decay: $ema_decay"
echo "EMA warmup: $ema_warmup"
echo "Use decoder feature: $use_decoder_feature"

MODEL="deit_tiny_patch4"
DATA_PATH="./cifar-10-dataset"

BASE_PATH="MAE-EMA"
BASE_DIR="decay-$ema_decay-warmup-$ema_warmup"

if [ $use_decoder_feature = 1 ]; then
    echo "Using decoder feature"
    USE_DECODER_FEATURE=" --use_decoder_feature"
else
    echo "Not using decoder feature"
    BASE_DIR="$BASE_DIR-wo-decoder-feature"
    USE_DECODER_FEATURE=" --not_use_decoder_feature"
fi

OUTPUT_DIR="./$BASE_PATH/$BASE_DIR/output_dir"
LOG_DIR="./$BASE_PATH/$BASE_DIR/log_dir"

BATCH_SIZE=256
EPOCHS=200
LR=1e-4
WEIGHT_DECAY=0.05

EMA_DECAY=$ema_decay
EMA_WARMUP=$ema_warmup

cmd="python main_pretrain.py \
    --model $MODEL \
    --data_path $DATA_PATH \
    --output_dir $OUTPUT_DIR \
    --log_dir $LOG_DIR \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --lr $LR \
    --weight_decay $WEIGHT_DECAY \
    --device cuda \
    --input_size 32  \
    --use_ema \
    --ema_decay $EMA_DECAY \
    --ema_warmup $EMA_WARMUP"

cmd=$cmd$USE_DECODER_FEATURE
echo $cmd
eval $cmd
