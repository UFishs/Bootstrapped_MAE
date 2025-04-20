
# Initialize variables with default values
gpu_id=${1}
batch_size=256
use_ema=false
bootstrap_k=${2:-4}
feature_layer=${3:-11}
epochs_sum=200
base_path="./bootstrap"
USE_PIXEL_NORM=${4:-1}
use_decoder_feature=${5:-1}

echo "Using GPU ID: $gpu_id"
echo "Pretrain using batch size: $batch_size"
echo "base_path: $base_path"
echo "bootstrap_k: $bootstrap_k"
echo "feature_layer: $feature_layer"
echo "use pixel norm: $USE_PIXEL_NORM"
echo "use decoder feature: $use_decoder_feature"

export CUDA_VISIBLE_DEVICES=$gpu_id

MODEL="deit_tiny_patch4"
DATA_PATH="./cifar-10-dataset"
img_size=32


LR=1e-4
WEIGHT_DECAY=0.05
EPOCHS=$epochs_sum


if [ $USE_PIXEL_NORM = 1 ]; then
    echo "Using pixel norm"
    NORM_PIX_LOSS="--norm_pix_loss"
    BASE_DIR="MAE-$bootstrap_k-feature-$feature_layer"
else
    echo "Not using pixel norm"
    NORM_PIX_LOSS=""
    BASE_DIR="MAE-$bootstrap_k-feature-$feature_layer-wo-pixelnorm"
fi

if [ $use_decoder_feature = 1 ]; then
    echo "Using decoder feature"
    USE_DECODER_FEATURE=" --use_decoder_feature"
else
    echo "Not using decoder feature"
    BASE_DIR=$BASE_DIR"-wo-decoder-feature"
    USE_DECODER_FEATURE=" --not_use_decoder_feature"
fi

LOG_DIR="$base_path/$BASE_DIR/log_dir"
OUTPUT_DIR="$base_path/$BASE_DIR/output_dir"

cmd="python main_pretrain.py \
    --model $MODEL \
    --data_path $DATA_PATH \
    --output_dir $OUTPUT_DIR \
    --log_dir $LOG_DIR \
    --batch_size $batch_size \
    --epochs $EPOCHS \
    --lr $LR \
    --weight_decay $WEIGHT_DECAY \
    --device cuda \
    --input_size $img_size \
    --bootstrap_k $bootstrap_k \
    --feature_layer $feature_layer "

cmd=$cmd$NORM_PIX_LOSS$USE_DECODER_FEATURE

echo "Running command: $cmd"
eval $cmd