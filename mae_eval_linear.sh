gpu_id=${1}
USE_PIXEL_NORM=${2:-1}
USE_SOFT_VERSION=${3:-0}
feature_layer=${4:-11}

export CUDA_VISIBLE_DEVICES=$gpu_id
echo "Using GPU: $gpu_id"
echo "Using pixel norm: $USE_PIXEL_NORM"
echo "Using soft version: $USE_SOFT_VERSION"
echo "Feature layer: $feature_layer"

MODEL="vit_deit_tiny_patch4"
DATA_PATH="./cifar-10-dataset"
IMG_SIZE=32
NB_CLASSES=10

BASE_DIR="MAE-1"
if [ $USE_PIXEL_NORM = 1 ]; then
    echo "Using pixel norm"
    BASE_DIR="$BASE_DIR-pixelnorm"
    NORM_PIX_LOSS=" --norm_pix_loss"
else
    echo "Not using pixel norm"
    BASE_DIR="$BASE_DIR"
    NORM_PIX_LOSS=""
fi

if [ $USE_SOFT_VERSION = 1 ]; then
    echo "Using soft version"
    BASE_DIR="$BASE_DIR-soft-f$feature_layer"
    SOFT_VERSION=" --soft_version"
    FEATURE_LAYER_NUM=" --feature_layer $feature_layer"
else
    echo "Not using soft version"
    BASE_DIR="$BASE_DIR"
    SOFT_VERSION=""
    FEATURE_LAYER_NUM=""
fi


OUTPUT_DIR="./$BASE_DIR/eval_linear/output_dir"
LOG_DIR="./$BASE_DIR/eval_linear/log_dir"

BATCH_SIZE=256
EPOCHS=100 # follow the requirement
LR=1
WEIGHT_DECAY=0

CHECK_POINT="./$BASE_DIR/pretrain/output_dir/checkpoint-199.pth"

python main_linprobe.py \
    --model $MODEL \
    --data_path $DATA_PATH \
    --output_dir $OUTPUT_DIR \
    --log_dir $LOG_DIR \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --lr $LR \
    --weight_decay $WEIGHT_DECAY \
    --device cuda \
    --nb_classes $NB_CLASSES \
    --finetune $CHECK_POINT 