gpu_id=${1}
USE_PIXEL_NORM=${2:-1}
USE_SOFT_VERSION=${3:-0}
feature_layer=${4:-11}

# choose GPU
export CUDA_VISIBLE_DEVICES=$gpu_id
echo "Using GPU: $gpu_id"
echo "Using pixel norm: $USE_PIXEL_NORM"
echo "Using soft version: $USE_SOFT_VERSION"
echo "Feature layer: $feature_layer"

MODEL="deit_tiny_patch4"  # choose model
DATA_PATH="./cifar-10-dataset"  # the path of CIFAR10

# define the path to save models and the log, and the save frequency
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
    SOFT_VERSION=" --soft_version --use_ema --ema_decay 0.999"
    FEATURE_LAYER_NUM=" --feature_layer $feature_layer"
else
    echo "Not using soft version"
    BASE_DIR="$BASE_DIR"
    SOFT_VERSION=""
    FEATURE_LAYER_NUM=""
fi


OUTPUT_DIR="./$BASE_DIR/pretrain/output_dir"
LOG_DIR="./$BASE_DIR/pretrain/log_dir"

# hyperparameters
BATCH_SIZE=256
EPOCHS=200
LR=1e-4
WEIGHT_DECAY=0.05

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
    --input_size 32"

cmd=$cmd$NORM_PIX_LOSS$SOFT_VERSION$FEATURE_LAYER_NUM

echo "Running command: $cmd"
eval $cmd