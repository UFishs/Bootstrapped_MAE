gpu_id=${1:-1}
bootstrap_k=${2:-2}
feature_layer=${3:-11}
USE_PIXEL_NORM=${4:-1}
use_decoder_feature=${5:-1}

export CUDA_VISIBLE_DEVICES=$gpu_id
echo "Using GPU: $gpu_id"

MODEL="vit_deit_tiny_patch4"
DATA_PATH="./cifar-10-dataset"
IMG_SIZE=32
NB_CLASSES=10


BASE_PATH="bootstrap"

if [ $USE_PIXEL_NORM = 1 ]; then
    echo "Using pixel norm"
    BASE_DIR="MAE-$bootstrap_k-feature-$feature_layer"
else
    echo "Not using pixel norm"
    BASE_DIR="MAE-$bootstrap_k-feature-$feature_layer-wo-pixelnorm"
fi

if [ $use_decoder_feature = 1 ]; then
    echo "Using decoder feature"
else
    echo "Not using decoder feature"
    BASE_DIR=$BASE_DIR"-wo-decoder-feature"
fi

OUTPUT_DIR="./$BASE_PATH/$BASE_DIR/eval_finetune/output_dir"
LOG_DIR="./$BASE_PATH/$BASE_DIR/eval_finetune/log_dir"

BATCH_SIZE=256
EPOCHS=100 # follow the requirement
LR=1e-3

CHECK_POINT="./$BASE_PATH/$BASE_DIR/output_dir/checkpoint-199.pth"


python main_finetune.py \
    --model $MODEL \
    --data_path $DATA_PATH \
    --output_dir $OUTPUT_DIR \
    --log_dir $LOG_DIR \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --lr $LR \
    --device cuda \
    --nb_classes $NB_CLASSES \
    --finetune $CHECK_POINT \
    --input_size $IMG_SIZE