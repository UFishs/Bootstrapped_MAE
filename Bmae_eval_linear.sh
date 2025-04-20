gpu_id=${1:-1}
bootstrap_k=${2:-4}
feature_layer=${3:-11}
USE_PIXEL_NORM=${4:-1}
use_decoder_feature=${5:-1}

echo gpu_id: $gpu_id
echo bootstrap_k: $bootstrap_k
echo feature_layer: $feature_layer
echo USE_PIXEL_NORM: $USE_PIXEL_NORM
echo use_decoder_feature: $use_decoder_feature

# choose GPU
export CUDA_VISIBLE_DEVICES=$gpu_id
echo "Using GPU: $gpu_id"

# define the network and dataset path
MODEL="vit_deit_tiny_patch4"  # choose model
DATA_PATH="./cifar-10-dataset"  # the path of CIFAR10
IMG_SIZE=32
NB_CLASSES=10

# define the path to save models and the log, and the save frequency
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
    use
else
    echo "Not using decoder feature"
    BASE_DIR=$BASE_DIR"-wo-decoder-feature"
fi

OUTPUT_DIR="./$BASE_PATH/$BASE_DIR/eval_linear/output_dir"
LOG_DIR="./$BASE_PATH/$BASE_DIR/eval_linear/log_dir"
# SAVE_FREQ=20

# hyperparameters
BATCH_SIZE=256
EPOCHS=100 # follow the requirement
LR=1
WEIGHT_DECAY=0

# finetuning
CHECK_POINT="./$BASE_PATH/$BASE_DIR/output_dir/checkpoint-199.pth"


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