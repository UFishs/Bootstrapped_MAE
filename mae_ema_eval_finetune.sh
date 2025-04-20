gpu_id=${1:-1}
ema_decay=${2:-0.999}
ema_warmup=${3:-40}
use_decoder_feature=${4:-1}

# choose GPU
export CUDA_VISIBLE_DEVICES=$gpu_id
echo "Using GPU: $gpu_id"
echo "EMA decay: $ema_decay"
echo "EMA warmup: $ema_warmup"
echo "Use decoder feature: $use_decoder_feature"

# define the network and dataset path
MODEL="vit_deit_tiny_patch4"  # choose model
DATA_PATH="./cifar-10-dataset"  # the path of CIFAR10
IMG_SIZE=32
NB_CLASSES=10

# define the path to save models and the log, and the save frequency
BASE_PATH="MAE-EMA"
BASE_DIR="decay-$ema_decay-warmup-$ema_warmup"

if [ $use_decoder_feature = 1 ]; then
    echo "Using decoder feature"
else
    echo "Not using decoder feature"
    BASE_DIR="$BASE_DIR-wo-decoder-feature"
fi

OUTPUT_DIR="./$BASE_PATH/$BASE_DIR/eval_finetune/output_dir"
LOG_DIR="./$BASE_PATH/$BASE_DIR/eval_finetune/log_dir"
# SAVE_FREQ=20

# hyperparameters
BATCH_SIZE=256
EPOCHS=100 # follow the requirement
LR=1e-3

# finetuning
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