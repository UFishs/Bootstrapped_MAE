# Bootstrapped MAE

This repository implements Bootstrapped MAE, EMA-MAE, and its soft version. The code is based on the original MAE repository: [MAE GitHub Repo](https://github.com/facebookresearch/mae).

## Installation
Same as MAE. Please refer to the original MAE repository for installation instructions. But one may encounter some problems, such as
```bash
ImportError: cannot import name 'container_abcs' from 'torch._six'
```
Then you can try to change the package file as below:
```python
import collections.abc as container_abcs
```
We also provide a specific environment file for the experiments in this repository. You can create a conda environment using the following command, just in case. It will encounter the same problem as above, do as mentioned above.
```bash
conda env create -f environment.yml
```

## Training
For original MAE training, use the following command:
```bash
sh mae_train.sh gpu_id 1 0 0
sh mae_eval_linear.sh gpu_id 1 0 0
sh mae_eval_finetune.sh gpu_id 1 0 0
```

For BMAE training, use the following command:
```bash
sh Bmae_train.sh gpu_id bootstrap_k feature_layer 1 0
sh Bmae_eval_linear.sh gpu_id bootstrap_k feature_layer 1 0
sh Bmae_eval_finetune.sh gpu_id bootstrap_k feature_layer 1 0
```
or just use scripts in the `scripts` folder.
```bash
sh scripts/train_and_eval.sh gpu_id bootstrap_k feature_layer use_pixel_norm use_decoder_feature
```

For EMA-MAE training, use the following command:
```bash
sh mae_ema_train.sh gpu_id ema_decay ema_warmup use_decoder_feature
sh mae_ema_eval_linear.sh gpu_id ema_decay ema_warmup use_decoder_feature
sh mae_ema_eval_finetune.sh gpu_id ema_decay ema_warmup use_decoder_feature
```
or just use scripts in the `scripts` folder.
```bash
sh scripts/train_and_eval_ema.sh gpu_id ema_decay ema_warmup use_decoder_feature
```

For soft version training, use the following command:
```bash
sh mae_train.sh gpu_id 1 1 feature_layer
sh mae_eval_linear.sh gpu_id 1 1 feature_layer
sh mae_eval_finetune.sh gpu_id 1 1 feature_layer
```
or just use scripts in the `scripts` folder.
```bash
sh scripts/train_and_eval_soft.sh gpu_id 1 1 feature_layer
```

## Contributors

- Zihang Rui