# Bootstrapping MAE

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
We also provide a specific environment file for the experiments in this repository. You can create a conda environment using the following command, just in case.
```bash
conda env create -f environment.yml
```

## Training
