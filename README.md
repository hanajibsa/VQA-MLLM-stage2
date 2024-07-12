# Improving VQA Using MLLM
![image](https://github.com/pej0918/BLIVA/assets/79118751/d3de9fc7-cbda-4fb1-ba88-202ac09ee28f)


## Train

After downloading the training datasets and specify their path in [dataset configs](bliva/configs/datasets/), we are ready for training. We utilized 8x A6000 Ada in our experiments. Please adjust hyperparamters according to your GPU resources. It may take transformers around 2 minutes to load the model, give some time for the model to start training. Here we give an example of traning BLIVA Vicuna version, the Flant5 version follows the same format.

0. Setting Environments
```Shell
conda create -n fusion python=3.9
```
```Shell
git clone 
```
```Shell
cd BLIVA
```
```Shell
pip install -e .
```
<br>
if packaging error, then
```Shell
pip install setuptools==69.5.1
```
<br>

1. Pretraining of visual assistant branch

```Shell
python train.py --cfg-path train_configs/pretrain_bliva_vicuna.yaml
```

2. Instruction Finetuning BLIVA

```Shell
torchrun --nnodes=1 --nproc_per_node=8 \
    train.py \
    --cfg-path train_configs/finetune_bliva_vicuna.yaml
```
