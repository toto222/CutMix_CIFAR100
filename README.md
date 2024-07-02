# Preparation
## environment
```shell
pip install -r requirements.txt
```

# Train & Evaluation
There are a number of options that can be set, most of which can be used by default, which you can view in `train.py`.
## for train
```
python train_rn34.py --pretrained # if you want to finetune the pretrained ResNet34 model
python train_vits16.py --pretrained # if you want to finetune the pretrained ViT-S/16 model

python train_cutmix.py --net_type <model('vit' or 'rn34')> # If you want to train with CutMix method
```
Thanks to [`CutMix-PyTorch`](https://github.com/clovaai/CutMix-PyTorch) for the algorithmic implementation of CutMix

## for evaluation
```
python eval.py --file <model_ckp_path> --model <model('vit' or 'rn34')>  # please remember to correspond to the model and weight file
```
The weights after model training can be downloaded [`here`](https://drive.google.com/drive/folders/1l80-NafS600iEPAjuyusmL7crXnyifJd?usp=drive_link)


