# Consistency Mid-Training (CMT)

The code follows [ECT](https://github.com/locuslab/ect/)'s format. The dependencies are the same as ECT. Before training, please put the dataset zip file under the datasets folder.

The main branch is for unconditional CIFAR10, FFHQ, and AFHQv2.

## CIFAR10

### CIFAR10 Dataset

Download the dataset zip from Google Drive and put it under the datasets folder, i.e., datasets/cifar10-32x32.zip. Otherwise, one can obtain the dataset by following [EDM's Preparing Dataset](https://github.com/NVlabs/edm#preparing-datasets) section.

https://drive.google.com/drive/folders/1EPl9fc8XOgc135x8l0EGwTsHLf0ip_pQ?usp=drive_link.

### CMT Mid-Training Stage

```bash
bash run_dcp_cifar.sh <NGPUs> <PORT> --desc bs128.300k
```

The command is mid-training with a 38.4Mimgs budget in 300k iterations. The resulting model checkpoint, logs, and all other details are available at https://drive.google.com/drive/folders/1Kc5pGlR-4aeXYYGEM6XzN9r4svmc6eRV?usp=drive_link.

### CMT Post-Training Stage with ECT

```bash
bash run_dcp+ecm_cifar.sh <NGPUs> <PORT> --desc bs128.100k
```

It uses \Delta t = 1/4096 throughout the 12.8Mimgs/100K iterations training and uses the mid-training checkpoint. The resulting model is a **SOTA CM with 2.74 one-step FID and 1.97 two-step FID**. The resulting SOTA model checkpoint, logs, and all other details are available at https://drive.google.com/drive/folders/1Di17c-QdUV9yfC9rpZgjg-I2Tcjp-WLu?usp=drive_link.


## FFHQ (64 x 64)

### FFHQ Dataset

Download the dataset zip from Google Drive and put it under the datasets folder, i.e., datasets/ffhq-64x64.zip. Otherwise, one can obtain the dataset by following [EDM's Preparing Dataset](https://github.com/NVlabs/edm#preparing-datasets) section.

We provide the processed FFHQ dataset in https://drive.google.com/drive/folders/1EPl9fc8XOgc135x8l0EGwTsHLf0ip_pQ?usp=drive_link.

### CMT Mid-Training Stage

```bash
bash run_dcp_ffhq.sh <NGPUs> <PORT> --desc bs128.300k
```

The command is mid-training with a 38.4 Mimgs budget in 300k iterations. The resulting model checkpoint, logs, and all other details are available at https://drive.google.com/drive/folders/1UlPyNUEAZ5aM8OKc01F1Nw4DvoFxZQbf?usp=drive_link.

### CMT Post-Training Stage with ECT

```bash
bash run_dcp+ecm_ffhq.sh <NGPUs> <PORT> --desc bs128.100k
```

It uses \Delta t = 1/512 throughout the 12.8Mimgs/100K training and uses the mid-training checkpoint. The resulting model is a **SOTA CM with 3.89 one-step FID and 2.75 two-step FID**. The resulting SOTA model checkpoint, logs, and all other details are available at https://drive.google.com/drive/folders/1m4v8cqcd1nelBZdMAUxxfj9EHOTlkReW?usp=drive_link.


## AFHQv2 (64 x 64)

### AFHQ Dataset

Download the dataset zip from Google Drive and put it under the datasets folder, i.e., datasets/afhqv2-64x64.zip. Otherwise, one can obtain the dataset by following [EDM's Preparing Dataset](https://github.com/NVlabs/edm#preparing-datasets) section.

We provide the processed AFHQv2 dataset in https://drive.google.com/drive/folders/1EPl9fc8XOgc135x8l0EGwTsHLf0ip_pQ?usp=drive_link.

### CMT Mid-Training Stage

```bash
bash run_dcp_afhq.sh <NGPUs> <PORT> --desc bs128.300k
```

Same hyperparameters as FFHQ. The results are at https://drive.google.com/drive/folders/1PezXgBQLLvNib_iKpYrPj26NX59Mml0y?usp=drive_link.

### CMT Post-Training Stage with ECT

```bash
bash run_dcp+ecm_afhq.sh <NGPUs> <PORT> --desc bs128.100k
```

1/2-step FID=3.28/2.34. The results are at https://drive.google.com/drive/folders/1Bq7g7l-ErK_7EsrAomsGCaR7QMLVU_TJ?usp=drive_link.

## Evaluations

First, one needs to download the pretrained checkpoints from the above links.

Then, run the following:

```bash
bash eval_ecm.sh <NGPUs> <PORT> --resume <CKPT_PATH>
```

```bash 
bash eval_ecm_ffhq.sh <NGPUs> <PORT> --resume <CKPT_PATH>
```

## ImageNet 64x64, 256x256, and 512x512

Please refer to other branches for the three resolutions of ImageNet.







