import os
import argparse
import math
import logging
import numpy as np
import pickle
from functools import partial
from tqdm.auto import tqdm
from PIL import Image

import torch
import torch.nn.functional as F
from diffusers.models import AutoencoderKL
from torch_utils import misc
import dnnlib
from elatentlpips import VGG16_Latent, VGG16_Latent_BN, VGG16_Latent_GN

def parse_args():
    parser = argparse.ArgumentParser(description="Train latent vgg model")
    parser.add_argument("--eval_only", action="store_true", help="Whether to only evaluate the model")
    parser.add_argument("--eval_model_path", type=str, default="checkpoints/latent_vgg16_gn_sd1/checkpoint-ep=99/model.safetensors", help="path to the model checkpoint for evaluation")

    parser.add_argument("--dataset_path", type=str, default="/data2/img512-sd.zip", help="path to the training dataset")

    parser.add_argument("--ckpt", type=str, default=None, help="path to previous ckpt")

    parser.add_argument("--model_type", type=str, default="VGG16_Latent_GN", choices=["VGG16_Latent", "VGG16_Latent_BN", "VGG16_Latent_GN"], help="model type to use")
    parser.add_argument("--batch_size", type=int, default=32*8, help="batch size for training (effective batch size will be batch_size * gradient_accumulation_steps * num_processes")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=1e-1, help="learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="momentum for SGD optimizer")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="weight decay for optimizer")

    parser.add_argument("--dataloader_num_workers", type=int, default=4, help="number of workers for the dataloader")
    parser.add_argument("--num_epochs", type=int, default=100, help="number of epochs to train")
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"], help="Whether to use mixed precision, requires specific hardware and software")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to a checkpoint folder to resume from (e.g. checkpoint-1000)")
    parser.add_argument("--output_dir", type=str, default="checkpoints/latent_vgg16_gn_sd1", help="Directory for model predictions and checkpoints")
    parser.add_argument("--seed", type=int, default=4727, help="Random seed for reproducibility")

    args = parser.parse_args()
    return args

def preprocess(examples, transforms):
    images = [np.array(transforms(Image.open(img).convert("RGB"))).transpose(2,0,1) for img in examples["image"]]
    examples["image_tensors"] = 2 * (torch.tensor(np.stack(images)) / 255) - 1
    examples["label_tensors"] = torch.tensor(examples["label"])
    return examples

def collate_fn(examples):
    images = torch.stack([example["image_tensors"] for example in examples])
    labels = torch.stack([example["label_tensors"] for example in examples])
    return {"image": images, "label": labels}

def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-mse").to(device)

    if args.model_type == "VGG16_Latent":
        model = VGG16_Latent(in_channels=4).to(device)
    elif args.model_type == "VGG16_Latent_BN":
        model = VGG16_Latent_BN(in_channels=4).to(device)
    else:
        model = VGG16_Latent_GN(in_channels=4).to(device)
    
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    # Load dataset.
    print('Loading dataset...')
    dataset_kwargs = dnnlib.EasyDict(class_name='training.dataset.ImageFolderDataset', path=args.dataset_path, use_labels=True, xflip=False, cache=True)
    dataset_obj = dnnlib.util.construct_class_by_name(**dataset_kwargs) # subclass of training.dataset.Dataset
    dataset_sampler = misc.InfiniteSampler(dataset=dataset_obj, seed=args.seed)
    data_loader_kwargs = dnnlib.EasyDict(pin_memory=True, num_workers=1, prefetch_factor=2)
    dataset_iterator = iter(torch.utils.data.DataLoader(dataset=dataset_obj, sampler=dataset_sampler, batch_size=args.batch_size, **data_loader_kwargs))

    resume_pkl = "https://nvlabs-fi-cdn.nvidia.com/edm2/posthoc-reconstructions/edm2-img512-s-2147483-0.130.pkl"
    print(f'Loading encoder from "{resume_pkl}"...')
    with dnnlib.util.open_url(resume_pkl) as f:
        data = pickle.load(f)
    encoder = data.get('encoder', None)
    assert encoder is not None
    encoder.init(device)
    del data # conserve memory

    if args.ckpt is not None:
        print(f'Loading training state from "{args.ckpt}"...')
        data = torch.load(args.ckpt, map_location=torch.device('cpu'), weights_only=False)
        misc.copy_params_and_buffers(src_module=data['model'], dst_module=model, require_all=True)
        optimizer.load_state_dict(data['optimizer_state'])
        del data # conserve memory

        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr

    total_iter = int(len(dataset_obj) / args.batch_size * args.num_epochs)
    report_iter = int(100000 / args.batch_size)
    epoch_iter = int(len(dataset_obj) / args.batch_size)
    print("***** Running training *****")
    print(f"  Num examples = {len(dataset_obj)}")
    print(f"  Num Epochs = {args.num_epochs}")

    log_train_loss = 0.0
    log_train_accuracy = 0.0
    num_train_samples = 0

    for it in tqdm(range(1, 1 + total_iter)):
        model.train()

        images, labels = next(dataset_iterator)
        images = encoder.encode_latents(images.to(device))
        labels = labels.to(device)
        labels = torch.argmax(labels, dim=1)
    
        output = model(images)
        loss = F.cross_entropy(output, labels)
        acc1, = accuracy(output, labels)

        loss.backward()
        optimizer.step()
        
        log_train_loss += loss.item()
        log_train_accuracy += acc1.item()
        num_train_samples += 1
        optimizer.zero_grad()

        if it % report_iter == 0:
            # Calculate average metrics
            avg_train_loss = log_train_loss / num_train_samples
            avg_train_accuracy = log_train_accuracy / num_train_samples

            print(
                "train_loss: ", avg_train_loss,
                "train_accuracy: ", avg_train_accuracy,
                "lr: ", scheduler.get_last_lr()[0]
                )
            
            log_train_loss = 0.0
            log_train_accuracy = 0.0
            num_train_samples = 0

            torch.save(dict(model=model, optimizer_state=optimizer.state_dict()), os.path.join(f'vgg_ckpt_default/training-state-latest.pt'))
        
        if it % epoch_iter == 0:
            scheduler.step()

if __name__ == "__main__":
    args = parse_args()
    main(args)