import os
import os.path
import numpy as np
import logging
import argparse
import pycuda.driver as cuda

import torch
import torchvision
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils import clip_grad_value_

from torch.utils.tensorboard import SummaryWriter
from utils import AverageMeter
from utils import aboutCudaDevices

from datasets import load_dataset_ann
import models.ann_ae as ann_ae


max_accuracy = 0
min_loss = 1000

def train(network, trainloader, opti, epoch):
    loss_meter = AverageMeter()

    network = network.train()

    for batch_idx, (real_img, label) in enumerate(trainloader):         
        opti.zero_grad()
        real_img = real_img.to(device)
        recons, latent = network(real_img)
        loss = network.loss_function(recons, real_img)
        loss.backward()

        opti.step()

        loss_meter.update(loss.detach().cpu().item())

        print(f'Train[{epoch}/{max_epoch}] [{batch_idx}/{len(trainloader)}] Loss: {loss_meter.avg}')

        if batch_idx == len(trainloader)-1:
            os.makedirs(f'checkpoint/{args.name}/imgs/train/', exist_ok=True)
            torchvision.utils.save_image((real_img+1)/2, f'checkpoint/{args.name}/imgs/train/epoch{epoch}_input.png')
            torchvision.utils.save_image((recons+1)/2, f'checkpoint/{args.name}/imgs/train/epoch{epoch}_recons.png')
            writer.add_images('Train/input_img', (real_img+1)/2, epoch)
            writer.add_images('Train/recons_img', (recons+1)/2, epoch)

    logging.info(f"Train [{epoch}] Loss: {loss_meter.avg}")
    writer.add_scalar('Train/loss', loss_meter.avg, epoch)

    return loss_meter.avg

def test(network, trainloader, epoch):
    loss_meter = AverageMeter()

    network = network.eval()
    with torch.no_grad():
        for batch_idx, (real_img, label) in enumerate(trainloader):         
            real_img = real_img.to(device)
            #normalized_img = normalized_img.to(device)
            recons, latent = network(real_img)
            loss = network.loss_function(recons, real_img)

            loss_meter.update(loss.detach().cpu().item())

            print(f'Test[{epoch}/{max_epoch}] [{batch_idx}/{len(trainloader)}] Loss: {loss_meter.avg}')

            if batch_idx == len(trainloader)-1:
                os.makedirs(f'checkpoint/{args.name}/imgs/test/', exist_ok=True)
                torchvision.utils.save_image((real_img+1)/2, f'checkpoint/{args.name}/imgs/test/epoch{epoch}_input.png')
                torchvision.utils.save_image((recons+1)/2, f'checkpoint/{args.name}/imgs/test/epoch{epoch}_recons.png')
                writer.add_images('Test/input_img', (real_img+1)/2, epoch)
                writer.add_images('Test/recons_img', (recons+1)/2, epoch)

    logging.info(f"Test [{epoch}] Loss: {loss_meter.avg}")
    writer.add_scalar('Test/loss', loss_meter.avg, epoch)

    return loss_meter.avg


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('name', type=str)
    parser.add_argument('-dataset', type=str, required=True)
    parser.add_argument('-batch_size', type=int, default=250)
    parser.add_argument('-latent_dim', type=int, default=128)
    parser.add_argument('-checkpoint', action='store', dest='checkpoint', help='The path of checkpoint, if use checkpoint')
    parser.add_argument('-device', type=int, default=0)

    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        exit(0)

    if args.device is None:
        device = torch.device("cuda:0")
    else:
        device = torch.device(f"cuda:{args.device}")

    if args.dataset.lower() == 'mnist':     
        train_loader, test_loader = load_dataset_ann.load_mnist(args.batch_size)
        in_channels = 1 
        net = ann_ae.AE(in_channels, args.latent_dim)
    elif args.dataset.lower() == 'fashion':
        train_loader, test_loader = load_dataset_ann.load_fashionmnist(args.batch_size)
        in_channels = 1
        net = ann_ae.AE(in_channels, args.latent_dim)
    elif args.dataset.lower() == 'celeba':
        train_loader, test_loader = load_dataset_ann.load_celeba(args.batch_size)
        in_channels = 3
        net = ann_ae.AELarge(in_channels, args.latent_dim)
    elif args.dataset.lower() == 'cifar10':
        train_loader, test_loader = load_dataset_ann.load_cifar10(args.batch_size)
        in_channels = 3
        net = ann_ae.AE(in_channels, args.latent_dim)
    else:
        raise ValueError("invalid dataset")

    net = net.to(device)

    os.makedirs(f'checkpoint/{args.name}', exist_ok=True)

    writer = SummaryWriter(log_dir=f'checkpoint/{args.name}/tb')
    logging.basicConfig(filename=f'checkpoint/{args.name}/{args.name}.log', level=logging.INFO)
    
    logging.info(args)

    if torch.cuda.is_available():
        cuda.init()
        c_device = aboutCudaDevices()
        print(c_device.info())
        print("selected device: ", args.device)
    else:
        raise Exception("only support gpu")


    if args.checkpoint is not None:
        checkpoint_path = args.checkpoint
        checkpoint = torch.load(checkpoint_path)
        net.load_state_dict(checkpoint)  

    optimizer = torch.optim.AdamW(net.parameters(), lr=0.001, betas=(0.9, 0.999))
    best_loss = 1e8
    max_epoch = 150
    for e in range(max_epoch):
        train_loss = train(net, train_loader, optimizer, e)
        test_loss = test(net, test_loader, e)
        torch.save(net.state_dict(), f'checkpoint/{args.name}/checkpoint.pth')
        if test_loss < best_loss:
            best_loss = test_loss
            torch.save(net.state_dict(), f'checkpoint/{args.name}/best.pth')
        
    writer.close()