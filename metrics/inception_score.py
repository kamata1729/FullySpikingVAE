import torch
from torch import nn
from torch.nn import functional as F
import torch.utils.data

from torchvision.models.inception import inception_v3

import numpy as np
from scipy.stats import entropy
from tqdm import tqdm

def get_inception_score(model, device=torch.device("cuda:0"), batch_size=256, batch_times=8):
    # function that accepts a latent and returns an image in range[0,255]
    def sample_from_vae(batch_size):
        sampled_x, _ = model.sample(batch_size)
        sampled_x = (sampled_x+1)/2 # 0 to 1
        sampled_x = 255 * sampled_x # 0 to 255
        if sampled_x.shape[1] == 1:
            sampled_x = sampled_x.repeat(1,3,1,1) # gray to RGB
        return sampled_x

    inception_mean, inception_std = calc_inception_score(sample_from_vae, device, batch_size=batch_size, batch_times=batch_times)
    return inception_mean, inception_std

def get_inception_score_ann(model, device=torch.device("cuda:0"), batch_size=256, batch_times=8):
    # function that accepts a latent and returns an image in range[0,255]
    def sample_from_vae(batch_size):
        sampled_x = model.sample(batch_size, device)
        sampled_x = (sampled_x+1)/2 # 0 to 1
        sampled_x = 255 * sampled_x # 0 to 255
        if sampled_x.shape[1] == 1:
            sampled_x = sampled_x.repeat(1,3,1,1) # gray to RGB
        return sampled_x

    inception_mean, inception_std = calc_inception_score(sample_from_vae, device, batch_size=batch_size, batch_times=batch_times)
    return inception_mean, inception_std
    

def calc_inception_score(img_generator, device=torch.device("cuda:0"), resize=True, batch_size=256, batch_times=4):
    """Computes the inception score of the generated images imgs

    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    device -- cuda device id
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """
    # Load inception model
    inception_model = inception_v3(pretrained=True, transform_input=False).to(device)
    inception_model.eval()
    up = nn.Upsample(size=(299, 299), mode='bilinear').to(device)
    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x, dim=-1).data.cpu().numpy()

    # Get predictions
    N = batch_size*batch_times
    preds = np.zeros((N, 1000))

    for i in range(batch_times):
        batch = img_generator(batch_size)
        preds[i*batch_size:(i+1)*batch_size ] = get_pred(batch)

    # Now compute the mean kl-div
    split_scores = []

    for k in tqdm(range(batch_times),desc="[Inception score]"):
        part = preds[k * (N // batch_times): (k+1) * (N // batch_times), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)
