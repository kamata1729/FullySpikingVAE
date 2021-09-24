from cleanfid import fid

def get_clean_fid_score(model, dataset, device, num_gen=5000):
    if dataset.lower() == "mnist":
        dataset_name = 'mnist_test'
    elif dataset.lower() == "fashionmnist":
        dataset_name = 'fashion_test'
    elif dataset.lower() == 'celeba':
        dataset_name = 'celeba_valid'
    elif dataset.lower() == 'cifar10':
        dataset_name = 'cifar10_test'
    else:
        raise ValueError()

    # function that accepts a latent and returns an image in range[0,255]
    def sample_from_vae(z):
        """
        z : dummy latent value (batch_size, z_dim)
        """
        batch_size = z.shape[0]
        sampled_x, _ = model.sample(batch_size)
        sampled_x = (sampled_x+1)/2 # 0 to 1
        sampled_x = 255 * sampled_x # 0 to 255
        if sampled_x.shape[1] == 1:
            sampled_x = sampled_x.repeat(1,3,1,1) # gray to RGB
        return sampled_x

    score = fid.compute_fid(gen=sample_from_vae, dataset_name=dataset_name,
            num_gen=num_gen, dataset_split="custom", batch_size=256, device=device, z_dim=2)

    return score

def get_clean_fid_score_ann(model, dataset, device, num_gen=5000):
    if dataset.lower() == "mnist":
        dataset_name = 'mnist_test'
    elif dataset.lower() == "fashionmnist":
        dataset_name = 'fashion_test'
    elif dataset.lower() == 'celeba':
        dataset_name = 'celeba_valid'
    elif dataset.lower() == 'cifar10':
        dataset_name = 'cifar10_test'
    else:
        raise ValueError()

    # function that accepts a latent and returns an image in range[0,255]
    def sample_from_vae(z):
        """
        z : dummy latent value (batch_size, z_dim)
        """
        batch_size = z.shape[0]
        sampled_x= model.sample(batch_size, device)
        sampled_x = (sampled_x+1)/2 # 0 to 1
        sampled_x = 255 * sampled_x # 0 to 255
        if sampled_x.shape[1] == 1:
            sampled_x = sampled_x.repeat(1,3,1,1) # gray to RGB
        return sampled_x

    score = fid.compute_fid(gen=sample_from_vae, dataset_name=dataset_name,
            num_gen=num_gen, dataset_split="custom", batch_size=256, device=device, z_dim=2)

    return score