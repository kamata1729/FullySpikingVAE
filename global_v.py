import torch


dtype = None
n_steps = None
network_config = None
layer_config = None
devices = None
params = {}

def init(n_config, devs):
    global dtype, devices, n_steps, tau_s, network_config, layer_config, params
    dtype = torch.float32
    devices = devs
    network_config = n_config
    network_config['batch_size'] = network_config['batch_size'] * len(devices)
    network_config['lr'] = network_config['lr'] * len(devices) * network_config['batch_size'] / 250
    layer_config = {'threshold': 0.2}
    n_steps = network_config['n_steps']
    
    
