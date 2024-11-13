import argparse

import torch
import yaml

from pytorch3dunet.unet3d import utils

logger = utils.get_logger('ConfigLoader')


def load_config():
    parser = argparse.ArgumentParser(description='UNet3D')
    parser.add_argument('--config', type=str, help='Path to the YAML config file', required=True)
    # OOB
    parser.add_argument('--batch_size', default=1, type=int, help='batch size')
    parser.add_argument('--precision', default="float32", type=str, help='precision')
    parser.add_argument('--channels_last', default=1, type=int, help='Use NHWC or not')
    parser.add_argument('--jit', action='store_true', default=False, help='enable JIT')
    parser.add_argument('--profile', action='store_true', default=False, help='collect timeline')
    parser.add_argument('--num_iter', default=200, type=int, help='test iterations')
    parser.add_argument('--num_warmup', default=20, type=int, help='test warmup')
    parser.add_argument('--device', default='cpu', type=str, help='cpu, cuda or xpu')
    parser.add_argument('--nv_fuser', action='store_true', default=False, help='enable nv fuser')
    parser.add_argument('--ipex', action='store_true', default=False)
    args = parser.parse_args()
    config = yaml.safe_load(open(args.config, 'r'))
    # Get a device to train on
    device_str = config.get('device', None)
    if device_str is not None:
        logger.info(f"Device specified in config: '{device_str}'")
        if device_str.startswith('cuda') and not torch.cuda.is_available():
            logger.warning('CUDA not available, using CPU')
            device_str = 'cpu'
    else:
        device_str = "cuda:0" if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using '{device_str}' device")

    device_str = args.device
    device = torch.device(device_str)
    config['device'] = device
    config['device_str'] = device_str
    config['loaders']['batch_size'] = args.batch_size
    config['precision'] = args.precision
    config['channels_last'] = args.channels_last
    config['jit'] = args.jit
    config['profile'] = args.profile
    config['num_iter'] = args.num_iter
    config['num_warmup'] = args.num_warmup
    config['nv_fuser'] = args.nv_fuser
    config['ipex'] = args.ipex
    return config

def _load_config_yaml(config_file):
    return yaml.safe_load(open(config_file, 'r'))
