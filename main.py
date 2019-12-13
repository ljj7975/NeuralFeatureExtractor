import argparse
import torch

from pprint import pprint

from utils import prepare_device
from utils import color_print as cp

import model as model_modules
import data_loader as data_loader_modules


def main(config):

    # Preapre model
    model_class = getattr(model_modules, config.model)

    # TODO:: read in model config and initialize
    model_config = {}
    model = model_class(**model_config)

    # TODO:: load trained model

    print("model\n", model)

    # Preapre data loader
    data_loader_class = getattr(data_loader_modules, config.data_loader)

    # TODO:: read in model config and initialize
    data_loader_config = {
        'data_dir': config.data_folder, 
        'batch_size': 512
    }
    data_loader = data_loader_class(**data_loader_config)

    print("data loader\n", data_loader)

    # Prepare extraction
    device, gpu_device_ids = prepare_device(config.num_gpu)

    if len(gpu_device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=gpu_device_ids)

    model.eval()
    model.to(device)

    # Extract features
    for batch_idx, (data, target) in enumerate(data_loader):
        data, target = data.to(device), target.to(device)
        output = model(data)

        # TODO:: Extract Features from given layer

        # TODO:: Save into a file

        # TODO:: loss calculation for progress
        import torch.nn.functional as F
        loss = F.nll_loss(output, target)
        print('Train Epoch: \tLoss: {:.6f}'.format(loss.item()))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Neural Feature Extractor')

    parser.add_argument('--model', default=None, type=str,
                      help='name of the model class')

    parser.add_argument('--trained_model', default=None, type=str,
                      help='path to the trained model')

    parser.add_argument('--data_loader', default=None, type=str,
                      help='name of data loader class')

    parser.add_argument('--data_folder', default=None, type=str,
                      help='path to the dataset')

    parser.add_argument('--num_gpu', default=0, type=int,
                      help='number of GPU to use (default:0)')

    config = parser.parse_args()

    main(config)
