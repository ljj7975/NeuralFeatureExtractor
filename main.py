import argparse
import torch
from tqdm import tqdm

from pprint import pprint

from utils import prepare_device
from utils import color_print as cp

import model as model_modules
import data_loader as data_loader_modules
import file_handler as file_handler_modules


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

    # Preapre file handler
    
    file_handler = file_handler_modules.handler_mapping[config.file_type]('generated')

    feature_size = None

    print(f"file type: {config.file_type}")
    print(f"file handler\n", file_handler)

    # Prepare extraction
    device, gpu_device_ids = prepare_device(config.num_gpu)

    if len(gpu_device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=gpu_device_ids)

    model.eval()
    model.to(device)

    # Extract features
    total = 0
    for batch_idx, (data, target) in enumerate(tqdm(data_loader)):
        data, target = data.to(device), target.to(device)

        extracted_features = model(data)

        if feature_size is None:
            feature_size = list(extracted_features.size())[1:]

        extracted_features = extracted_features.data.tolist()

        target = target.unsqueeze(1).data.tolist()

        file_handler.add_sample(extracted_features, target)

        total += len(target)

        file_handler.flush()


    meta = {
        'feature_size': feature_size,
        'total': total
    }

    file_handler.generate_meta_file(meta)

        
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

    parser.add_argument('--file_type', default='csv', type=str,
                      help='type of output file')

    parser.add_argument('--num_gpu', default=0, type=int,
                      help='number of GPU to use (default:0)')

    config = parser.parse_args()

    main(config)
