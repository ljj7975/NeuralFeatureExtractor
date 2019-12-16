import argparse
import torch
import numpy as np
import random

from tqdm import tqdm

from pprint import pprint

from utils import prepare_device, load_json
from utils import color_print as cp

import model as model_modules
import data_loader as data_loader_modules
import file_handler as file_handler_modules

def extract_feature(model, activation, data_loader, file_handler, device):
    # Extract features
    feature_size = None
    total = 0
    min_value = float("inf")
    max_value = float("-inf")
    labels = set([])
    for (data, target) in tqdm(data_loader):
        data, target = data.to(device), target.to(device)

        extracted_features = model(data)
        if activation is not None:
            extracted_features = activation(extracted_features)

        if feature_size is None:
            feature_size = list(extracted_features.size())[1:]

        batch_min_value = extracted_features.min().item()
        if batch_min_value < min_value:
            min_value = batch_min_value

        batch_max_value = extracted_features.max().item()
        if batch_max_value > max_value:
            max_value = batch_max_value

        extracted_features = extracted_features.data.tolist()

        target = target.data.tolist()
        labels = labels.union(set(target))

        target = np.expand_dims(target, axis=1)
        file_handler.add_sample(extracted_features, target)

        total += len(target)

        file_handler.flush()

    meta = {
        'feature_size': feature_size,
        'total': total,
        'min': min_value,
        'max': max_value,
        'labels': list(labels)
    }

    return meta

def main(config):

    # set random seed

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)


    # Preapre model

    model_config = load_json(config.model_config)
    model_class = getattr(model_modules, model_config["model"])
    model = model_class(model_config["config"])

    trained_model = model_config["trained_model"]
    cp.print_green(f"pretrained model: {trained_model}")

    model.load_state_dict(torch.load(trained_model), strict=False)
    cp.print_green("model:\n", model)

    activation = None
    if "activation" in model_config:
        activation = getattr(torch, model_config["activation"], None)
    cp.print_green("activation: ", type(activation).__name__)


    # Prepare DataLoader

    data_config = load_json(config.data_config)
    data_loader_class = getattr(data_loader_modules, data_config["data_loader"])


    # Preapre file handler

    output_path = config.output_folder + "/" + type(model).__name__

    cp.print_green(f"file type: {config.file_type}")
    cp.print_green(f"output folder: {output_path}")


    # Prepare extraction

    device, gpu_device_ids = prepare_device(config.num_gpu)

    if len(gpu_device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=gpu_device_ids)

    if "cuda" in str(device):
        cp.print_green(f"utilizing gpu devices : {gpu_device_ids}")
        torch.cuda.manual_seed(config.seed)

    model.eval()
    model.to(device)


    # Extract train features

    train_data_loader = data_loader_class(data_config["train_config"])
    train_file_handler = file_handler_modules.handler_mapping[config.file_type](output_path + "/train")

    meta = extract_feature(model, activation, train_data_loader, train_file_handler, device)

    cp.print_green('train meta file:\n', meta)

    train_file_handler.generate_meta_file(meta)

    del train_file_handler, train_data_loader


    # Extract test features

    test_data_loader = data_loader_class(data_config["test_config"])
    test_file_handler = file_handler_modules.handler_mapping[config.file_type](output_path + "/test")

    meta = extract_feature(model, activation, test_data_loader, test_file_handler, device)

    cp.print_green('test meta file:\n', meta)

    test_file_handler.generate_meta_file(meta)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Neural Feature Extractor')

    parser.add_argument('--model_config', default=None, required=True, type=str,
                      help='path to model config')

    parser.add_argument('--data_config', default=None, required=True, type=str,
                      help='path to data config')

    parser.add_argument('--output_folder', default='generated', type=str,
                      help='path to store generated dataset')

    parser.add_argument('--file_type', default='csv', type=str,
                      help='type of output file')

    parser.add_argument('--num_gpu', default=0, type=int,
                      help='number of GPU to use (default: 0)')

    parser.add_argument('--seed', default=10, type=int,
                      help='random seed (default: 10')

    config = parser.parse_args()

    main(config)
