import argparse
import torch
from tqdm import tqdm

from pprint import pprint

from utils import prepare_device, load_json
from utils import color_print as cp

import model as model_modules
import data_loader as data_loader_modules
import file_handler as file_handler_modules

def extract_feature(model, device, data_loader, file_handler):
    # Extract features
    feature_size = None
    total = 0
    for (data, target) in tqdm(data_loader):
        data, target = data.to(device), target.to(device)

        extracted_features = model(data)

        if feature_size is None:
            feature_size = list(extracted_features.size())[1:]

        extracted_features = extracted_features.data.tolist()

        target = target.unsqueeze(1).data.tolist()

        file_handler.add_sample(extracted_features, target)

        total += len(target)

        file_handler.flush()

    return feature_size, total

def main(config):

    # Preapre model
    
    model_config = load_json(config.model_config)
    model_class = getattr(model_modules, model_config["model"])
    model = model_class(**model_config["config"])

    trained_model = model_config["trained_model"]
    cp.print_green(f"pretrained model: {trained_model}")
    model.load_state_dict(torch.load(trained_model), strict=False)

    cp.print_green("model:\n", model)


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
        cp.print_green(f"utilizing gpu devices : {gpu_device_ids}")
        model = torch.nn.DataParallel(model, device_ids=gpu_device_ids)

    model.eval()
    model.to(device)

    # Extract train features

    train_data_loader = data_loader_class(**data_config["train_config"])
    train_file_handler = file_handler_modules.handler_mapping[config.file_type](output_path + "/train")

    feature_size, total = extract_feature(model, device, train_data_loader, train_file_handler)

    meta = {
        'feature_size': feature_size,
        'total': total
    }

    cp.print_green('train meta file:\n', meta)

    train_file_handler.generate_meta_file(meta)

    del train_file_handler, train_data_loader


    # Extract test features

    test_data_loader = data_loader_class(**data_config["test_config"])
    test_file_handler = file_handler_modules.handler_mapping[config.file_type](output_path + "/test")

    feature_size, total = extract_feature(model, device, test_data_loader, test_file_handler)

    meta = {
        'feature_size': feature_size,
        'total': total
    }

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
                      help='number of GPU to use (default:0)')

    config = parser.parse_args()

    main(config)
