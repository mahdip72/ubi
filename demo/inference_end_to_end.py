import argparse
import os
import numpy as np
import pandas as pd
import torch
import yaml
from dataloader import TestDataset
from torch.utils.data import DataLoader
from models import prepare_model
import time
from utils import load_tokenizer, test_gpu_cuda


def test(dataloader, model, device):
    model.eval()
    results = []
    start = time.time()
    for sequence in dataloader:
        with torch.no_grad():
            sequence = sequence.to(device)
            pred = model(sequence)
            results.append(torch.argmax(torch.softmax(pred.cpu(), dim=-1), dim=-1).numpy())
    end = time.time()

    print('time of process:', round(end - start, 4), 'seconds')

    final_results = np.concatenate(results)
    return final_results


def main(configs):
    test_gpu_cuda()

    if type(configs['fix_seed']) == int:
        print('check random seed:')
        torch.manual_seed(configs['fix_seed'])
        print(torch.rand(1, 3))
        torch.manual_seed(configs['fix_seed'])
        print(torch.rand(1, 3))

        torch.manual_seed(configs['fix_seed'])
        torch.random.manual_seed(configs['fix_seed'])
        np.random.seed(configs['fix_seed'])

    save_path = os.path.abspath(configs['save_path'])
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    data_path = os.path.join(configs['data_path'])

    test_df = pd.read_csv(data_path)

    tokenizer = load_tokenizer('.')

    test_data = TestDataset(test_df, tokenizer)
    test_dataloader = DataLoader(test_data, batch_size=configs['batch_size'], shuffle=False, drop_last=False)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # training configs
    print('configs:')
    for item, config in configs.items():
        print('\t', '-', item, ":", config)

    print('Evaluating the model on a test set ...')
    best_model = prepare_model(device, configs, tokenizer, print_params=False)
    best_model.load_state_dict(torch.load(configs['checkpoint_path']))
    results = test(test_dataloader, best_model, device)

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(save_path, 'results.csv'), index=False)
    print('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Predict ubi sites based using given checkpoint")
    parser.add_argument("--config_path", "-c", help="The location of config file", default='./end_to_end_config.yaml')
    args = parser.parse_args()
    config_path = args.config_path

    with open(config_path) as file:
        config_file = yaml.full_load(file)

    main(config_file)
    print('done!')

