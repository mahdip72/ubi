import os
import numpy as np
import pandas as pd
import torch
from dataloader import TestDataset
from torch.utils.data import DataLoader
from models import prepare_model
import time
from utils import load_tokenizer, test_gpu_cuda


def test(dataloader, model, tools):
    model.eval()
    results = []
    start = time.time()
    for sequence in dataloader:
        with torch.no_grad():
            sequence = sequence.to(tools['device'])
            pred = model(sequence)
            results.append(torch.argmax(torch.softmax(pred.cpu(), dim=-1), dim=-1).numpy())
    end = time.time()

    print('time of process:', end - start)

    final_results = np.concatenate(results)
    return final_results


def main(window_size, batch, seed):
    configs = {
        'fix_seed': seed,
        'batch': batch,
        'window_size': window_size,
        'backbone': 'lstm',
        'pretrained_weights': False,  # here it relates to preparing the tokenizer
        'checkpoint_path': r'S:\Programming\Ubiquitylation\train\end_to_end\results\lstm_final_3',
        'save_path': './results/bert_final',
        'model_name': f'{window_size}_{seed}',
        'test_path': r'S:\Programming\Ubiquitylation\data\test data\processed\test_hybrid',
    }
    csv_name = os.path.basename(configs['test_path'])

    if type(configs['fix_seed']) == int:
        print('check random seed:')
        torch.manual_seed(configs['fix_seed'])
        print(torch.rand(1, 3))
        torch.manual_seed(configs['fix_seed'])
        print(torch.rand(1, 3))

        torch.manual_seed(configs['fix_seed'])
        torch.random.manual_seed(configs['fix_seed'])
        np.random.seed(configs['fix_seed'])

    model_name = configs['model_name']

    checkpoint_path = os.path.join(configs['checkpoint_path'], model_name)
    save_path = os.path.abspath(configs['save_path'])
    save_path = os.path.join(save_path, model_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    test_path = os.path.join(configs['test_path'], f'{configs["window_size"]}.csv')

    test_df = pd.read_csv(test_path)

    tokenizer = load_tokenizer('.')

    test_data = TestDataset(test_df, tokenizer)
    test_dataloader = DataLoader(test_data, batch_size=configs['batch'], shuffle=False, drop_last=False)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tools = {
        'device': device,
        'save_path': save_path,
        'test_name': csv_name,
    }

    # training configs
    print('configs:')
    for item, config in configs.items():
        print('\t', '-', item, ":", config)

    print('Evaluating the model on a test set:')
    best_model = prepare_model(device, configs, tokenizer, print_params=False)
    best_model.load_state_dict(torch.load(os.path.join(checkpoint_path, "best_valid_f1_checkpoint.pth")))
    results = test(test_dataloader, best_model, tools)

    df = pd.DataFrame(results)
    df.to_csv('./results/results.csv', index=False)
    print(f'Window size {window_size} done')
    print('\n')


if __name__ == '__main__':
    test_gpu_cuda()

    window_list = [(55, 128),
                   (77, 128),
                   ]
    for s in [0, 1, 2, 3, 4]:
        for w, b in window_list:
            main(w, b, s)

