import os
import numpy as np
import pandas as pd
import torch
from torchmetrics import Accuracy, Precision, Recall, F1Score, Specificity, AUC, MatthewsCorrCoef
from torchmetrics import ConfusionMatrix
from dataloader import CustomDataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from models import prepare_model
import time
from transformers import BertTokenizer
from torch import nn
from utils import load_tokenizer, test_gpu_cuda
from utils import write_test_results


def test(dataloader, model, tools):
    model.eval()
    test_loss, test_correct, counter = 0, 0, 0

    tools['accuracy'].reset()
    tools['precision'].reset()
    tools['recall'].reset()
    tools['f1'].reset()
    tools['specificity'].reset()
    tools['auc'].reset()
    tools['mcc'].reset()
    tools['matrix'].reset()

    start = time.time()
    with torch.no_grad():
        for x, p, y, w in dataloader:
            x, y, w = x.to(tools['device']), y.to(tools['device']), w.to(tools['device'])
            p = {'pssm': p['pssm'].to(tools['device']), 'physiochemical': p['physiochemical'].to(tools['device']),
                 'aac': p['aac'].to(tools['device']), 'dpc': p['dpc'].to(tools['device'])}
            pred = model(x, p)
            loss = tools['loss'](pred, y)
            loss = torch.mean(loss)

            test_loss += loss.item()
            tools['accuracy'].update(torch.argmax(pred, dim=1), y)
            tools['precision'].update(torch.argmax(pred, dim=1), y)
            tools['recall'].update(torch.argmax(pred, dim=1), y)
            tools['f1'].update(torch.argmax(pred, dim=1), y)
            tools['specificity'].update(torch.argmax(pred, dim=1), y)
            tools['auc'].update(torch.argmax(pred, dim=1), y)
            tools['mcc'].update(torch.argmax(pred, dim=1), y)
            tools['matrix'].update(torch.argmax(pred, dim=1), y)

            counter += 1

    end = time.time()
    epoch_time = int((end - start))

    test_loss /= counter
    test_correct /= counter

    write_test_results(test_loss, tools, tools['tensorboard_test'])

    print(f"{epoch_time}s - test_loss: {test_loss:>8f} test_acc: {tools['accuracy'].compute().item() * 100 :>0.2f}% "
          f"test_precision: {tools['precision'].compute().item() :>0.4f} "
          f"test_recall: {tools['recall'].compute().item() :>0.4f} "
          f"test_f1: {tools['f1'].compute().item() :>0.4f} "
          f"test_specificity: {tools['specificity'].compute().item() :>0.4f} "
          f"test_auc: {tools['auc'].compute().item() :>0.4f} "
          f"test_mcc: {tools['mcc'].compute().item() :>0.4f}")


def main(window_size, batch, seed):
    configs = {
        'fix_seed': seed,
        'batch': batch,
        'window_size': window_size,
        'backbone': 'lstm',
        'features': {'pssm': True, 'physiochemical': False, 'aac': False, 'dpc': False},
        'pretrained_weights': False,  # here it relates to preparing the tokenizer
        'checkpoint_path': './results/lstm_final_pssm',
        'save_path': './results/test',
        'model_name': f'{window_size}_{seed}',
        'test_path': '../../data/test data/processed/test_hybrid',
    }
    csv_name = 'test_hybrid'

    if configs['fix_seed']:
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

    if configs['pretrained_weights']:
        tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
    else:
        tokenizer = load_tokenizer('.')

    test_data = CustomDataset(test_df, tokenizer, configs, configs['pretrained_weights'])
    test_dataloader = DataLoader(test_data, batch_size=configs['batch'], shuffle=False, num_workers=2)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tools = {
        'loss': nn.CrossEntropyLoss(reduction='none'),
        'device': device,
        'save_path': save_path,
        'test_name': csv_name,
        'tensorboard_test': SummaryWriter(os.path.join(save_path, 'tensorboard_logs', csv_name)),
        'f1': F1Score(average='macro', num_classes=2).to(device),
        'precision': Precision(average='macro', num_classes=2).to(device),
        'recall': Recall(average='macro', num_classes=2).to(device),
        'accuracy': Accuracy().to(device),
        'specificity': Specificity(average='macro', num_classes=2).to(device),
        'auc': AUC(reorder=True).to(device),
        'mcc': MatthewsCorrCoef(num_classes=2).to(device),
        'matrix': ConfusionMatrix(task='binary', num_classes=2).to(device),
    }

    # training configs
    print('configs:')
    for item, config in configs.items():
        print('\t', '-', item, ":", config)

    print('Evaluating best model on the test set:')
    best_model = prepare_model(device, configs, tokenizer, print_params=False)
    best_model.load_state_dict(torch.load(os.path.join(checkpoint_path, "best_valid_f1_checkpoint.pth")))
    test(test_dataloader, best_model, tools)

    print(f'Window size {window_size} done')
    print('\n')

    del test_data, test_dataloader
    del test_df,  tools
    del best_model


if __name__ == '__main__':
    test_gpu_cuda()

    window_list = [(5, 4096), (7, 4096), (9, 4096), (15, 4096), (21, 4096), (27, 4096), (33, 4096), (45, 4096), (55, 4096),
                   (77, 4096), (99, 4096),
                   ]
    for s in [0, 1, 2, 3, 4]:
        for w, b in window_list[7:8]:
            main(w, b, s)

