import os
import numpy as np
import pandas as pd
import torch
from torchmetrics import Accuracy, Precision, Recall, F1Score, Specificity, AUC, MatthewsCorrCoef
from dataloader import CustomDataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from models import prepare_model
import time
from utils import SAM
from transformers import BertTokenizer
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
from torch import nn
from utils import load_tokenizer, test_gpu_cuda, compute_sam, write_tensorboard, print_log, append_list_as_row
from utils import write_test_results
from adabelief_pytorch import AdaBelief


def train(dataloader, model, tools, e):
    num_batches = len(dataloader)
    model.train()
    train_loss, train_correct, counter = 0, 0, 0

    tools['accuracy'].reset()
    tools['precision'].reset()
    tools['recall'].reset()
    tools['f1'].reset()
    start = time.time()
    for batch, (x, p, y, w) in enumerate(dataloader):
        x, y, w = x.to(tools['device']), y.to(tools['device']), w.to(tools['device'])
        p = {'pssm': p['pssm'].to(tools['device']), 'physiochemical': p['physiochemical'].to(tools['device']),
             'aac': p['aac'].to(tools['device']), 'dpc': p['dpc'].to(tools['device'])}
        tools['optimizer'].zero_grad()
        with torch.cuda.amp.autocast(enabled=tools['mixed_precision']):
            pred = model(x, p)
            assert pred.dtype is torch.float16
            loss = tools['loss'](pred, y) * w
            assert loss.dtype is torch.float32
            loss = torch.mean(loss)

        loss_value = loss.item()
        train_loss += loss_value

        # Backpropagation
        if tools['mixed_precision']:
            if tools['sam']:
                compute_sam(tools, loss, model, x, p, y, w)
            else:
                tools['scaler'].scale(loss).backward()
                tools['scaler'].unscale_(tools['optimizer'])
                torch.nn.utils.clip_grad_norm_(model.parameters(), tools['grad_clip'])
                tools['scaler'].step(tools['optimizer'])
                tools['scaler'].update()
        else:
            loss.backward()
            tools['optimizer'].step()

        tools['lr_scheduler'].step()

        counter += 1
        if counter % 10 == 0:
            tools['tensorboard_train'].add_scalar('lr', tools['lr_scheduler'].get_lr()[0], (e * num_batches + counter))

        acc = tools['accuracy'].forward(torch.argmax(pred, dim=1), y).item() * 100
        precision = tools['precision'].forward(torch.argmax(pred, dim=1), y).item()
        recall = tools['recall'].forward(torch.argmax(pred, dim=1), y).item()
        f1 = tools['f1'].forward(torch.argmax(pred, dim=1), y).item()

        # log every s steps
        # print_log(counter, loss_value, acc, precision, recall, f1, steps=100)

    end = time.time()
    epoch_time = int((end - start))
    train_loss /= counter

    write_tensorboard(train_loss, tools, tools['tensorboard_train'], e, mode='train')

    print(f"{epoch_time}s - train_loss: {train_loss:>8f} train_acc: {tools['accuracy'].compute().item() * 100 :>0.2f}% "
          f"train_precision: {tools['precision'].compute().item() :>0.4f} "
          f"train_recall: {tools['recall'].compute().item() :>0.4f} "
          f"train_f1: {tools['f1'].compute().item() :>0.4f}", end=' ')


def valid(dataloader, model, tools, e):
    model.eval()
    test_loss, test_correct, counter = 0, 0, 0

    tools['accuracy'].reset()
    tools['precision'].reset()
    tools['recall'].reset()
    tools['f1'].reset()

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

            counter += 1

    test_loss /= counter
    test_correct /= counter

    write_tensorboard(test_loss, tools, tools['tensorboard_valid'], e, mode='valid')

    if tools['best_valid_f1'] < tools['f1'].compute().item():
        torch.save(model.state_dict(), os.path.join(tools['save_path'], "best_valid_f1_checkpoint.pth"))
        tools['best_valid_f1'] = tools['f1'].compute().item()

    print(f"valid_loss: {test_loss:>8f} valid_acc: {tools['accuracy'].compute().item() * 100 :>0.2f}% "
          f"valid_precision: {tools['precision'].compute().item() :>0.4f} "
          f"valid_recall: {tools['recall'].compute().item() :>0.4f} "
          f"valid_f1: {tools['f1'].compute().item() :>0.4f}")


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
        'epochs': 80,
        'batch': batch,
        'window_size': window_size,
        'mixed_precision': True,
        'shuffle': True,
        'label_smoothing': 0.0,
        'warmup': 1000,
        'backbone': 'lstm',
        'pretrained_weights': False,
        'optimizer': 'Adabelief',
        'features': {'pssm': True, 'physiochemical': True, 'aac': True, 'dpc': True},
        'sam': False,
        'lr': 8e-4,
        'weight_decouple': False,
        'weight_decay': 1.2e-6,
        'rectify': False,
        'grad_clip': 5,
        'lr_scheduler': 'CosineAnnealingWarmRestarts',
        'save_path': './results/lstm_final_all_1',
        'model_name': f'{window_size}_{seed}',
        'train_path': '../../data/train data/processed/train_hybrid',
        'valid_path': '../../data/train data/processed/valid_hybrid',
        'test_path': '../../data/test data/processed/test_hybrid',
        'details': 'batchnorm 1d, 2 layer 128 unit, bidirectional'

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

    save_path = os.path.abspath(configs['save_path'])
    save_path = os.path.join(save_path, model_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    with open(os.path.join(save_path, "configs.txt"), 'w') as file:
        for k, v in configs.items():
            file.write(str(k) + ': ' + str(v) + '\n\n')

    train_path = os.path.join(configs['train_path'], f'{configs["window_size"]}.csv')
    valid_path = os.path.join(configs['valid_path'], f'{configs["window_size"]}.csv')
    test_path = os.path.join(configs['test_path'], f'{configs["window_size"]}.csv')

    train_df = pd.read_csv(train_path, low_memory=True)
    valid_df = pd.read_csv(valid_path, low_memory=True)
    test_df = pd.read_csv(test_path, low_memory=True)

    if configs['pretrained_weights']:
        tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
    else:
        tokenizer = load_tokenizer('.')

    training_data = CustomDataset(train_df, tokenizer, configs, configs['pretrained_weights'])
    validation_data = CustomDataset(valid_df, tokenizer, configs, configs['pretrained_weights'])
    test_data = CustomDataset(test_df, tokenizer, configs, configs['pretrained_weights'])
    train_dataloader = DataLoader(training_data, batch_size=configs['batch'], shuffle=configs['shuffle'])
    valid_dataloader = DataLoader(validation_data, batch_size=configs['batch'], shuffle=False)
    test_dataloader = DataLoader(test_data, batch_size=configs['batch'], shuffle=False)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = prepare_model(device, configs, tokenizer)
    epoch_steps = int(np.ceil(len(train_df) / configs['batch']))

    if configs['sam']:
        optimizer = SAM(model.parameters(), AdaBelief, adaptive=True, rho=0.05,
                        lr=configs['lr'],
                        weight_decouple=configs['weight_decouple'], weight_decay=configs['weight_decay'],
                        fixed_decay=False, eps=1e-16, rectify=configs['rectify'], print_change_log=False)
    else:
        if configs['optimizer'].lower() == 'adamw':
            optimizer = torch.optim.AdamW(model.parameters(), lr=configs['lr'], weight_decay=configs['weight_decay'])
        elif configs['optimizer'].lower() == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=configs['lr'])
        elif configs['optimizer'].lower() == 'adabelief':
            optimizer = AdaBelief(model.parameters(), lr=configs['lr'],
                                  weight_decouple=configs['weight_decouple'], weight_decay=configs['weight_decay'],
                                  fixed_decay=False,
                                  eps=1e-16, rectify=configs['rectify'], print_change_log=False)
        else:
            print('wrong given optimizer')
            optimizer = None
            exit()

    lr_scheduler = CosineAnnealingWarmupRestarts(optimizer,
                                                 first_cycle_steps=epoch_steps * configs['epochs'],
                                                 cycle_mult=1.0,
                                                 max_lr=configs['lr'],
                                                 min_lr=1e-7,
                                                 warmup_steps=configs['warmup'],
                                                 gamma=1.0)

    tools = {
        'mixed_precision': configs['mixed_precision'],
        'optimizer': optimizer,
        'grad_clip': configs['grad_clip'],
        'sam': configs['sam'],
        'loss': nn.CrossEntropyLoss(reduction='none', label_smoothing=configs['label_smoothing']),
        'device': device,
        'save_path': save_path,
        'test_name': csv_name,
        'scaler': torch.cuda.amp.GradScaler(enabled=True),
        'lr_scheduler': lr_scheduler,
        'tensorboard_train': SummaryWriter(os.path.join(save_path, 'tensorboard_logs', 'train')),
        'tensorboard_valid': SummaryWriter(os.path.join(save_path, 'tensorboard_logs', 'valid')),
        'tensorboard_test': SummaryWriter(os.path.join(save_path, 'tensorboard_logs', 'test')),
        'f1': F1Score(average='macro', num_classes=2).to(device),
        'precision': Precision(average='macro', num_classes=2).to(device),
        'recall': Recall(average='macro', num_classes=2).to(device),
        'accuracy': Accuracy().to(device),
        'specificity': Specificity(average='macro', num_classes=2).to(device),
        'auc': AUC(reorder=True).to(device),
        'mcc': MatthewsCorrCoef(num_classes=2).to(device),
        'best_valid_f1': 0
    }

    # training configs
    print('configs:')
    for item, config in configs.items():
        print('\t', '-', item, ":", config)

    append_list_as_row(save_path, mode=False, list_of_elem=False)

    print('\nepoch steps:', epoch_steps)
    for e in range(configs['epochs']):
        print(f"Epoch {e + 1}")
        train(train_dataloader, model, tools, e)
        valid(valid_dataloader, model, tools, e)

    print('Evaluating best model on the test set:')
    best_model = prepare_model(device, configs, tokenizer, print_params=False)
    best_model.load_state_dict(torch.load(os.path.join(tools['save_path'], "best_valid_f1_checkpoint.pth")))
    test(test_dataloader, best_model, tools)

    torch.save(model.state_dict(), os.path.join(save_path, "final_checkpoint.pth"))
    print("Saved PyTorch Model State to model.pth")
    print(f'Window size {window_size} done')
    print('\n')

    del train_dataloader, valid_dataloader, test_dataloader
    del train_df, valid_df, test_df,  tools, training_data, validation_data, test_data
    del model, best_model


if __name__ == '__main__':
    test_gpu_cuda()

    window_list = [(5, 512), (7, 512), (9, 512), (15, 512), (21, 512), (27, 512), (33, 512), (45, 512), (55, 512),
                   (77, 512), (99, 512),
                   ]
    for s in [0, 1, 2, 3, 4]:
        for w, b in window_list:
            main(w, b, s)
