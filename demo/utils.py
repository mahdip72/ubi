import os
import pickle
import torch
from csv import writer


def append_list_as_row(save_path, mode, list_of_elem):
    if os.path.exists(os.path.join(save_path, 'log.csv')):
        with open(os.path.join(save_path, 'log.csv'), 'a+', newline='') as file:
            # Create a writer object from csv module
            if mode == 'train':
                file.write(', '.join([str(i) for i in list_of_elem]) + ', ')
            elif mode == 'valid':
                file.write(', '.join([str(i) for i in list_of_elem[1:]]) + '\n')
    else:
        with open(os.path.join(save_path, 'log.csv'), 'a+', newline='') as file:
            csv_writer = writer(file)
            titles = ['epoch', 'loss', 'acc', 'precision', 'recall', 'f1',
                      'val_loss', 'val_acc', 'val_precision', 'val_recall', 'val_f1']
            csv_writer.writerow(titles)


def save_tokenizer(tokenizer, path):
    # saving
    with open(os.path.join(path, 'tokenizer.pkl'), 'wb') as file:
        pickle.dump(tokenizer, file, protocol=pickle.HIGHEST_PROTOCOL)


def load_tokenizer(path):
    # loading
    with open(os.path.join(path, 'tokenizer.pkl'), 'rb') as handle:
        tokenizer = pickle.load(handle)
    return tokenizer


def test_gpu_cuda():
    print('Testing gpu and cuda:')
    print('\tcuda is available:', torch.cuda.is_available())
    print('\tdevice count:', torch.cuda.device_count())
    print('\tcurrent device:', torch.cuda.current_device())
    print(f'\tdevice:', torch.cuda.device(0))
    print('\tdevice name:', torch.cuda.get_device_name(), end='\n\n')


def print_log(c, loss_value, acc, precision, recall, f1, steps):
    if c % steps == 0:
        print(f"step {c} - loss: {loss_value:>8f} acc: {acc :>0.2f}% "
              f"precision: {precision:>4f} recall: {recall:>4f} f1: {f1:>4f}\t")


def write_test_results(compute_loss, tools):
    acc = round(tools['accuracy'].compute().item() * 100, 2)
    precision = round(tools['precision'].compute().item(), 4)
    recall = round(tools['recall'].compute().item(), 4)
    f1 = round(tools['f1'].compute().item(), 4)
    specificity = round(tools['specificity'].compute().item(), 4)
    mcc = round(tools['mcc'].compute().item(), 4)

    list_of_elem = [compute_loss, acc, precision, recall, f1, specificity, auc, mcc]

    with open(os.path.join(tools['save_path'], tools['test_name']+'.csv'), 'a+', newline='') as file:
        csv_writer = writer(file)
        titles = ['test_loss', 'test_acc', 'test_precision', 'test_recall', 'test_f1',
                  'test_specificity', 'test_mcc']
        csv_writer.writerow(titles)
        file.write(', '.join([str(i) for i in list_of_elem]))
