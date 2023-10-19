import pandas as pd
import torch
import numpy as np
import os
import re
from torch.utils.data import Dataset, DataLoader
from utils import load_tokenizer
from torchvision.transforms import ToTensor


class TestDataset(Dataset):
    def __init__(self, csv_file, tok):
        self.df = csv_file[['window']]
        self.tokenizer = tok
        self.transform = ToTensor()

    def __len__(self):
        return len(self.df)

    @staticmethod
    def text_preprocessing(text):
        text = ' '.join(re.findall('.{1,1}', text))
        text = re.sub(r"[UZOB]", "X", text)
        text = re.sub(r"[/^]", "[PAD]", text)
        return text

    def text_to_sequence(self, text):
        text = ' '.join(re.findall('.{1,1}', text))
        sequence = []
        for token in text.split(' '):
            try:
                sequence.append(self.tokenizer[token])
            except KeyError:
                sequence.append(self.tokenizer['<oov>'])
        return sequence

    def __getitem__(self, idx):
        text = self.df.iloc[idx, 0]

        encoded_text = self.text_to_sequence(text)
        encoded_text = torch.tensor(np.squeeze(encoded_text), dtype=torch.int32)

        return encoded_text


if __name__ == '__main__':
    data_path = '../data/processed/windowed_all_labels'
    train_path = os.path.join(data_path, 'train', '9.csv')
    train_df = pd.read_csv(train_path)

    tokenizer = load_tokenizer('.')
    training_data = TestDataset(train_df, tokenizer)
    train_dataloader = DataLoader(training_data, batch_size=4, shuffle=False)
    for i, j in train_dataloader:
        print(i)
        print(j)
