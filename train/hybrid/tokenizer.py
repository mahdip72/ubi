import os
import re
import pandas as pd
import tensorflow as tf
import pickle
from utils import save_tokenizer


def create_tokenizer():
    data_path = '../data/processed/windowed_all_labels'
    train_path = os.path.join(data_path, 'train', '15.csv')

    train_df = pd.read_csv(train_path)
    tokenizer = tf.keras.preprocessing.text.Tokenizer(char_level=True, oov_token='<oov>', lower=False)
    tokenizer.fit_on_texts(train_df['window'])
    print(tokenizer.word_index)
    save_tokenizer(tokenizer.word_index, '.')


if __name__ == '__main__':
    create_tokenizer()
