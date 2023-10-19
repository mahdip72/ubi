import os
import pandas as pd
from sklearn.model_selection import train_test_split


def main():
    dataset_path = os.path.abspath('./../data/FinalTotalData.xlsx')

    df = pd.read_excel(dataset_path)
    train_df, valid_df = train_test_split(df, test_size=0.1, random_state=1)

    save_path = os.path.abspath('./../data/processed')
    train_df.to_csv(os.path.join(save_path, 'train.csv'), index=False)
    valid_df.to_csv(os.path.join(save_path, 'valid.csv'), index=False)
    print('done')


if __name__ == '__main__':
    main()
