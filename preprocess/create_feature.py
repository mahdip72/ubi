import pandas as pd
import os
import json
from utils import prepare_physiochemical_dict


def pssm_compute(pssm_path, sequence_path, save_path):
    df = pd.read_csv(sequence_path, low_memory=True)
    pssm_df = pd.read_csv(pssm_path, low_memory=True)

    pssm_dict = {}
    for row in pssm_df.itertuples():
        pssm_dict[row[1]] = json.loads(row[2])
    del pssm_df

    pssm_features = []
    for index, pr_id, sequence in df[['PrName', 'Seq']].itertuples():
        pssm_features.append(pssm_dict[pr_id])

    df['pssm'] = pssm_features
    df.to_csv(save_path, index=False)


def physiochemical_compute(physiochemical_path, sequence_path, save_path):
    physiochemical_property_df = pd.read_csv(physiochemical_path, low_memory=True)
    df = pd.read_csv(sequence_path, low_memory=True)

    id_physiochemical_dict = prepare_physiochemical_dict(physiochemical_property_df)

    physiochemical_features = []
    for index, pr_id, sequence in df[['PrName', 'Seq']].itertuples():
        sample = []
        for amino in sequence:
            sample.append(id_physiochemical_dict[amino])
        physiochemical_features.append(sample)

    df['physiochemical'] = physiochemical_features

    df.to_csv(save_path, index=False)


def pssm_main():
    pssm_path = '../data/train data/pssm features/pssm_features.csv'

    sequence_path = 'S:/Programming/Ubiquitylation/data/train data/processed/train.csv'
    save_path = os.path.join(os.path.abspath('./../data/train data/processed'), 'train_hybrid.csv')
    pssm_compute(pssm_path, sequence_path, save_path)
    print('pssm train done')

    sequence_path = 'S:/Programming/Ubiquitylation/data/train data/processed/valid.csv'
    save_path = os.path.join(os.path.abspath('./../data/train data/processed'), 'valid_hybrid.csv')
    pssm_compute(pssm_path, sequence_path, save_path)
    print('pssm valid done')

    pssm_path = 'S:/Programming/Ubiquitylation/data/test data/pssm features/pssm_features.csv'
    sequence_path = 'S:/Programming/Ubiquitylation/data/test data/processed/test.csv'
    save_path = os.path.join(os.path.abspath('./../data/test data/processed'), 'test_hybrid.csv')
    pssm_compute(pssm_path, sequence_path, save_path)
    print('pssm test done')


def physiochemical_main():
    physiochemical_path = '../data/sixteen kinds of the physiochemical property.csv'

    sequence_path = 'S:/Programming/Ubiquitylation/data/train data/processed/train_hybrid.csv'
    save_path = os.path.join(os.path.abspath('./../data/train data/processed'), 'train_hybrid.csv')
    physiochemical_compute(physiochemical_path, sequence_path, save_path)
    print('physiochemical train done')

    sequence_path = 'S:/Programming/Ubiquitylation/data/train data/processed/valid_hybrid.csv'
    save_path = os.path.join(os.path.abspath('./../data/train data/processed'), 'valid_hybrid.csv')
    physiochemical_compute(physiochemical_path, sequence_path, save_path)
    print('physiochemical valid done')

    sequence_path = 'S:/Programming/Ubiquitylation/data/test data/processed/test_hybrid.csv'
    save_path = os.path.join(os.path.abspath('./../data/test data/processed'), 'test_hybrid.csv')
    physiochemical_compute(physiochemical_path, sequence_path, save_path)
    print('physiochemical test done')


if __name__ == '__main__':
    pssm_main()
    physiochemical_main()
