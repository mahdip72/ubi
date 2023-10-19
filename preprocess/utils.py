import json
import os
import pandas as pd
import numpy as np
import tqdm
from mat4py import loadmat
from Bio import SeqIO


def calculating_writing_info(save_path, window_df, window_size):
    with open(os.path.join(save_path, "info.txt"), 'a') as file:
        file.write(f'\n*window size: {window_size}')
        file.write('\n\nall samples: ' + str(len(window_df)))
        file.write('\nlabel 1 samples: ' + str(len(window_df[window_df['label'] == 1])))
        file.write('\nlabel 0 samples: ' + str(len(window_df[window_df['label'] == 0])))

        duplicate_window_label_df = window_df.drop_duplicates(subset=['window', 'label'], keep=False, inplace=False)
        file.write('\n\nall samples without duplicate sequences and labels: ' +
                   str(len(duplicate_window_label_df)))
        file.write('\nlabel 1 without duplicates: ' +
                   str(len(duplicate_window_label_df[duplicate_window_label_df['label'] == 1])))
        with_duplicates = len(window_df[window_df['label'] == 1])
        without_duplicates = len(duplicate_window_label_df[duplicate_window_label_df['label'] == 1])
        file.write('\tnumber of duplicates: ' +
                   str(with_duplicates - without_duplicates))
        file.write('\nlabel 0 without duplicates: ' +
                   str(len(duplicate_window_label_df[duplicate_window_label_df['label'] == 0])))
        with_duplicates = len(window_df[window_df['label'] == 0])
        without_duplicates = len(duplicate_window_label_df[duplicate_window_label_df['label'] == 0])
        file.write('\tnumber of duplicates: ' +
                   str(with_duplicates - without_duplicates))

        duplicate_window_df = window_df.drop_duplicates(subset='window', keep=False, inplace=False)
        file.write('\n\nall samples without duplicate sequences: ' +
                   str(len(duplicate_window_df)))
        file.write('\nlabel 1 without duplicates: ' +
                   str(len(duplicate_window_df[duplicate_window_df['label'] == 1])))
        with_duplicates = len(window_df[window_df['label'] == 1])
        without_duplicates = len(duplicate_window_df[duplicate_window_df['label'] == 1])
        file.write('\tnumber of duplicates: ' +
                   str(with_duplicates - without_duplicates))
        file.write('\nlabel 0 without duplicates: ' +
                   str(len(duplicate_window_df[duplicate_window_df['label'] == 0])))
        with_duplicates = len(window_df[window_df['label'] == 0])
        without_duplicates = len(duplicate_window_df[duplicate_window_df['label'] == 0])
        file.write('\tnumber of duplicates: ' +
                   str(with_duplicates - without_duplicates))

        window_df = window_df.sort_values(by='label', ascending=False).drop_duplicates(subset='window', keep='first')
        file.write('\n\npreprocessed all samples: ' + str(len(window_df)))
        file.write('\nlabel 1 samples: ' + str(len(window_df[window_df['label'] == 1])))
        file.write('\nlabel 0 samples: ' + str(len(window_df[window_df['label'] == 0])))

        file.write('\n' + "." * 50 + '\n')


def converting_pssm_from_mat_to_dataframe(file_path):
    """loading, converting and saving pssm features from a .mat file a .csv file"""
    data = loadmat(file_path)
    data = data['NewArgPSSM']
    data = pd.DataFrame({'protein_id': data['pr'], 'feature': data['Feat']})
    data['protein_id'] = data['protein_id'].str.split(pat="|", n=2, expand=True)[1]
    return data


def prepare_pssm_test_features(file_path='S:/Programming/Ubiquitylation/data/test data/pssm features/raw data'):
    folders = os.listdir(file_path)
    id_sequence_pssm_pairs = []
    for folder in folders[:5]:
        class_path = os.path.join(file_path, folder)

        fasta_sequences = SeqIO.parse(open(os.path.join(class_path, os.path.basename(class_path) + ".fasta")), 'fasta')

        pssm_folder_path = os.path.join(class_path, 'pssm')
        path = os.listdir(pssm_folder_path)
        pssm_files_path = os.path.join(pssm_folder_path, path[0])
        all_pssm_files_path = os.listdir(pssm_files_path)
        all_pssm_files_path.sort(key=len)
        for pssm_file_path in tqdm.tqdm(all_pssm_files_path, desc=f'{folder}'):
            pssm_list = []
            with open(os.path.join(pssm_files_path, pssm_file_path), 'r') as file:
                for line in file.readlines()[3:-6]:
                    pssm = line.replace(' ', ',').replace('  ', ',').replace('   ', ',').strip().split(',')
                    pssm = list(filter(lambda x: x != "", pssm))[2:]
                    pssm_list.append(pssm)

            pssm_array = np.array(pssm_list).astype(np.float16)
            pssm_array_str = json.dumps(pssm_array.tolist())
            fasta = next(fasta_sequences)
            prot_id, sequence = fasta.id.split('|')[1], str(fasta.seq)
            assert len(sequence) == pssm_array.shape[0]
            id_sequence_pssm_pairs.append([prot_id, sequence, pssm_array_str])

    df = pd.DataFrame(id_sequence_pssm_pairs)
    df.columns = ['PrName', 'Seq', 'pssm']
    df[['PrName', 'pssm']].to_csv('S:/Programming/Ubiquitylation/data/test data/pssm features/pssm_features.csv', index=False)

    df2 = pd.read_csv('S:/Programming/Ubiquitylation/data/test data/processed/test.csv')
    df2.drop(['Seq'], axis=1, inplace=True)
    df3 = df.merge(df2, how='inner', on='PrName')
    df3[['PrName', 'Seq', 'PositiveSite']].to_csv('S:/Programming/Ubiquitylation/data/test data/processed/test.csv', index=False)


def prepare_physiochemical_dict(df):
    id_physiochemical_dict = {}
    for row in df.itertuples():
        id_physiochemical_dict[row[1]] = list(row[2:])
    return id_physiochemical_dict


if __name__ == '__main__':
    prepare_pssm_test_features()
    pass
