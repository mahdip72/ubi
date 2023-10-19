import pandas as pd
import numpy as np
import os
import glob
import re
import tqdm as tqdm
import json
import ast
from utils import calculating_writing_info


np.set_printoptions(suppress=True)


def padding_positive_positions(positions, padding=50):
    masked_intervals = []
    for position in positions:
        minimum = position - padding
        if minimum < 1:
            minimum = 1
        maximum = position + padding
        masked_intervals.append([minimum, maximum])
    return masked_intervals


def computing_all_labels():
    csv_path = '../data/processed/'
    output_path = '../data/processed/windowed_all_labels'
    window_min = 5
    window_max = 99

    csv_paths = glob.glob((os.path.join(csv_path, '**.csv')))
    for csv in csv_paths:
        if os.path.basename(csv) in ['test.csv', 'test-hybrid.csv']:
            continue
        save_path = os.path.join(output_path, os.path.basename(csv).split('.csv')[0])
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        df = pd.read_csv(csv)
        window_sizes = list(range(window_min, window_max + 1, 2))
        for window_size in tqdm.tqdm(window_sizes, total=len(window_sizes)):
            dataset = []
            for row in df.itertuples():
                positions = [int(p) for p in re.findall(r'\b\d+\b', row[3])]
                positions = sorted(list(set(positions)))
                sequence = row[2]
                for index, amino in enumerate(sequence):
                    if amino in ['K']:
                        left_pad = (window_size - 1) / 2 - index
                        right_pad = (window_size - 1) / 2 - len(sequence) + (index + 1)
                        if left_pad > 0:
                            left = int(left_pad) * '^' + sequence[:index]
                        else:
                            left = sequence[int(index - (window_size - 1) / 2):index]
                        if right_pad > 0:
                            right = sequence[index + 1:] + int(right_pad) * '^'
                        else:
                            right = sequence[index + 1: int(index + 1 + (window_size - 1) / 2)]
                        sample = left + amino + right
                        # sample = sample.replace("B", "X").replace("U", "X")

                        if (index + 1) in positions:
                            label = 1
                        else:
                            label = 0
                        dataset.append((sample, label))

            window_df = pd.DataFrame(dataset, columns=['window', 'label'])
            window_df.sort_values(by='label', ascending=False).drop_duplicates(
                subset='window', keep='first').to_csv(os.path.join(save_path, f'{window_size}.csv'), index=False)

            calculating_writing_info(save_path, window_df, window_size)
            del window_df, dataset
    print('computing_all_labels done')


def computing_processed_labels():
    csv_path = '../data/processed/'
    output_path = '../data/processed/windowed_processed_labels'
    window_min = 5
    window_max = 99

    csv_paths = glob.glob((os.path.join(csv_path, '**.csv')))
    for csv in csv_paths:
        if os.path.basename(csv) in ['test.csv', 'test-hybrid.csv']:
            continue
        save_path = os.path.join(output_path, os.path.basename(csv).split('.csv')[0])
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        df = pd.read_csv(csv)
        window_sizes = list(range(window_min, window_max + 1, 2))
        for window_size in tqdm.tqdm(window_sizes, total=len(window_sizes)):
            dataset = []
            for row in df.itertuples():
                positions = [int(p) for p in re.findall(r'\b\d+\b', row[3])]
                positions = sorted(list(set(positions)))

                # creating the intervals in which negative samples cannot be created
                masked_intervals = padding_positive_positions(positions, padding=50)
                sequence = row[2]
                for index, amino in enumerate(sequence):
                    if amino in ['K']:
                        ignore = False
                        left_pad = (window_size - 1) / 2 - index
                        right_pad = (window_size - 1) / 2 - len(sequence) + (index + 1)
                        if left_pad > 0:
                            left = int(left_pad) * '^' + sequence[:index]
                        else:
                            left = sequence[int(index - (window_size - 1) / 2):index]
                        if right_pad > 0:
                            right = sequence[index + 1:] + int(right_pad) * '^'
                        else:
                            right = sequence[index + 1: int(index + 1 + (window_size - 1) / 2)]
                        sample = left + amino + right
                        # sample = sample.replace("B", "X").replace("U", "X")

                        if (index + 1) in positions:
                            label = 1
                        else:
                            for interval in masked_intervals:
                                minimum = interval[0]
                                maximum = interval[1]
                                if minimum <= index + 1 <= maximum:
                                    ignore = True
                                    continue
                            if ignore:
                                continue
                            label = 0
                        dataset.append((sample, label))

            window_df = pd.DataFrame(dataset, columns=['window', 'label'])
            window_df.sort_values(by='label', ascending=False).drop_duplicates(
                subset='window', keep='first').to_csv(os.path.join(save_path, f'{window_size}.csv'), index=False)
            calculating_writing_info(save_path, window_df, window_size)
            del window_df, dataset
    print('computing_processed_labels done')


def computing_all_labels_with_features():
    csv_path = '../data/processed/'
    pssm_path = '../data/PSSM feature/pssm_features.csv'
    output_path = '../data/processed/windowed_all_labels_with_pssm'
    window_min = 5
    window_max = 23

    pssm_df = pd.read_csv(pssm_path)
    pssm_dict = {}
    for row in pssm_df.itertuples():
        pssm_dict[row[1]] = np.array(json.loads(row[2])).astype(np.float16)
    del pssm_df

    csv_paths = glob.glob((os.path.join(csv_path, '**.csv')))
    for csv in csv_paths:
        if os.path.basename(csv) in ['test.csv', 'test-hybrid.csv']:
            continue
        save_path = os.path.join(output_path, os.path.basename(csv).split('.csv')[0])
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        df = pd.read_csv(csv)
        window_sizes = list(range(window_min, window_max + 1, 2))
        for window_size in tqdm.tqdm(window_sizes, total=len(window_sizes)):
            dataset = []
            for row in df.itertuples():
                protein_id = row[1]
                positions = [int(p) for p in re.findall(r'\b\d+\b', row[3])]
                positions = sorted(list(set(positions)))
                sequence = row[2]
                pssm = pssm_dict[protein_id]

                for index, amino in enumerate(sequence):
                    if amino in ['K']:
                        left_pad = (window_size - 1) / 2 - index
                        right_pad = (window_size - 1) / 2 - len(sequence) + (index + 1)
                        if left_pad > 0:
                            left = int(left_pad) * '^' + sequence[:index]
                            left_pssm = np.concatenate([np.zeros((int(left_pad), 42)), pssm[:index]])
                        else:
                            left = sequence[int(index - (window_size - 1) / 2):index]
                            left_pssm = pssm[int(index - (window_size - 1) / 2):index]
                        if right_pad > 0:
                            right = sequence[index + 1:] + int(right_pad) * '^'
                            right_pssm = np.concatenate([pssm[index + 1:],
                                                         np.zeros((int(right_pad), 42))])
                        else:
                            right = sequence[index + 1: int(index + 1 + (window_size - 1) / 2)]
                            right_pssm = pssm[index + 1: int(index + 1 + (window_size - 1) / 2)]

                        sample = left + amino + right
                        pssm_sample = np.concatenate([left_pssm, pssm[index].reshape((1, 42)), right_pssm])
                        # pssm_sample = pssm[int(index - (window_size - 1) / 2): int(index + 1 + (window_size - 1) / 2)]

                        if (index + 1) in positions:
                            label = 1
                        else:
                            label = 0
                        dataset.append((sample,
                                        json.dumps(pssm_sample.tolist()),
                                        label))

            window_df = pd.DataFrame(dataset, columns=['window', 'pssm', 'label'])
            window_df.sort_values(by='label', ascending=False).drop_duplicates(
                subset='window', keep='first').to_csv(os.path.join(save_path, f'{window_size}.csv'), index=False)
            calculating_writing_info(save_path, window_df, window_size)
            del window_df, dataset
    print('computing_all_labels_with_pssm done')


def computing_processed_labels_with_pssm():
    csv_path = '../data/processed/'
    pssm_path = '../data/PSSM feature/pssm_features.csv'
    output_path = '../data/processed/windowed_processed_labels_with_pssm'
    window_min = 5
    window_max = 23

    pssm_df = pd.read_csv(pssm_path)
    pssm_dict = {}
    for row in pssm_df.itertuples():
        pssm_dict[row[1]] = np.array(json.loads(row[2])).astype(np.float16)
    del pssm_df

    csv_paths = glob.glob((os.path.join(csv_path, '**.csv')))
    for csv in csv_paths:
        if os.path.basename(csv) in ['test.csv', 'test-hybrid.csv']:
            continue
        save_path = os.path.join(output_path, os.path.basename(csv).split('.csv')[0])
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        df = pd.read_csv(csv)
        window_sizes = list(range(window_min, window_max + 1, 2))
        for window_size in tqdm.tqdm(window_sizes, total=len(window_sizes)):
            dataset = []
            for row in df.itertuples():
                protein_id = row[1]
                positions = [int(p) for p in re.findall(r'\b\d+\b', row[3])]
                positions = sorted(list(set(positions)))
                pssm = pssm_dict[protein_id]

                # creating the intervals in which negative samples cannot be created
                masked_intervals = padding_positive_positions(positions, padding=50)

                sequence = row[2]
                for index, amino in enumerate(sequence):
                    if amino in ['K']:
                        ignore = False
                        left_pad = (window_size - 1) / 2 - index
                        right_pad = (window_size - 1) / 2 - len(sequence) + (index + 1)
                        if left_pad > 0:
                            left = int(left_pad) * '^' + sequence[:index]
                            left_pssm = np.concatenate([np.zeros((int(left_pad), 42)), pssm[:index]])
                        else:
                            left = sequence[int(index - (window_size - 1) / 2):index]
                            left_pssm = pssm[int(index - (window_size - 1) / 2):index]
                        if right_pad > 0:
                            right = sequence[index + 1:] + int(right_pad) * '^'
                            right_pssm = np.concatenate([pssm[index + 1:],
                                                         np.zeros((int(right_pad), 42))])
                        else:
                            right = sequence[index + 1: int(index + 1 + (window_size - 1) / 2)]
                            right_pssm = pssm[index + 1: int(index + 1 + (window_size - 1) / 2)]

                        sample = left + amino + right
                        pssm_sample = np.concatenate([left_pssm, pssm[index].reshape((1, 42)), right_pssm])
                        # pssm_sample = pssm[int(index - (window_size - 1) / 2): int(index + 1 + (window_size - 1) / 2)]

                        if (index + 1) in positions:
                            label = 1
                        else:
                            for interval in masked_intervals:
                                minimum = interval[0]
                                maximum = interval[1]
                                if minimum <= index + 1 <= maximum:
                                    ignore = True
                                    continue
                            if ignore:
                                continue
                            label = 0
                        dataset.append((sample,
                                        json.dumps(pssm_sample.tolist()),
                                        label))

            window_df = pd.DataFrame(dataset, columns=['window', 'pssm', 'label'])
            window_df.sort_values(by='label', ascending=False).drop_duplicates(
                subset='window', keep='first').to_csv(os.path.join(save_path, f'{window_size}.csv'), index=False)
            calculating_writing_info(save_path, window_df, window_size)
            del window_df, dataset
    print('computing_processed_labels_with_pssm done')


def computing_test_labels():
    csv_path = '../data/processed/test.csv'
    output_path = '../data/processed/'
    window_min = 5
    window_max = 99

    save_path = os.path.join(output_path, os.path.basename(csv_path).split('.csv')[0])
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    df = pd.read_csv(csv_path)
    window_sizes = list(range(window_min, window_max + 1, 2))
    for window_size in tqdm.tqdm(window_sizes, total=len(window_sizes)):
        dataset = []
        for row in df.itertuples():
            positions = [int(p) for p in re.findall(r'\b\d+\b', row[3])]
            positions = sorted(list(set(positions)))
            sequence = row[2]
            for index, amino in enumerate(sequence):
                if amino in ['K']:
                    left_pad = (window_size - 1) / 2 - index
                    right_pad = (window_size - 1) / 2 - len(sequence) + (index + 1)
                    if left_pad > 0:
                        left = int(left_pad) * '^' + sequence[:index]
                    else:
                        left = sequence[int(index - (window_size - 1) / 2):index]
                    if right_pad > 0:
                        right = sequence[index + 1:] + int(right_pad) * '^'
                    else:
                        right = sequence[index + 1: int(index + 1 + (window_size - 1) / 2)]
                    sample = left + amino + right
                    # sample = sample.replace("B", "X").replace("U", "X")

                    if (index + 1) in positions:
                        label = 1
                    else:
                        label = 0
                    dataset.append((sample, label))

        window_df = pd.DataFrame(dataset, columns=['window', 'label'])
        window_df.sort_values(by='label', ascending=False).to_csv(os.path.join(save_path, f'{window_size}.csv'),
                                                                  index=False)

        calculating_writing_info(save_path, window_df, window_size)
        del window_df, dataset
    print('computing_test_labels done')


def computing_with_features(csv_path, output_path):
    window_min = 5
    window_max = 99

    save_path = os.path.join(output_path, os.path.basename(csv_path).split('.csv')[0])
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    df = pd.read_csv(csv_path)

    window_sizes = [5, 7, 9, 15, 21, 27, 33, 45, 55, 77, 99]
    for window_size in tqdm.tqdm(window_sizes, total=len(window_sizes)):
        dataset = []
        for row in df[['PrName', 'Seq', 'PositiveSite', 'pssm', 'physiochemical']].itertuples():
            protein_id = row[1]
            sequence = row[2]
            positions = [int(p) for p in re.findall(r'\b\d+\b', row[3])]
            positions = sorted(list(set(positions)))
            pssm = np.array(ast.literal_eval(row[4])).astype(np.float16)
            physiochemical = np.array(ast.literal_eval(row[5])).astype(np.float16)
            for index, amino in enumerate(sequence):
                if amino in ['K']:
                    left_pad = (window_size - 1) / 2 - index
                    right_pad = (window_size - 1) / 2 - len(sequence) + (index + 1)
                    if left_pad > 0:
                        left = int(left_pad) * '^' + sequence[:index]
                        left_pssm = np.concatenate([np.zeros((int(left_pad), 42)), pssm[:index]])
                        left_physiochemical = np.concatenate([np.zeros((int(left_pad), 16)), physiochemical[:index]])
                    else:
                        left = sequence[int(index - (window_size - 1) / 2):index]
                        left_pssm = pssm[int(index - (window_size - 1) / 2):index]
                        left_physiochemical = physiochemical[int(index - (window_size - 1) / 2):index]
                    if right_pad > 0:
                        right = sequence[index + 1:] + int(right_pad) * '^'
                        right_pssm = np.concatenate([pssm[index + 1:], np.zeros((int(right_pad), 42))])
                        right_physiochemical = np.concatenate([physiochemical[index + 1:], np.zeros((int(right_pad), 16))])
                    else:
                        right = sequence[index + 1: int(index + 1 + (window_size - 1) / 2)]
                        right_pssm = pssm[index + 1: int(index + 1 + (window_size - 1) / 2)]
                        right_physiochemical = physiochemical[index + 1: int(index + 1 + (window_size - 1) / 2)]

                    sample = left + amino + right
                    pssm_sample = np.concatenate([left_pssm, pssm[index].reshape((1, 42)), right_pssm])
                    physiochemical_sample = np.concatenate([left_physiochemical, physiochemical[index].reshape((1, 16)), right_physiochemical])

                    if (index + 1) in positions:
                        label = 1
                    else:
                        label = 0
                    dataset.append((sample,
                                    json.dumps(pssm_sample.tolist()),
                                    json.dumps(physiochemical_sample.tolist()),
                                    label))

        window_df = pd.DataFrame(dataset, columns=['window', 'pssm', 'physiochemical', 'label'])
        # window_df.sort_values(by='label', ascending=False).drop_duplicates(
        #     subset='window', keep='first').to_csv(os.path.join(save_path, f'{window_size}.csv'), index=False)
        window_df.to_csv(os.path.join(save_path, f'{window_size}.csv'), index=False)
        calculating_writing_info(save_path, window_df, window_size)
        del window_df, dataset
    print('computing features done')


def main():
    # computing_all_labels()
    # computing_processed_labels()
    # computing_all_labels_with_features()
    # computing_processed_labels_with_pssm()
    # computing_test_labels()

    csv_path = '../data/train data/processed/train_hybrid.csv'
    output_path = '../data/train data/processed/'
    computing_with_features(csv_path, output_path)

    csv_path = '../data/train data/processed/valid_hybrid.csv'
    output_path = '../data/train data/processed/'
    computing_with_features(csv_path, output_path)

    csv_path = '../data/test data/processed/test_hybrid.csv'
    output_path = '../data/test data/processed/'
    computing_with_features(csv_path, output_path)


if __name__ == '__main__':
    main()
