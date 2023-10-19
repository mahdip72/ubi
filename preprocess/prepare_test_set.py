import os
import pandas as pd


def prepare_id_seq():
    data_path = './../data/test data/2368 new ubi proteins with 40% CDHIT.txt'
    with open(data_path) as file:
        seqs = file.readlines()

    seqs = ''.join(seqs)
    proteins = seqs.split('>sp')
    id_seq_dict = {}
    for protein in proteins[1:]:
        prot_id = protein.split('|')[1]
        prot_seq = ''.join(protein.split('\n')[1:])
        id_seq_dict[prot_id] = prot_seq

    return id_seq_dict


def main():
    dataset_path = os.path.abspath('./../data/test data/cod Id and positive sites_2022 dbPTM database.xlsx')

    df_sites = pd.read_excel(dataset_path, names=['id', 'position'], sheet_name='total positive sites', header=None)

    id_position_dict = {}
    for row in df_sites.itertuples():
        if row[1] not in id_position_dict.keys():
            id_position_dict[row[1]] = []
        id_position_dict[row[1]].append(row[2])

    id_seq_dict = prepare_id_seq()
    id_seq_position_list = []
    for prot_id, seq in id_seq_dict.items():
        try:
            id_seq_position_list.append([prot_id, seq, '   '.join([str(i) for i in id_position_dict[prot_id]])])
        except KeyError:
            print(prot_id)

    df = pd.DataFrame(id_seq_position_list, columns=['PrName', 'Seq', 'PositiveSite'])
    save_path = os.path.abspath('./../data/test data/processed')
    df.to_csv(os.path.join(save_path, 'test.csv'), index=False)
    print('done')


if __name__ == '__main__':
    main()
