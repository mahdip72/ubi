import os
from utils import converting_pssm_from_mat_to_dataframe


def main():
    pssm_path = '../data/train data/pssm features/NewArrgPSSM_26_11_1400.mat'
    df = converting_pssm_from_mat_to_dataframe(pssm_path)
    df.to_csv(os.path.join('../data/train data/pssm features', 'pssm_features.csv'), index=False)
    print('done')


if __name__ == '__main__':
    main()
