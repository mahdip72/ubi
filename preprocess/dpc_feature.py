import numpy as np
from itertools import product
from collections import Counter


def dpc_compute(seq):
    seq = seq.replace('^', '')
    all_aminos = 'ARNDCQEGHILKMFPSTWYV'
    all_probability = ["".join(a) for a in product(all_aminos, repeat=2)]
    seq_product = ["".join(a) for a in product(seq, repeat=2)]
    seq_count = Counter(seq_product)
    aam_count = Counter(all_probability)
    aam_count = {x: 0 for x in aam_count}

    for i in seq_count:
        seq_count[i] = (seq_count[i] / 400) * 100
        seq_count[i] = float("{:.2f}".format(seq_count[i]))
    aam_count.update(seq_count)
    aa_value = list(aam_count.values())
    return aa_value


def test_dpc():
    if dpc_compute('^^EEEKV') == dpc_compute('EEEKV'):
        print('identical')

    dpc_list = []
    seq = 'EEEKVDG'
    for i in range(10000):
        dpc_list.append(np.array(dpc_compute(seq)).astype(np.float32))
    test = np.array(dpc_list)
    print(f'DPC for {seq} is :', dpc_compute(seq))


if __name__ == '__main__':
    test_dpc()
