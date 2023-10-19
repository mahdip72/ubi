from collections import Counter


def aac_compute(seq):
    seq = seq.replace('^', '')
    aa_dict = {'A': 0, 'C': 0, 'D': 0, 'E': 0, 'F': 0, 'G': 0, 'H': 0, 'I': 0, 'K': 0, 'L': 0, 'M': 0, 'N': 0, 'P': 0,
               'Q': 0, 'R': 0, 'S': 0, 'T': 0,
               'V': 0, 'W': 0, 'Y': 0}
    count_each_aa = Counter(seq)
    for i in count_each_aa:
        count_each_aa[i] = (count_each_aa[i] / len(seq)) * 100
        count_each_aa[i] = float("{:.2f}".format(count_each_aa[i]))
    aa_dict.update(count_each_aa)
    return list(aa_dict.values())


def test_aac():
    seq = '^^^ACDEPQRSTVWY'
    for i in range(10000):
        aac_compute(seq)
    print(f'AAC for {seq} ', aac_compute(seq))


if __name__ == '__main__':
    test_aac()
