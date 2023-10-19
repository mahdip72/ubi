import itertools


def be_compute(seq):
    encodings = []
    for index, a in enumerate(seq):
        code = [0] * 20
        code[index] = 1
        encodings.append(code)
    encodings = list(itertools.chain.from_iterable(encodings))
    return encodings


def main():
    seq = 'ALS'
    print(be_compute(seq))


if __name__ == "__main__":
    main()
