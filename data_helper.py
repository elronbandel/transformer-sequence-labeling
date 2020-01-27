from collections import Counter
import numpy as np
from operator import itemgetter
import pandas as pd
import re


class PreTrainedEmbeddings:
    def __init__(self):
        self.words = pd.read_fwf("embedding/vocab.txt", header=None)[0].values
        self.vecs = np.loadtxt("embedding/wordVectors.txt").astype(np.float32)
        self.word2ix = {word: i for i, word in enumerate(self.words)}


def get_test_data(test_file_path, lowercase=False):
    return TagData.load_data(test_file_path, tagged=False, lowercase=lowercase)


def get_dev_data(dev_file_path, lowercase=False):
    return TagData.load_data(dev_file_path, tagged=True, lowercase=lowercase)


class TagData:
    def __init__(self, dir, threshold=None, lowercase=False):
        self.train, self.dev, self.test = TagData.load(dir, lowercase=lowercase)
        self.vocab = TagData.build_vocab(self.train, threshold=threshold)
        self.words = list(self.vocab)
        self.word2idx = {word: i for i, word in enumerate(self.words)}
        self.tags = TagData.build_tags_list(self.train)
        self.tag2idx = {tag: i + 1 for i, tag in enumerate(self.tags)}
        self.tag2idx['NONE'] = 0

    @staticmethod
    def load(dir, lowercase=False):
        train = TagData.load_data('{}/train'.format(dir), tagged=True, lowercase=lowercase)
        dev = TagData.load_data('{}/dev'.format(dir), tagged=True, lowercase=lowercase)
        test = TagData.load_data('{}/test'.format(dir), tagged=False, lowercase=lowercase)
        return train, dev, test

    @staticmethod
    def load_data(path, tagged=True, lowercase=False):
        data = [[]]
        load_func = (lambda line: (re.split(' |\t', line, 1)[0] if not lowercase else re.split(' |\t', line, 1)[0].lower(), re.split(' |\t', line, 1)[1])) \
            if tagged else str.strip
        for line in open(path):
            line = line.strip()
            if not line:
                data.append(list())
            else:
                data[-1].append(load_func(line))

        return data[:-1]

    @staticmethod
    def count_words(data):
        counter = Counter()
        for line in data:
            for pair in line:
                counter[pair[0]] += 1
        return counter

    @staticmethod
    def build_tags_list(data):
        tags = set()
        for line in data:
            for pair in line:
                tags.add(pair[1])
        return list(tags)

    @staticmethod
    def build_vocab(data, threshold=None):
        counter = TagData.count_words(data)
        vocab = None
        #no threshold - no unkown handeling
        if not threshold:
            vocab = set(counter.keys())
        # threshold as precentage of most common words:
        elif 0 < threshold <= 1:
            n_words = int(len(counter) * threshold)
            vocab = set(map(itemgetter(0), counter.most_common(n_words)))
        elif threshold > 1:
            vocab = set()
            for word, count in counter.items():
                if count > threshold:
                    vocab.add(word)
        return vocab


if __name__ == '__main__':
    # test and use example
    data = TagData('pos')
    w10 = data.words[10]
    idx10 = data.word2idx[w10]
    if w10 in data.vocab:
        print(w10)
        print(idx10)
    print(data.dev)
