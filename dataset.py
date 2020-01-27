from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import torch
from operator import attrgetter
from data_helper import TagData
from random import randint

class SeqSeqDataSet(Dataset):
    def __init__(self, data) -> None:
        super().__init__()
        self.data = data

    def __getitem__(self, index):
        return list(map(torch.tensor, self.data[index]))

    def __len__(self):
        return len(self.data)


def collate_sequences(batch):
    return [item[0] for item in batch], [item[1] for item in batch]



def loader(data, section, batch_size, workers=0):
    section = attrgetter(section)(data)
    seqseq = [([data.word2idx[word[0]] if word[0] in data.word2idx else randint(0, len(data.words)) for word in line], [data.tag2idx[word[1]] for word in line]) for line in section]
    return DataLoader(SeqSeqDataSet(seqseq), batch_size=batch_size, shuffle=True, num_workers=workers, collate_fn=collate_sequences)


def test():
    train = loader(TagData('pos'), 'train', 3)
    print(next(iter(train)))


if __name__ == "__main__":
    test()