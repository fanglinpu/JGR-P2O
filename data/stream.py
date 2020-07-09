# a data wrapper to iterate over large matrices in numpy

import numpy as np

# random, unlimited number of access to the corresponding data
class RandomAccessData(object):
    def __init__(self, data):
        self.data = data
        self.lens = len(self.data)

    def next(self, rng, batchsize):
        ll = rng.choice(self.lens, batchsize)
        return self.data[ll]

class DataStream(object):
    def __init__(self, data):
        self.data = data

    def iterate(self, batchsize, shuffle=True):
        if shuffle:
            indices = np.arange(len(self.data))
            np.random.shuffle(indices)

        for start_idx in range(0, len(self.data)-batchsize+1, batchsize):
            if shuffle:
                excerpt = indices[start_idx:start_idx+batchsize]
            else:
                excerpt = slice(start_idx, start_idx + batchsize)

            yield self.data[excerpt]


class MultiDataStream(object):
    def __init__(self, datas):
        self.datas = list(datas)
        self.lens = [len(data) for data in self.datas]
        self.data_len = min(self.lens)
        self.init_data_len = self.data_len

    def iterate(self, batchsize, seed, shuffle=True):
        if shuffle:
            np.random.seed(seed)
            indices = np.arange(self.init_data_len)
            np.random.shuffle(indices)

        for start_idx in range(0, self.data_len-batchsize+1, batchsize):
            if shuffle:
                excerpt = indices[start_idx:start_idx+batchsize]
            else:
                excerpt = slice(start_idx, start_idx + batchsize)

            yield tuple([data[excerpt] for data in self.datas])

class MultiDataSemiStream(object):
    def __init__(self, datas, semi_datas):
        # semi_datas: no strict correspondence to datas, any time can randomly
        # sample
        self.datas = list(datas)
        self.lens = [len(data) for data in self.datas]
        self.data_len = min(self.lens)
        self.semi_datas = list(semi_datas)
        self.semi_data_len = min([len(d) for d in self.semi_datas])

    def iterate(self, batchsize, shuffle=True):
        if shuffle:
            indices = np.arange(self.data_len)
            np.random.shuffle(indices)

        semi_indices = np.arange(self.semi_data_len)
        np.random.shuffle(semi_indices) 

        for start_idx in range(0, self.data_len-batchsize+1, batchsize):
            if shuffle:
                excerpt = indices[start_idx:start_idx+batchsize]
            else:
                excerpt = slice(start_idx, start_idx + batchsize)

            semi_excerpt = semi_indices[start_idx:start_idx+batchsize]

            yield tuple([data[excerpt] for data in self.datas] +
                        [data[semi_excerpt] for data in self.semi_datas])
