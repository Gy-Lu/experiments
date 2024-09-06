import itertools
import torch

class DummyDS(torch.utils.data.IterableDataset):
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def __iter__(self):
        for i in itertools.count():
            print("outer")
            count = 0
            while count < self.batch_size:
                print("inner")
                count += 1
                yield i
 
bs = 2
ds = DummyDS(batch_size=bs)
dl = torch.utils.data.DataLoader(ds, batch_size=bs)
for i, batch in enumerate(dl):
    print(i, batch)
    if i > 3:
        break