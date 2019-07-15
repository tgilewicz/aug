import random

from aug import Sequential


class Shuffle(Sequential):
    """Pipeline which contains other pipelines or ops, and applies
        them in random order to produce output.
    """

    def __init__(self, *args):
        super(Shuffle, self).__init__()
        self._ops = list(args)
        random.shuffle(self._ops)

    def apply(self, sample):
        ops_list = list(self._ops)
        random.shuffle(ops_list)

        for t in ops_list:
            sample = t.apply(sample)

        return sample
