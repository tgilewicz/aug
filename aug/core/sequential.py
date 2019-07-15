from aug import Pipeline


class Sequential(Pipeline):
    """Pipeline which contains other pipelines or ops, and applies
        them in sequence to produce output.
    """

    def __init__(self, *args):
        super(Sequential, self).__init__()
        self._ops = args

    def apply(self, sample):
        for t in self._ops:
            sample = t.apply(sample)

        return sample

    def time(self, img):
        """Measure time needed to apply operations in pipeline. """
        op_times = {}
        for t in self._ops:
            op_time = t.time(img)
            op_times = {
                k: op_time.get(k, 0) + op_times.get(k, 0) for k in set(op_time) | set(op_times)
            }
        return op_times
