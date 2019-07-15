import numpy as np

from aug import Pipeline


class Choice(Pipeline):
    """Pipeline which contains other pipelines or ops, and applies
        one of them (randomly) to produce output.
    """

    def __init__(self, *args, weights=None):
        super(Choice, self).__init__()
        self._transformations = args
        assert len(args) > 0, "Choice requires 1 or more input transformations"

        self._weights = weights

        if weights is not None:
            assert len(self._weights) == len(self._transformations)

        self.op = self.choose_op()

    def choose_op(self):
        if self._weights is not None:
            op = np.random.choice(list(self._transformations), 1, p=self._weights)[0]
        else:
            op = np.random.choice(list(self._transformations), 1)[0]
        return op

    def apply(self, sample):
        return self.op.apply(sample)

    def time(self, img):
        """Measure time needed to apply operations in pipeline. """
        op = self.choose_op()
        return op.time(img)
