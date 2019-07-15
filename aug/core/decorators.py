from aug import Sample
import random


class BaseWrapperMeta(type):

    def __getattr__(self, item):
        """Implements the possibility to access static variables of a decorated class. """
        try:
            return self.__getattribute__(self, item)

        except AttributeError:
            return getattr(self.cls, item)


class BaseWrapper(object, metaclass=BaseWrapperMeta):
    """Common tools for other class wrappers. """

    @staticmethod
    def validate_input_args(args):
        for arg in args:
            classes = [int, float, str, list, tuple, bool, type(None), dict]
            assert any([isinstance(arg, c) for c in classes]), "Invalid type of input parameter"

    def __getattr__(self, item):
        """Implements the possibility to access attributes of a wrapped object. """
        try:
            return self.__getattribute__(item)

        except AttributeError:
            return getattr(self._wrapped, item)


def perform_randomly(cls):
    """Wrapper for Operation class. Implements the possibility of determining the
        probability of performing an operation.
    """

    class Wrapper(BaseWrapper):
        cls = 1

        def __init__(self, *args, p=1., **kwargs):
            """

            Args:
                *args: A list of input parameters forwarded to the constructor of
                    operation
                p: A probability of operation performing
            """
            assert 0 < p <= 1
            arg_values = list(args) + list(kwargs.values())
            self.validate_input_args(arg_values)

            self._wrapped = cls(*args, **kwargs)
            self._probability = p
            self._allowed_to_perform = random.random() < self._probability
            Wrapper.cls = cls

        def apply(self, sample):
            assert isinstance(sample, Sample) or isinstance(sample, list), \
                "Invalid argument type (only ndarray or a list of ndarrays allowed)."

            assert sample.image.shape[0] > 0 and sample.image.shape[1] > 0, \
                "Invalid input shape. Width and height should be greater than 0."

            if self._allowed_to_perform:
                if isinstance(sample, list):
                    return [self._wrapped.apply(s) for s in sample]
                else:
                    return self._wrapped.apply(sample)
            else:
                return sample

        def apply_on_image(self, image):
            return self._wrapped.apply_on_image(image)

        def apply_on_list(self, images):
            assert isinstance(images, list), "Input argument should be a list"
            return self._wrapped.apply_on_list(images)

        def time(self, image):
            return self._wrapped.time(image)

        def apply_on_annotations(self, annotations):
            return self._wrapped.apply_on_annotations(annotations)

    return Wrapper
