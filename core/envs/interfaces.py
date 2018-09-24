
import abc

class Measurable:
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def measure_error(self, true_y, predicted_y, mask):
        pass
