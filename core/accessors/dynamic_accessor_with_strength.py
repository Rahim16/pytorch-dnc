from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import torch.nn as nn
from torch.autograd import Variable

from core.accessors.dynamic_accessor import DynamicAccessor
from core.heads.dynamic_read_head_with_similarity import DynamicReadHeadWithSimilarity as ReadHead

class DynamicAccessorWithStrength(DynamicAccessor):
    def __init__(self, args):
        super(DynamicAccessorWithStrength, self).__init__(args)

        self.read_heads = ReadHead(self.read_head_params)

        self._reset()
