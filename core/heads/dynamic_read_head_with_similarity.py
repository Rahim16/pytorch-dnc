from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.heads.dynamic_read_head import DynamicReadHead

class DynamicReadHeadWithSimilarity(DynamicReadHead):
    def __init__(self, args):
        super(DynamicReadHeadWithSimilarity, self).__init__(args)
        self._reset()

    def _access(self, memory_vb): # read
        # get the read vector [batch_size x num_heads x mem_wid]
        read_vec_vb = super(DynamicReadHeadWithSimilarity, self)._access(memory_vb)
        # append the max cosine similarity to it [batch_size x num_heads x mem_hei]
        max_similarities = F.max_pool1d(self.similarities, self.similarities.size(-1)) # [batch_size x num_heads x 1]

        read_vector_with_similarity = torch.cat((read_vec_vb, max_similarities), -1)
        return read_vector_with_similarity
