import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch.nn as nn
import torch
import torch_scatter

'''
Over-segment->SLIC
Input:img->B×C×H×W   spx->B×H×W
Output:out->B×C×H×W
'''
class SupPixPool(nn.Module):
    def __init__(self, mode='max'):
        super(SupPixPool, self).__init__()
        self.mode = mode
    def forward(self, img, spx):
        assert (spx.shape[-2:] == img.shape[-2:])
        batch_size = int(img.shape[0])
        out = {}
        feature_dim = int(img.shape[1])
        for batch in range(batch_size):
            K = int(spx[batch].max()+1)
            feature = torch.empty([feature_dim, K]).to(img[0].device)
            for i in range(feature_dim):
                feature[i] = torch_scatter.scatter(img[batch][i].flatten(), spx[batch].flatten(), reduce=self.mode)
            out[batch] = feature
        return out

class SupPixUnpool(nn.Module):
    def __init__(self):
        super(SupPixUnpool, self).__init__()
    def forward(self, pooled, spx):
        bs = len(pooled)
        outShape = [bs, pooled[0].shape[0], spx.shape[-2], spx.shape[-1]]
        out = torch.empty(outShape).to(pooled[0].device)
        for batch in range(bs):
            out[batch, :, :, :] = pooled[batch][:, spx[batch, :, :]]
        return out