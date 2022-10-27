# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''
GIN graph classifier.
'''
import torch
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_mean_pool

from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR

class GIN(torch.nn.Module):
    def __init__(self, num_features=1, num_classes=1, num_hidden=32):
        super(GIN, self).__init__()
        dim = num_hidden

        nn1 = Sequential(Linear(num_features, dim), ReLU(), Linear(dim, dim))
        self.conv1 = GINConv(nn1)
        self.bn1 = torch.nn.BatchNorm1d(dim)

        nn2 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv2 = GINConv(nn2)
        self.bn2 = torch.nn.BatchNorm1d(dim)

        nn3 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv3 = GINConv(nn3)
        self.bn3 = torch.nn.BatchNorm1d(dim)

        nn4 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv4 = GINConv(nn4)
        self.bn4 = torch.nn.BatchNorm1d(dim)

        nn5 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv5 = GINConv(nn5)
        self.bn5 = torch.nn.BatchNorm1d(dim)

        self.fc1 = Linear(dim, dim)
        self.fc2 = Linear(dim, num_classes)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        x = F.relu(self.conv3(x, edge_index))
        x = self.bn3(x)
        x = F.relu(self.conv4(x, edge_index))
        x = self.bn4(x)
        x = F.relu(self.conv5(x, edge_index))
        x = self.bn5(x)

        x = global_mean_pool(x, batch)

        x = F.relu(self.fc1(x))

        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)

def mixup_cross_entropy_loss(input, target, size_average=True):
    '''
    https://github.com/moskomule/mixup.pytorch
    '''
    assert input.size() == target.size()
    assert isinstance(input, Variable) and isinstance(target, Variable)
    # Negative log likelihood loss function as dot of input and target
    # Normalize by size if averaging flag is on
    loss = - torch.sum(input * target)
    return loss / input.size()[0] if size_average else loss
