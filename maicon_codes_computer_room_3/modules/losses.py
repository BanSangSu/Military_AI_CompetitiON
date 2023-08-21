"""Losses
    * https://github.com/JunMa11/SegLoss
"""


from torch.nn import functional as F
import torch.nn as nn
import torch

def get_loss_function(loss_function_str: str):

    if loss_function_str == 'DiceBCELoss':
        
        return DiceBCELoss

    elif loss_function_str == 'MeanCCELoss':

        return CCE

    elif loss_function_str == 'GDLoss':

        return GeneralizedDiceLoss

class DiceBCELoss(nn.Module):
    def __init__(self, **kwargs):
        super(DiceBCELoss, self).__init__()

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss
        
        return Dice_BCE

class CCE(nn.Module):

    def __init__(self, weight, **kwargs):
        super(CCE, self).__init__()
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.weight = torch.Tensor(weight).to(device)

    def forward(self, inputs, targets):
        
        loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.weight)
        unique_values, unique_counts = torch.unique(targets, return_counts=True)
        selected_weight = torch.index_select(input=self.weight, dim=0, index=unique_values)

        numerator = loss.sum()                               # weighted losses
        denominator = (unique_counts*selected_weight).sum()  # weigthed counts

        loss = numerator/denominator

        return loss


class GeneralizedDiceLoss(nn.Module):
    
    def __init__(self, **kwargs):
        super(GeneralizedDiceLoss, self).__init__()
        self.scaler = nn.Softmax(dim=1)  # Softmax for loss

    def forward(self, inputs, targets):

        targets = targets.contiguous()
        targets = torch.nn.functional.one_hot(targets.to(torch.int64), inputs.size()[1])  # B, H, W, C

        inputs = inputs.contiguous()
        inputs = self.scaler(inputs)
        inputs = inputs.permute(0, 2, 3, 1)  # B, H, W, C

        w = 1. / (torch.sum(targets, (0, 1, 2)) ** 2 + 1e-9)

        numerator = targets * inputs
        numerator = w * torch.sum(numerator, (0, 1, 2))
        numerator = torch.sum(numerator)

        denominator = targets + inputs
        denominator = w * torch.sum(denominator, (0, 1, 2))
        denominator = torch.sum(denominator)

        dice = 2. * (numerator + 1e-9) / (denominator + 1e-9)

        return 1. - dice