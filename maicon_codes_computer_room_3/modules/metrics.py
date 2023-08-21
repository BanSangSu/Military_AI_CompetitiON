"""Metric 함수 정의
"""
import torch.nn as nn
import torch
import numpy as np
SMOOTH = 1e-6

def get_metric_function(metric_function_str):
    """
    Add metrics, weights for weighted score
    """
    if metric_function_str == 'dice_coef':
        dice_coef = DiceCoefficient()
        return dice_coef.get_dice_coefficient

    elif metric_function_str == 'miou':
        iou = Iou()
        return iou.get_miou

    elif metric_function_str == 'iou1':
        iou =Iou(class_num=1)
        return iou.get_iou 

    elif metric_function_str == 'iou2':
        iou =Iou(class_num=2)
        return iou.get_iou

    elif metric_function_str == 'iou3':
        iou =Iou(class_num=3)
        return iou.get_iou

class DiceCoefficient:
    def __init__(self, class_num:int=0):
        self.class_num = class_num

    def get_dice_coefficient(self, outputs: torch.Tensor, labels: torch.Tensor):
        outputs = torch.sigmoid(outputs)       
        
        #flatten label and prediction tensors
        outputs = outputs.view(-1)
        labels = labels.view(-1)
        intersection = (outputs * labels).sum()                            
        dice_loss = (2.*intersection)/(outputs.sum() + labels.sum())
        return dice_loss

        
class Iou:
    
    def __init__(self, class_num:int=0):
        self.class_num = class_num
        
    def get_iou(self, outputs: torch.Tensor, labels: torch.Tensor):
        mask_value = self.class_num

        batch_size = outputs.size()[0]
            
        intersection = ((outputs.int() == mask_value) & (labels.int() == mask_value) & (outputs.int() == labels.int())).float()
        intersection = intersection.view(batch_size, -1).sum(1)

        union = ((outputs.int() == mask_value) | (labels.int() == mask_value)).float()
        union = union.view(batch_size, -1).sum(1)

        iou = (intersection + SMOOTH) / (union + SMOOTH)
            
        return iou.mean()

    def get_miou(self, outputs: torch.Tensor, labels: torch.Tensor):
        # Not exactly match the mIoU definition
        batch_size = outputs.size()[0]
    
        intersection = ((outputs.int() > 0) & (labels.int() > 0) & (outputs.int() == labels.int())).float()
        intersection = intersection.view(batch_size, -1).sum(1)
    
        union = ((outputs.int() > 0) | (labels.int() > 0)).float()
        union = union.view(batch_size, -1).sum(1)
    
        iou = (intersection + SMOOTH) / (union + SMOOTH)
    
        return iou.mean()

