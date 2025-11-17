import os
import torch.nn as nn
from importlib import import_module
loss_class_dict = {}


def register_loss(loss_class):
    loss_name = loss_class.__name__
    assert loss_name not in loss_class_dict, f"Loss name '{loss_name}' is already registered in loss_class_dict."
    loss_class_dict[loss_name] = loss_class
    return loss_class


for module in os.listdir(os.path.dirname(__file__)):
    if module == '__init__.py' or module[-3:] != '.py':
        continue
    import_module('src.LiSCa_Net.loss.{}'.format(module[:-3]))
del module


class Loss(nn.Module):
    def __init__(self, loss_string):
        super().__init__()
        loss_string = loss_string.replace(' ', '')
        self.loss_list = []
        for item in loss_string.split('+'):
            weight, name = item.split('*')
            ratio = True if 'r' in weight else False
            weight = float(weight.replace('r', ''))

            if name in loss_class_dict:
                self.loss_list.append({
                    'name': name, 'weight': float(weight),
                    'func': loss_class_dict[name](), 'ratio': ratio
                })
            else:
                raise RuntimeError(f'Undefined loss term: {name}')

    def forward(self, target_noisy, model, weight):
        losses = {}
        for single_loss in self.loss_list:
            base_loss = single_loss['func'](target_noisy, model)
            if weight is not None:
                base_loss = weight * base_loss
            losses[single_loss['name']] = base_loss.mean()
        return losses




