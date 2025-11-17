import os
from importlib import import_module

model_class_dict = {}


def register_model(model_class):
    model_name = model_class.__name__.lower()
    assert model_name not in model_class_dict, f"Model name '{model_name}' is already registered in model_class_dict."
    model_class_dict[model_name] = model_class
    return model_class


def get_model_class(model_name: str):
    model_name = model_name.lower()
    return model_class_dict[model_name]


for module in os.listdir(os.path.dirname(__file__)):
    if module == '__init__.py' or module[-3:] != '.py':
        continue
    import_module('src.LiSCa_Net.model.{}'.format(module[:-3]))
del module
