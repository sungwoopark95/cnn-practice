from .alexnet import AlexNet
from .zfnet import ZFNet
from .vgg import VGG


METHOD_DICT = {
    "alexnet": AlexNet,
    "zfnet": ZFNet,
    "vgg": VGG,
}

def get_model(name):
    assert name in METHOD_DICT
    return METHOD_DICT[name]