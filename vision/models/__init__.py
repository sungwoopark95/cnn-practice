from .alexnet import AlexNet
from .zfnet import ZFNet
from .vgg import VGG
from .vgg19 import VGG19
from .googlenet import GoogLeNet
from .inception_v2 import InceptionV2

METHOD_DICT = {
    "alexnet": AlexNet,
    "zfnet": ZFNet,
    "vgg": VGG,
    "vgg19": VGG19,
    "googlenet": GoogLeNet,
    "inception-v2": InceptionV2,
}

def get_model(name):
    assert name in METHOD_DICT
    return METHOD_DICT[name]