import os
import sys


FILE_DIR = os.path.dirname(os.path.abspath(__file__))
# avoid annoying import errors...
sys.path.append(FILE_DIR)

import core.models.cifar10 as cifar10
import core.models.cifar10sm as cifar10sm
import core.models.vision as vision
import core.models.wide_resnet as wide
