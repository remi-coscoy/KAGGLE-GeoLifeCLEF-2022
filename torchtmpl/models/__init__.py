# coding: utf-8

# External imports
import torch
import logging
# Local imports
from .base_models import *
from .cnn_models import *
from .transfo_models import *
from .pretrained_models import *


def build_model(cfg, input_img_size, input_tab_size, num_classes):
    logging.info(input_img_size)
    logging.info(input_tab_size)
    logging.info(num_classes)

    return eval(f"{cfg['class']}(cfg, input_img_size, input_tab_size, num_classes)")
