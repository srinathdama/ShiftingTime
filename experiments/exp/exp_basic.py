# MIT License
#
# Original work Copyright (c) 2021 THUML @ Tsinghua University
# Modified work Copyright (c) 2025 DACElab @ University of Toronto
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT


import os
import torch
from models import KRNO

class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        if args.model == 'KRNO':
            self.model_dict = {
                'KRNO': KRNO
            }
        else:
            raise('model name should be [KRNO]')
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
