# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch optimization for BERT model."""

import logging
import math

import torch

logger = logging.getLogger(__name__)


class constant_schedule():
    def __init__(self, last_epoch=-1):
        """ Create a schedule with a constant learning rate."""
        self.last_epoch = last_epoch
        
    def get_lr_this_step(self, current_step, base_lr):

        return base_lr


class constant_schedule_with_warmup():
    
    def __init__(self, num_warmup_steps, last_epoch=-1):
        """ Create a schedule with a constant learning rate preceded by a warmup
        period during which the learning rate increases linearly between 0 and 1.
        """
        self.num_warmup_steps = num_warmup_steps
        self.last_epoch = last_epoch

    def lr_lambda(self, current_step):
        if current_step < self.num_warmup_steps:
            return float(current_step) / float(max(1.0, self.num_warmup_steps))
        return 1.0
    
    def get_lr_this_step(self, current_step, base_lr):

        return self.lr_lambda(current_step) * base_lr


class linear_schedule_with_warmup():
    def __init__(self, num_warmup_steps, num_training_steps, last_epoch=-1):
        """ Create a schedule with a learning rate that decreases linearly after
        linearly increasing during a warmup period.
        """
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        self.last_epoch = last_epoch

    def lr_lambda(self, current_step):
        if current_step < self.num_warmup_steps:
            return float(current_step) / float(max(1, self.num_warmup_steps))
        return max(
            0.0, float(self.num_training_steps - current_step) / float(max(1, self.num_training_steps - self.num_warmup_steps))
        )

    def get_lr_this_step(self, current_step, base_lr):

        return self.lr_lambda(current_step) * base_lr

class cosine_schedule_with_warmup():
    
    def __init__(self, num_warmup_steps, num_training_steps, num_cycles=0.5, last_epoch=-1):
        """ Create a schedule with a learning rate that decreases following the
        values of the cosine function between 0 and `pi * cycles` after a warmup
        period during which it increases linearly between 0 and 1.
        """
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        self.num_cycles = num_cycles
        self.last_epoch = last_epoch

    def lr_lambda(self, current_step):
        if current_step < self.num_warmup_steps:
            return float(current_step) / float(max(1, self.num_warmup_steps))
        progress = float(current_step - self.num_warmup_steps) / float(max(1, self.num_training_steps - self.num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(self.num_cycles) * 2.0 * progress)))

    def get_lr_this_step(self, current_step, base_lr):

        return self.lr_lambda(current_step) * base_lr

class cosine_with_hard_restarts_schedule_with_warmup():

    def __init__(self, 
        num_warmup_steps, num_training_steps, num_cycles=1.0, last_epoch=-1
        ):
        """ Create a schedule with a learning rate that decreases following the
        values of the cosine function with several hard restarts, after a warmup
        period during which it increases linearly between 0 and 1.
        """
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        self.num_cycles = num_cycles
        self.last_epoch = last_epoch

    def lr_lambda(self, current_step):
        if current_step < self.num_warmup_steps:
            return float(current_step) / float(max(1, self.num_warmup_steps))
        progress = float(current_step - self.num_warmup_steps) / float(max(1, self.num_training_steps - self.num_warmup_steps))
        if progress >= 1.0:
            return 0.0
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * ((float(self.num_cycles) * progress) % 1.0))))

    def get_lr_this_step(self, current_step, base_lr):

        return self.lr_lambda(current_step) * base_lr