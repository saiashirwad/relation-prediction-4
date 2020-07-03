# coding:utf-8
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import os
import time
import sys
import datetime
import ctypes
import json
import numpy as np
import copy
import tqdm
from livelossplot import PlotLosses
from ranger import *

import IPython


class Trainer(object):

    def __init__(self,
                 model=None,
                 data_loader=None,
                 train_times=1000,
                 lr=1e-3,
                 alpha=0.5,
                 use_gpu=True,
                 opt_method="sgd",
                 save_steps=None,
                 checkpoint_dir=None,):

        self.work_threads = 8
        self.train_times = train_times

        self.opt_method = opt_method
        self.optimizer = None
        self.lr_decay = 0
        self.weight_decay = 0
        self.alpha = alpha
        self.lr = lr

        self.model = model
        self.data_loader = data_loader
        self.use_gpu = use_gpu
        self.save_steps = save_steps
        self.checkpoint_dir = checkpoint_dir

        self.liveplot = PlotLosses()

    def train_one_step(self, data, stage=1):
        self.optimizer.zero_grad()
        self.model.zero_grad()
        loss = self.model({
            'batch_h': self.to_var(data['batch_h'], self.use_gpu),
            'batch_t': self.to_var(data['batch_t'], self.use_gpu),
            'batch_r': self.to_var(data['batch_r'], self.use_gpu),
            'batch_y': self.to_var(data['batch_y'], self.use_gpu),
            'mode': data['mode'],
            'stage': stage
        })

        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 2)
        self.optimizer.step()
        return loss.item()

    def run(self, lr=None, alpha=None, weight_decay=None, train_times=None, stage=1, multiplier=1):
        if lr:
            self.lr = lr
        if alpha:
            self.alpha = alpha
        if weight_decay:
            self.weight_decay = weight_decay
        if train_times:
            self.train_times = train_times
        if self.use_gpu:
            self.model.cuda()

        if self.optimizer is not None:
            pass
        elif self.opt_method == "Adagrad" or self.opt_method == "adagrad":
            self.optimizer = optim.Adagrad(
                self.model.parameters(),
                lr=self.lr,
                lr_decay=self.lr_decay,
                weight_decay=self.weight_decay,
            )
        elif self.opt_method == "Adadelta" or self.opt_method == "adadelta":
            self.optimizer = optim.Adadelta(
                self.model.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
        elif self.opt_method == "Adam" or self.opt_method == "adam":
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
        elif self.opt_method == "ranger":
            if not lr:
                self.optimizer = Ranger(
                    self.model.parameters(), lr=self.lr, alpha=self.alpha)
            else:
                self.optimizer = Ranger(
                    self.model.parameters(), lr=lr, alpha=self.alpha)
        elif self.opt_method == "rangerva":
            self.optimizer = RangerVA(self.model.parameters(), lr=lr)
        else:
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.alpha,
                weight_decay=self.weight_decay,
            )
        print("Finish initializing...")

        # training_range = tqdm.tqdm(range(self.train_times))
        training_range = tqdm.trange(self.train_times)
        # training_range = range(self.train_times)
        for epoch in training_range:
            res = 0.0
            for data in self.data_loader:
                loss = multiplier * self.train_one_step(data, stage)
                res += loss
            self.liveplot.update({
                'loss': res
            })
            self.liveplot.send()
            if self.save_steps and self.checkpoint_dir and (epoch + 1) % self.save_steps == 0:
                print("Epoch %d has finished, saving..." % (epoch))
                self.model.save_checkpoint(os.path.join(
                    self.checkpoint_dir + "-" + str(epoch) + ".ckpt"))

    def set_model(self, model):
        self.model = model

    def to_var(self, x, use_gpu):
        if use_gpu:
            return Variable(torch.from_numpy(x).cuda())
        else:
            return Variable(torch.from_numpy(x))

    def set_use_gpu(self, use_gpu):
        self.use_gpu = use_gpu

    def set_alpha(self, alpha):
        self.alpha = alpha

    def set_lr_decay(self, lr_decay):
        self.lr_decay = lr_decay

    def set_weight_decay(self, weight_decay):
        self.weight_decay = weight_decay

    def set_opt_method(self, opt_method):
        self.opt_method = opt_method

    def set_train_times(self, train_times):
        self.train_times = train_times

    def set_save_steps(self, save_steps, checkpoint_dir=None):
        self.save_steps = save_steps
        if not self.checkpoint_dir:
            self.set_checkpoint_dir(checkpoint_dir)

    def set_checkpoint_dir(self, checkpoint_dir):
        self.checkpoint_dir = checkpoint_dir
