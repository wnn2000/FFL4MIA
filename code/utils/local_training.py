import logging
import numpy as np

import torch
import torch.optim
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from utils.sam import SAM
from utils.gsam import GSAM
from utils.losses import LogitAdjust


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)
        self.targets = self.dataset.targets[self.idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        sample = self.dataset[self.idxs[item]]
        return sample

    def get_num_of_each_class(self, args):
        class_sum = np.array([0] * args.n_classes)
        for idx in self.idxs:
            label = self.dataset.targets[idx]
            class_sum[label] += 1
        return class_sum.tolist()



class LocalUpdate(object):
    def __init__(self, args, client_id, dataset, idxs):
        self.args = args
        self.client_id = client_id
        self.idxs = idxs
        self.local_dataset = DatasetSplit(dataset, idxs)
        self.class_num_list = self.local_dataset.get_num_of_each_class(args)
        logging.info(
            f"---> Client{client_id}, each class num: {self.class_num_list}, total num: {len(self.local_dataset)}")
        self.ldr_train = DataLoader(
            self.local_dataset, batch_size=self.args.batch_size, shuffle=True, num_workers=4)
        self.epoch = 0
        self.iter_num = 0
        self.lr = self.args.base_lr

    
    def train(self, net): # Logit Adjustment is used
        assert len(self.ldr_train.dataset) == len(self.idxs)
        print(f"Client ID: {self.client_id}, Num: {len(self.ldr_train.dataset)}")
        
        net.train()
        # set the optimizer
        optimizer = torch.optim.Adam(net.parameters(), lr=self.lr, betas=(0.9, 0.999), weight_decay=5e-4)

        # train and update
        epoch_loss = []
        ce_criterion = LogitAdjust(cls_num_list=self.class_num_list)
        for epoch in range(self.args.local_ep):
            batch_loss = []
            for samples in self.ldr_train:
                images, labels = samples["image"].to(self.args.device), samples["target"].to(self.args.device)

                logits = net(images)
                loss = ce_criterion(logits, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())

                self.iter_num += 1
            self.epoch = self.epoch + 1
            epoch_loss.append(np.array(batch_loss).mean())

        net.cpu()
        return net.state_dict(), np.array(epoch_loss).mean()
    

    def train_GSAM(self, net, rho):
        assert len(self.ldr_train.dataset) == len(self.idxs)
        print(f"Client ID: {self.client_id}, Num: {len(self.ldr_train.dataset)}")
        
        net.train()
        # set the optimizer
        base_optimizer = torch.optim.Adam(net.parameters(), lr=self.lr, betas=(0.9, 0.999), weight_decay=5e-4)
        optimizer = GSAM(params=net.parameters(), base_optimizer=base_optimizer, model=net, gsam_alpha=self.args.gsam_alpha, rho=rho, adaptive=False)

        # train and update
        epoch_loss = []
        ce_criterion = LogitAdjust(cls_num_list=self.class_num_list)
        for epoch in range(self.args.local_ep):
            batch_loss = []
            for samples in self.ldr_train:
                images, labels = samples["image"].to(self.args.device), samples["target"].to(self.args.device)

                def loss_fn(predictions, targets):
                    return ce_criterion(predictions, targets)
                
                optimizer.set_closure(loss_fn, images, labels)
                predictions, loss = optimizer.step()

                batch_loss.append(loss.item())

                self.iter_num += 1
            self.epoch = self.epoch + 1
            epoch_loss.append(np.array(batch_loss).mean())

        net.cpu()
        base_optimizer = None
        optimizer = None
        return net.state_dict(), np.array(epoch_loss).mean()
    
