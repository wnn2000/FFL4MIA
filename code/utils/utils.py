import copy
import random
import os
import sys
import shutil
import numpy as np
import pandas as pd
import logging
from tensorboardX import SummaryWriter


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def set_output_files(args):
    outputs_dir = 'output/' + str(args.dataset) + f'/n{args.noise}'
    if not os.path.exists(outputs_dir):
        os.mkdir(outputs_dir)

    exp_dir = os.path.join(outputs_dir, args.alg + '_a' + str(args.attempts) + '_lr' + str(args.base_lr) + '_n' + str(args.noise) 
                           + '_e' + str(args.local_ep) + '_alpha' + str(args.gsam_alpha)+ '_rho' + str(args.gsam_rho) + '_q' + str(args.q))
    if args.iid:
        exp_dir += '_iid'
    else:
        exp_dir += '_noniid'+'_dir'+str(args.alpha_dirichlet)
    if not os.path.exists(exp_dir):
        os.mkdir(exp_dir)

    models_dir = os.path.join(exp_dir, 'models')
    if not os.path.exists(models_dir):
        os.mkdir(models_dir)

    logs_dir = os.path.join(exp_dir, 'logs')
    if not os.path.exists(logs_dir):
        os.mkdir(logs_dir)

    tensorboard_dir = os.path.join(exp_dir, 'tensorboard')
    if not os.path.exists(tensorboard_dir):
        os.mkdir(tensorboard_dir)

    code_dir = os.path.join(exp_dir, 'code')
    if os.path.exists(code_dir):
        shutil.rmtree(code_dir)
    shutil.copytree('./code', code_dir, ignore=shutil.ignore_patterns('.git', '__pycache__'))

    logging.basicConfig(filename=logs_dir+'/logs.txt', level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    writer = SummaryWriter(tensorboard_dir)
    return writer, models_dir, logs_dir


def compute_loss(dataset, net, args):
    loader = DataLoader(copy.deepcopy(dataset), batch_size=64, shuffle=False, num_workers=4)
    ce_criterion = nn.CrossEntropyLoss(reduction="none")
    net.eval()
    with torch.no_grad():
        for i, samples in enumerate(loader):
            images, labels = samples["image"].to(args.device), samples["target"].to(args.device)
            logits = net(images)
            loss = ce_criterion(logits, labels)
            if i == 0:
                loss_all = loss.cpu().numpy()
                label_all = labels.cpu().numpy()
                prob_all = torch.softmax(logits, dim=-1).cpu().numpy()
                pred_all = torch.argmax(logits, dim=-1).cpu().numpy()
            else:
                loss_all = np.concatenate([loss_all, loss.cpu().numpy()], axis=0)
                label_all = np.concatenate([label_all, labels.cpu().numpy()], axis=0)
                prob_all = np.concatenate([prob_all, torch.softmax(logits, dim=-1).cpu().numpy()], axis=0)
                pred_all = np.concatenate([pred_all, torch.argmax(logits, dim=-1).cpu().numpy()], axis=0)

    assert (label_all == dataset.targets).all()

    return loss_all, label_all, prob_all, pred_all


def get_perturb_model(dataset, net, rho, args):
    loader = DataLoader(copy.deepcopy(dataset), batch_size=64, shuffle=False, num_workers=4)
    ce_criterion = nn.CrossEntropyLoss(reduction="sum")
    net.eval()
    net.zero_grad()

    with torch.enable_grad():
        for i, samples in enumerate(loader):
            images, labels = samples["image"].to(args.device), samples["target"].to(args.device)
            logits = net(images)
            loss = ce_criterion(logits, labels)
            loss.backward()
    
    grad_norm = torch.norm(torch.stack([p.grad.norm(p=2) for p in net.parameters() if p.grad is not None]), p=2)
    perturb_eps = 1e-12
    scale = rho / (grad_norm + perturb_eps)
    for p in net.parameters():
        if p.grad is not None:
            e_w = p.grad * scale.to(p)
            # temp = p.data
            # assert p.data == p
            p.data.add_(e_w)
            # assert p.data == temp + e_w

    return net


def get_minus_model(global_model, last_round_weight, fedce_weight):
    minus_model = copy.deepcopy(global_model)
    for key in minus_model.state_dict().keys():
        temp = (global_model.state_dict()[key] - fedce_weight * last_round_weight[key]) / (
            1 - fedce_weight
        )
        minus_model.state_dict()[key].data.copy_(temp)
    return minus_model


def local_valid(model, dataset, args):
    loader = DataLoader(copy.deepcopy(dataset), batch_size=args.batch_size, shuffle=False)
    model.eval()
    model.to(args.device)
    with torch.no_grad():
        for i, samples in enumerate(loader):
            images, labels = samples["image"].to(args.device), samples["target"].to(args.device)
            logits = model(images)
            preds = torch.argmax(logits, dim=-1)
            if i==0:
                pred_all = preds.cpu().numpy()
                label_all = labels.cpu().numpy()
            else:
                pred_all = np.concatenate([pred_all, preds.cpu().numpy()], axis=0)
                label_all = np.concatenate([label_all, labels.cpu().numpy()], axis=0)

    from sklearn.metrics import balanced_accuracy_score, accuracy_score

    return balanced_accuracy_score(label_all, pred_all)



def get_current_rho(rnd, args):
    if args.alg == "FedISM+":
        rho = args.gsam_rho * ((rnd+1) / args.rounds)**args.p_rho_curve
    else:
        rho = args.gsam_rho

    return rho