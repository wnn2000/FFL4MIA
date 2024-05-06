import numpy as np

import torch
import torch.optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, balanced_accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, confusion_matrix



def globaltest(net, test_dataset, args):
    net.eval()
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size*2, shuffle=False, num_workers=4)   # args.batch_size*4
    all_preds = np.array([])
    all_probs = []
    all_labels = np.array(test_dataset.targets)
    with torch.no_grad():
        for samples in test_loader:
            images = samples["image"].to(args.device)
            outputs = net(images)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            all_probs.append(probs.detach().cpu())
            all_preds = np.concatenate([all_preds, preds.detach().cpu().numpy()], axis=0)

    all_probs = torch.cat(all_probs).numpy()
    assert all_probs.shape[0] == len(test_dataset)
    assert all_probs.shape[1] == args.n_classes

    acc = accuracy_score(all_labels, all_preds)
    bacc = balanced_accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted")
    bf1 = f1_score(all_labels, all_preds, average="macro")
    auc = roc_auc_score(all_labels, all_probs, multi_class="ovr", average="weighted")
    bauc = roc_auc_score(all_labels, all_probs, multi_class="ovo", average="macro")

    cm = confusion_matrix(all_labels, all_preds)

    net.cpu()

    return {"acc": acc, 
            "bacc": bacc,
            "f1": f1,
            "bf1": bf1,
            "auc": auc,
            "bauc": bauc,
            "cm": cm}