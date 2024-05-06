# This file is borrowed from https://github.com/Xu-Jingyi/FedCorr/blob/main/util/sampling.py. We change something.

import numpy as np


def iid_sampling(n_train, num_users, seed):
    np.random.seed(seed)
    num_items = int(n_train/num_users)
    dict_users, all_idxs = {}, [i for i in range(n_train)] # initial user and index for whole dataset
    all_idxs = np.random.permutation(np.array(all_idxs))
    partition = np.array_split(all_idxs, num_users)
    for i in range(num_users):
        dict_users[i] = partition[i]

    for key in dict_users.keys():
        dict_users[key] = list(dict_users[key])
    return dict_users



def non_iid_dirichlet_sampling(y_train, num_classes, p, num_users, seed, alpha_dirichlet, corrupted_num):
    np.random.seed(seed)
    dict_users = {}
    for class_i in range(num_classes):
        all_idxs = np.where(y_train==class_i)[0]
        num_of_clean = int(np.round((num_users-corrupted_num) / num_users * len(all_idxs)))
        all_idxs_clean = all_idxs[:num_of_clean]
        all_idxs_corrupted = all_idxs[num_of_clean:]

        p_dirichlet_clean = np.random.dirichlet([alpha_dirichlet] * (num_users-corrupted_num))
        p_dirichlet_corrupted = np.random.dirichlet([alpha_dirichlet] * (corrupted_num))

        assignment_clean = np.random.choice(np.arange(num_users-corrupted_num), size=len(all_idxs_clean), p=p_dirichlet_clean.tolist())
        assignment_corrupted = np.random.choice(np.arange(num_users-corrupted_num, num_users), size=len(all_idxs_corrupted), p=p_dirichlet_corrupted.tolist())
        assignment = np.concatenate([assignment_clean, assignment_corrupted], axis=0)

        for client_k in range(num_users):
            if client_k in dict_users:
                dict_users[client_k] = set(dict_users[client_k] | set(all_idxs[(assignment == client_k)]))
            else:
                dict_users[client_k] = set(all_idxs[(assignment == client_k)])
    
    for key in dict_users.keys():
        dict_users[key] = list(dict_users[key])
    return dict_users