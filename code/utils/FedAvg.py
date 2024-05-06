import copy
import torch
import numpy as np


def FedAvg(w, dict_len):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():        
        w_avg[k] = w_avg[k] * dict_len[0] 
        for i in range(1, len(w)):
            w_avg[k] += w[i][k] * dict_len[i]
        w_avg[k] = w_avg[k] / sum(dict_len)
    return w_avg


def get_agg_weights_FedGA(d, last_weights, loss_results, args):
    d = d / args.n_clients
    weights = d * loss_results / np.max(np.abs(loss_results)) + last_weights
    weights = weights / weights.sum()
    return weights


class FedCE_aggregator():
    def __init__(self, model):
        self.model = model
        self.fedce_cos_param_list = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.fedce_cos_param_list.append(name)

        # Aggregator needs to keep record of historical
        # cosine similarity for FedCM coefficients
        self.fedce_cos_sim = {}
        self.fedce_coef_time = {}
        
    def assemble(self, local_models_weights, current_round, fedce_minus_vals, args):

        self.fedce_cos_sim[current_round] = {}

        if current_round == 0:
            # round 0, initialize uniform fedce_coef
            self.fedce_coef = np.ones(args.n_clients).astype("float")
            self.fedce_coef = self.fedce_coef / self.fedce_coef.sum()

        # generate consensus gradient with current FedCE coefficients
        consensus_grad = []
        global_weights = self.model.state_dict()
        for idx, name in enumerate(global_weights):
            if name in self.fedce_cos_param_list:
                temp = torch.zeros_like(global_weights[name])
                for client_id in range(args.n_clients):
                    temp += self.fedce_coef[client_id] * (torch.as_tensor(local_models_weights[client_id][name]) - torch.as_tensor(global_weights[name]))
                consensus_grad.append(temp.data.view(-1))

        # flatten for cosine similarity computation
        consensus_grads_vec = torch.cat(consensus_grad).to("cpu")

        # generate minus gradients and compute cosine similarity
        for client_id in range(args.n_clients):
            site_grad = []
            for name in self.fedce_cos_param_list:
                site_grad.append((torch.as_tensor(local_models_weights[client_id][name]) - torch.as_tensor(global_weights[name])).data.view(-1))
            site_grads_vec = torch.cat(site_grad).to("cpu")
            # minus gradient
            minus_grads_vec = consensus_grads_vec - self.fedce_coef[client_id] * site_grads_vec
            # compute cosine similarity
            fedce_cos_sim_site = (
                torch.cosine_similarity(site_grads_vec, minus_grads_vec, dim=0).detach().cpu().numpy().item()
            )
            # append to record dict
            self.fedce_cos_sim[current_round][client_id] = fedce_cos_sim_site

        # compute cos_weights and minus_vals based on the record for each site
        fedce_cos_weights = []
        for client_id in range(args.n_clients):
            # cosine similarity
            cos_accu_avg = np.mean([self.fedce_cos_sim[i][client_id] for i in range(current_round + 1)])
            fedce_cos_weights.append(1.0 - cos_accu_avg)

        # normalize
        fedce_cos_weights /= np.sum(fedce_cos_weights)
        # fedce_cos_weights = np.clip(fedce_cos_weights, a_min=1e-3, a_max=None)
        fedce_minus_vals /= np.sum(fedce_minus_vals)
        # fedce_minus_vals = np.clip(fedce_minus_vals, a_min=1e-3, a_max=None)

        # two aggregation strategies
        if args.fedce_mode == "times":
            new_fedce_coef = [c_w * mv_w for c_w, mv_w in zip(fedce_cos_weights, fedce_minus_vals)]
        elif args.fedce_mode == "plus":
            new_fedce_coef = [c_w + mv_w for c_w, mv_w in zip(fedce_cos_weights, fedce_minus_vals)]
        else:
            raise NotImplementedError

        # normalize again
        new_fedce_coef /= np.sum(new_fedce_coef)
        # new_fedce_coef = np.clip(new_fedce_coef, a_min=1e-3, a_max=None)

        # update fedce_coef
        self.fedce_coef_time[current_round] = new_fedce_coef
        self.fedce_coef = np.mean([self.fedce_coef_time[i] for i in range(current_round + 1)], axis=0)
        assert len(self.fedce_coef) == args.n_clients, self.fedce_coef
