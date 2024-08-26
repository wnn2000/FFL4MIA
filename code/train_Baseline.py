import os
import copy
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter

import torch
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter

from utils.options import args_parser
from utils.local_training import LocalUpdate
from utils.FedAvg import FedAvg, get_agg_weights_FedGA, FedCE_aggregator
from utils.utils import set_seed, set_output_files, compute_loss, get_minus_model, local_valid, get_perturb_model
from utils.evaluation import globaltest

from datasets.dataset import get_dataset
from models.build_model import build_model

np.set_printoptions(threshold=np.inf)





if __name__ == '__main__':
    args = args_parser()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    # ------------------------------ output files ------------------------------
    writer, models_dir, logs_dir = set_output_files(args)

    # ------------------------------ deterministic or not ------------------------------
    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        set_seed(args.seed)

    # ------------------------------ dataset ------------------------------
    dataset_train, datasets_test, dict_users = get_dataset(args)
    assert isinstance(datasets_test, list)
    for i in range(len(datasets_test)):
        logging.info(f"TestSet (d{i}), path: {datasets_test[i].datapath}, total num: {len(datasets_test[i])}, {Counter(datasets_test[i].targets)}")

    # ------------------------------ settings ------------------------------
    user_id = list(range(args.n_clients))
    dict_len = [len(dict_users[idx]) for idx in user_id]
    trainer_locals = []
    data_distribution = torch.ones((args.n_clients, args.n_classes)).float()
    for i in user_id:
        trainer_locals.append(LocalUpdate(
            args, i, copy.deepcopy(dataset_train), dict_users[i]))
        data_distribution[i] = torch.tensor(trainer_locals[i].class_num_list)

    data_distribution = data_distribution.float()
    logging.info("Data Distribution")
    logging.info(data_distribution.int())
    
    if args.dataset in ["ICH_20", "ISIC2019_20"]:
        Result_Metric = np.zeros((args.rounds, 36)).astype("float")
        column_names = ["clean_ACC_0", "clean_ACC_1", "clean_ACC_2",
                        "clean_AUC_0", "clean_AUC_1", "clean_AUC_2", 
                        "clean_BACC_0", "clean_BACC_1", "clean_BACC_2",
                        "clean_BAUC_0", "clean_BAUC_1", "clean_BAUC_2",
                        "clean_F1_0", "clean_F1_1", "clean_F1_2", 
                        "clean_BF1_0", "clean_BF1_1", "clean_BF1_2", 
                        "corrupted_ACC_0", "corrupted_ACC_1", "corrupted_ACC_2", 
                        "corrupted_AUC_0", "corrupted_AUC_1", "corrupted_AUC_2",
                        "corrupted_BACC_0", "corrupted_BACC_1", "corrupted_BACC_2", 
                        "corrupted_BAUC_0", "corrupted_BAUC_1", "corrupted_BAUC_2",
                        "corrupted_F1_0", "corrupted_F1_1", "corrupted_F1_2", 
                        "corrupted_BF1_0", "corrupted_BF1_1", "corrupted_BF1_2"]
    else:
        raise

    df = pd.DataFrame(Result_Metric, columns=column_names)

    # ------------------------------ begin training ------------------------------
    for Current_RUN in range(args.runs):

        # config
        set_seed(int(Current_RUN)+args.seed)
        netglob = build_model(args).cpu()

        last_aggregation_weights = None
        if args.alg == "FedGA":
            loss_results = None
        elif args.alg == "FedCE":
            FedCE_AGG = FedCE_aggregator(copy.deepcopy(netglob).cpu())
            model_last_round = []
            fedce_minus_val = {}

        logging.info(f"\n===============================> beging, run: {Current_RUN} <===============================\n")

        for Current_Round in range(args.rounds):
            logging.info(f"\n--------------------------> training, run: {Current_RUN}, round: {Current_Round} <--------------------------")
            w_locals, loss_locals = [], []

            ################ settings before local training, for FedCE only
            if args.alg == "FedCE":
                if Current_Round > 0:
                    fedce_minus_val[Current_Round] = []
                    assert len(model_last_round) == args.n_clients
                    for i in user_id:
                        fedce_minus_model = get_minus_model(copy.deepcopy(netglob), model_last_round[i], FedCE_AGG.fedce_coef[i])
                        local_acc = local_valid(fedce_minus_model, copy.deepcopy(trainer_locals[i].local_dataset), args)
                        fedce_minus_val[Current_Round].append(local_acc)
                else:
                    fedce_minus_val[Current_Round] = [0.0] * args.n_clients
                minus_val = 1.0 - np.mean([fedce_minus_val[i] for i in range(Current_Round + 1)], axis=0)
                assert len(minus_val) == args.n_clients
                logging.info(f"------> round: {Current_Round}, minus_val:")
                logging.info(minus_val)
            else:
                pass
            ################ settings before local training, for FedCE only

            #****#
            
            ########################################### BEGIN Local Training Algorithm ###########################################
            for clientID_training in tqdm(user_id):  # training over the subset
                local = trainer_locals[clientID_training]

                ### local training algorithm 
                if args.alg in ["FedAvg", "q_FedAvg", "FedCE", "FedGA", "Fair_Fed"]:
                    w_local, loss_local = local.train(
                        net=copy.deepcopy(netglob).to(args.device))

                elif args.alg in ["FedISM"]:
                    w_local, loss_local = local.train_GSAM(
                        net=copy.deepcopy(netglob).to(args.device))
                    
                else:
                    raise NotImplementedError
                
                # store every updated model
                w_locals.append(copy.deepcopy(w_local))
                loss_locals.append(copy.deepcopy(loss_local))
                writer.add_scalar(f'train_run{Current_RUN}/loss/client{clientID_training}', loss_local, Current_Round)

            assert clientID_training == user_id[-1] == args.n_clients-1
            assert len(w_locals) == len(dict_len) == args.n_clients
            logging.info(f"------> round: {Current_Round}, Training loss:")
            logging.info(np.array(loss_locals))
            if args.alg == "FedCE":
                model_last_round = copy.deepcopy(w_locals)
            ########################################### END Local Training Algorithm ###########################################

            #****#

            ########################################### BEGIN Global Aggregation Algorithm ###########################################
            if args.alg == "FedGA":
                if Current_Round == 0:
                    aggregation_weights = np.ones(args.n_clients).astype("float") / args.n_clients
                else:
                    d = 0.05
                    aggregation_weights = get_agg_weights_FedGA(d*(args.rounds-Current_Round)/args.rounds, aggregation_weights, loss_results=loss_results, args=args)
                    aggregation_weights = np.clip(aggregation_weights, a_min=0., a_max=None) # very important, due to the BN layer in the model

                w_glob_fl = FedAvg(w_locals, aggregation_weights)
                netglob.load_state_dict(copy.deepcopy(w_glob_fl))

                # in each client
                local_results = np.zeros(args.n_clients).astype("float")
                global_results = np.zeros(args.n_clients).astype("float")
                loss_all, _, _, _ = compute_loss(copy.deepcopy(dataset_train), copy.deepcopy(netglob).to(args.device), args)
                for i in range(args.n_clients):
                    global_results[i] = loss_all[np.array(dict_users[i])].mean()
                    local_dataset = copy.deepcopy(trainer_locals[i].local_dataset)
                    net_temp = copy.deepcopy(netglob)
                    net_temp.load_state_dict(w_locals[i])
                    loss, _, _, _ = compute_loss(local_dataset, net_temp.to(args.device), args)
                    local_results[i] = loss.mean()
                loss_results = global_results - local_results
                loss_results = loss_results - loss_results.mean()


            elif args.alg == "FedCE":
                FedCE_AGG.assemble(local_models_weights=w_locals, current_round=Current_Round, fedce_minus_vals=minus_val, args=args)
                w_glob_fl = FedAvg(w_locals, FedCE_AGG.fedce_coef)
                netglob.load_state_dict(copy.deepcopy(w_glob_fl))
                FedCE_AGG.model.load_state_dict(copy.deepcopy(w_glob_fl))


            elif args.alg == "Fair_Fed":
                if Current_Round == 0:
                    aggregation_weights = np.array(dict_len).astype("float")
                    aggregation_weights = aggregation_weights / aggregation_weights.sum()
                else:
                    local_acc = []
                    last_aggregation_weights = copy.deepcopy(aggregation_weights)
                    for id_Fair_Fed in range(args.n_clients):
                        local_dataset = copy.deepcopy(trainer_locals[id_Fair_Fed].local_dataset)
                        local_acc.append(local_valid(netglob, local_dataset, args))
                    local_acc = np.array(local_acc)
                    num_clients = np.array(dict_len).astype("float")
                    mean_acc = (local_acc * num_clients) / num_clients.sum()
                    delta = np.abs(local_acc - mean_acc)
                    beta_fair = args.q
                    aggregation_weights = last_aggregation_weights - beta_fair * (delta - delta.mean())
                    aggregation_weights = np.clip(aggregation_weights, a_min=0., a_max=None)
                    aggregation_weights = aggregation_weights / aggregation_weights.sum()

                w_glob_fl = FedAvg(w_locals, aggregation_weights)
                netglob.load_state_dict(copy.deepcopy(w_glob_fl))


            elif args.alg == "q_FedAvg":
                w_glob = copy.deepcopy(netglob.state_dict())
                loss_clients = []
                grad_clients = []
                loss_all, _, _, _ = compute_loss(copy.deepcopy(dataset_train), copy.deepcopy(netglob).to(args.device), args)
                for i in range(args.n_clients):
                    loss_client = loss_all[np.array(dict_users[i])]
                    loss_clients.append(loss_client.mean())
                    
                if args.q >= 100:  # inf
                    loss_clients_q = np.asarray(loss_clients)
                    max_index = np.argmax(loss_clients_q)
                    loss_clients_q *= 0.
                    loss_clients_q[max_index] = 1. # Only update the worst client (Agnostic-FL)
                    loss_clients_q = loss_clients_q.astype("float")
                else:
                    loss_clients_q = np.asarray(loss_clients)
                    loss_clients_q = np.float_power(loss_clients_q+1e-10, args.q)
                    loss_clients_q = loss_clients_q / loss_clients_q.sum()
                
                w_glob_fl = FedAvg(w_locals, loss_clients_q)
                netglob.load_state_dict(copy.deepcopy(w_glob_fl))


            elif args.alg == "FedISM": # Ours
                loss_clients = []
                loss_perturb_clients = []
                loss_all, _, _, _ = compute_loss(copy.deepcopy(dataset_train), copy.deepcopy(netglob).to(args.device), args)

                for i in range(args.n_clients):
                    loss = loss_all[np.array(dict_users[i])].mean()
                    loss_clients.append(loss)
                    local_dataset = copy.deepcopy(trainer_locals[i].local_dataset)
                    net_perturb = get_perturb_model(local_dataset, copy.deepcopy(netglob).to(args.device), args)
                    loss_all_p, _, _, _ = compute_loss(local_dataset, copy.deepcopy(net_perturb).to(args.device), args)
                    loss_perturb_clients.append(loss_all_p.mean())
                
                loss_clients = np.array(loss_clients)
                loss_perturb_clients = np.array(loss_perturb_clients)
                sharpness_clients = loss_perturb_clients - loss_clients
                sharpness_clients = np.clip(sharpness_clients, a_min=1e-10, a_max=None)
                logging.info("loss and sharpness of each client:")
                logging.info(loss_clients)
                logging.info(sharpness_clients)

                if Current_Round>0:
                    our_last_weights = copy.deepcopy(agg_weights)

                agg_weights = sharpness_clients / sharpness_clients.sum()
                agg_weights = np.float_power(agg_weights+1e-10, args.q)
                agg_weights = agg_weights / agg_weights.sum()

                if Current_Round>0:
                    agg_weights = args.beta * agg_weights + (1-args.beta) * our_last_weights # Moving Average

                w_glob_fl = FedAvg(w_locals, agg_weights)
                netglob.load_state_dict(copy.deepcopy(w_glob_fl))


            elif args.alg == "FedAvg":
                w_glob_fl = FedAvg(w_locals, dict_len)
                netglob.load_state_dict(copy.deepcopy(w_glob_fl))


            else:
                raise NotImplementedError

            ########################################### END Global Aggregation Algorithm ###########################################



            # global test
            logging.info("\n------------------------------> testing, run: %d, round: %d <------------------------------"  % (Current_RUN, Current_Round))
            if args.dataset in ["ICH_20", "ISIC2019_20"]:
                if dataset_train.datapath in ["gaussian3_14_6_dir1.0", "gaussian3_16_4_dir1.0", "gaussian3_18_2_dir1.0"]:
                    test_severity = [0, 3]
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError
            

            for s in tqdm(test_severity):
                result = globaltest(copy.deepcopy(netglob).to(args.device), test_dataset=datasets_test[s], args=args)
                acc = result["acc"]
                auc = result["auc"]
                bacc = result["bacc"]
                bauc =  result["bauc"]
                f1 = result["f1"]
                bf1 = result["bf1"]
                cm = result["cm"]
                logging.info("-----> s: %d, acc: %.2f, bacc: %.2f, auc: %.2f, bauc: %.2f, f1: %.2f, bf1: %.2f"  % (s, acc*100, bacc*100, auc*100, bauc*100, f1*100, bf1*100))
                logging.info(cm)
                writer.add_scalar(f'test_run{Current_RUN}/acc_s{s}', acc, Current_Round)
                writer.add_scalar(f'test_run{Current_RUN}/bacc_s{s}', bacc, Current_Round)
                writer.add_scalar(f'test_run{Current_RUN}/auc_s{s}', auc, Current_Round)
                writer.add_scalar(f'test_run{Current_RUN}/bauc_s{s}', bauc, Current_Round)
                writer.add_scalar(f'test_run{Current_RUN}/f1_s{s}', f1, Current_Round)
                writer.add_scalar(f'test_run{Current_RUN}/bf1_s{s}', bf1, Current_Round)

                if s == 0:
                    df.iloc[Current_Round, df.columns.get_loc(f"clean_ACC_{Current_RUN}")] = acc
                    df.iloc[Current_Round, df.columns.get_loc(f"clean_BACC_{Current_RUN}")] = bacc
                    df.iloc[Current_Round, df.columns.get_loc(f"clean_AUC_{Current_RUN}")] = auc
                    df.iloc[Current_Round, df.columns.get_loc(f"clean_BAUC_{Current_RUN}")] = bauc
                    df.iloc[Current_Round, df.columns.get_loc(f"clean_F1_{Current_RUN}")] = f1
                    df.iloc[Current_Round, df.columns.get_loc(f"clean_BF1_{Current_RUN}")] = bf1
                else:
                    df.iloc[Current_Round, df.columns.get_loc(f"corrupted_ACC_{Current_RUN}")] = acc
                    df.iloc[Current_Round, df.columns.get_loc(f"corrupted_BACC_{Current_RUN}")] = bacc
                    df.iloc[Current_Round, df.columns.get_loc(f"corrupted_AUC_{Current_RUN}")] = auc
                    df.iloc[Current_Round, df.columns.get_loc(f"corrupted_BAUC_{Current_RUN}")] = bauc
                    df.iloc[Current_Round, df.columns.get_loc(f"corrupted_F1_{Current_RUN}")] = f1
                    df.iloc[Current_Round, df.columns.get_loc(f"corrupted_BF1_{Current_RUN}")] = bf1

            logging.info('\n')

            # save model
            torch.save(netglob.state_dict(),  models_dir + '/current_model.pth')
            if Current_Round >= args.rounds-5:
                torch.save(netglob.state_dict(),  models_dir + f'/best_model_{Current_RUN}_{Current_Round}.pth')
            df.to_csv(logs_dir+"/result.csv", index=False)

        # This run ends
        assert Current_Round == args.rounds - 1
        agg_weights = None
        our_last_weights = None
        aggregation_weights = None
        last_aggregation_weights = None

        

    # All runs end
    if args.dataset in ["ICH_20", "ISIC2019_20"]:
        last_epoch = -5
        bacc_clean = np.array(df.loc[:, ["clean_BACC_0", "clean_BACC_1", "clean_BACC_2"]])[last_epoch:, :]
        bauc_clean = np.array(df.loc[:, ["clean_BAUC_0", "clean_BAUC_1", "clean_BAUC_2"]])[last_epoch:, :]
        bacc_corrupted = np.array(df.loc[:, ["corrupted_BACC_0", "corrupted_BACC_1", "corrupted_BACC_2"]])[last_epoch:, :]
        bauc_corrupted = np.array(df.loc[:, ["corrupted_BAUC_0", "corrupted_BAUC_1", "corrupted_BAUC_2"]])[last_epoch:, :]

        acc_clean_mean, acc_clean_std = bacc_clean.mean(), bacc_clean.std()
        auc_clean_mean, auc_clean_std = bauc_clean.mean(), bauc_clean.std()
        acc_corrupted_mean, acc_corrupted_std = bacc_corrupted.mean(), bacc_corrupted.std()
        auc_corrupted_mean, auc_corrupted_std = bauc_corrupted.mean(), bauc_corrupted.std()

        acc_avg = (bacc_clean + bacc_corrupted) / 2.
        auc_avg = (bauc_clean + bauc_corrupted) / 2.
        acc_avg_mean, acc_avg_std = acc_avg.mean(), acc_avg.std()
        auc_avg_mean, auc_avg_std = auc_avg.mean(), auc_avg.std()

        logging.info("Clean data ---> ACC: %.2f+-%.2f, AUC: %.2f+-%.2f" % (acc_clean_mean*100, acc_clean_std*100, auc_clean_mean*100, auc_clean_std*100))
        logging.info("Corrupted data ---> ACC: %.2f+-%.2f, AUC: %.2f+-%.2f" % (acc_corrupted_mean*100, acc_corrupted_std*100, auc_corrupted_mean*100, auc_corrupted_std*100))
        logging.info("Avg data ---> ACC: %.2f+-%.2f, AUC: %.2f+-%.2f" % (acc_avg_mean*100, acc_avg_std*100, auc_avg_mean*100, auc_avg_std*100))
        torch.cuda.empty_cache()

    else:
        raise NotImplementedError
