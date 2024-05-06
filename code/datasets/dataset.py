import numpy as np
import os
import torchvision.transforms as transforms

from .all_datasets import IchDataset, ISIC2019Dataset
from utils.sampling import iid_sampling, non_iid_dirichlet_sampling


def get_dataset(args):
    
    if args.dataset == "ICH_20":
        args.n_classes = 5
        args.n_clients = 20
        args.client_num = args.n_clients
        args.input_channel = 3

        # Data transforms
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        train_transform = transforms.Compose([
            # transforms.Resize((224, 224)),
            transforms.RandomAffine(degrees=10, translate=(0.02, 0.02)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        val_transform = transforms.Compose([
            # transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize,
        ])

        if args.noise == 0:
            noise_type = "clean"
            train_path = "clean"

        elif args.noise == 1:
            noise_type = "gaussian"
            train_path = "gaussian3_14_6_dir1.0"
            assert args.alpha_dirichlet == 1.0
            assert args.corrupted_num == 6

        elif args.noise == 2:
            noise_type = "gaussian"
            train_path = "gaussian3_16_4_dir1.0"
            assert args.alpha_dirichlet == 1.0
            assert args.corrupted_num == 4

        elif args.noise == 3:
            noise_type = "gaussian"
            train_path = "gaussian3_18_2_dir1.0"
            assert args.alpha_dirichlet == 1.0
            assert args.corrupted_num == 2

        else:
            raise

        train_dataset = IchDataset(datapath=train_path, mode="train", transform=train_transform, root="ICH_20")
        test_dataset_clean = IchDataset(datapath="clean", mode="test", transform=val_transform, root="ICH_20")
        assert len(set(train_dataset.image_list).intersection(set(test_dataset_clean.image_list))) == 0

        if noise_type != "clean":
            test_dataset_noisy1 = IchDataset(datapath=noise_type+'_1', mode="test", transform=val_transform, root="ICH_20")
            test_dataset_noisy2 = IchDataset(datapath=noise_type+'_2', mode="test", transform=val_transform, root="ICH_20")
            test_dataset_noisy3 = IchDataset(datapath=noise_type+'_3', mode="test", transform=val_transform, root="ICH_20")
            test_dataset_noisy4 = IchDataset(datapath=noise_type+'_4', mode="test", transform=val_transform, root="ICH_20")
            test_dataset_noisy5 = IchDataset(datapath=noise_type+'_5', mode="test", transform=val_transform, root="ICH_20")
            test_datasets = [test_dataset_clean, test_dataset_noisy1, test_dataset_noisy2, test_dataset_noisy3, test_dataset_noisy4, test_dataset_noisy5]
        else:
            test_datasets = [test_dataset_clean] * 6



    elif args.dataset == "ISIC2019_20":
        args.n_classes = 8
        args.n_clients = 20
        args.client_num = args.n_clients
        args.input_channel = 3

        # Data transforms
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        train_transform = transforms.Compose([
            # transforms.Resize((224, 224)),
            transforms.RandomAffine(degrees=10, translate=(0.02, 0.02)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        val_transform = transforms.Compose([
            # transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize,
        ])

        if args.noise == 0:
            noise_type = "clean"
            train_path = "clean"

        elif args.noise == 1:
            noise_type = "gaussian"
            train_path = "gaussian3_16_4_dir1.0"
            assert args.alpha_dirichlet == 1.0
            assert args.corrupted_num == 4

        else:
            raise NotImplementedError

        train_dataset = ISIC2019Dataset(datapath=train_path, mode="train", transform=train_transform, root="ISIC2019_20")
        test_dataset_clean = ISIC2019Dataset(datapath="clean", mode="test", transform=val_transform, root="ISIC2019_20")
        assert len(set(train_dataset.image_list).intersection(set(test_dataset_clean.image_list))) == 0

        if noise_type != "clean":
            test_dataset_noisy1 = ISIC2019Dataset(datapath=noise_type+'_1', mode="test", transform=val_transform, root="ISIC2019_20")
            test_dataset_noisy2 = ISIC2019Dataset(datapath=noise_type+'_2', mode="test", transform=val_transform, root="ISIC2019_20")
            test_dataset_noisy3 = ISIC2019Dataset(datapath=noise_type+'_3', mode="test", transform=val_transform, root="ISIC2019_20")
            test_dataset_noisy4 = ISIC2019Dataset(datapath=noise_type+'_4', mode="test", transform=val_transform, root="ISIC2019_20")
            test_dataset_noisy5 = ISIC2019Dataset(datapath=noise_type+'_5', mode="test", transform=val_transform, root="ISIC2019_20")
            test_datasets = [test_dataset_clean, test_dataset_noisy1, test_dataset_noisy2, test_dataset_noisy3, test_dataset_noisy4, test_dataset_noisy5]
        else:
            test_datasets = [test_dataset_clean] * 6
    
    else:
        exit("Error: Unrecognized Dataset")


    # Sampling for Clients
    n_train = len(train_dataset)
    y_train = np.array(train_dataset.targets)
    assert n_train == len(y_train)

    if args.iid:
        if os.path.exists(f"data/{args.dataset}/dict_users_iid.npy"):
            dict_users = np.load(f"data/{args.dataset}/dict_users_iid.npy", allow_pickle=True).item()
        else:
            dict_users = iid_sampling(n_train, args.n_clients, seed=2024)
            np.save(f"data/{args.dataset}/dict_users_iid.npy", dict_users, allow_pickle=True)
    else:
        if os.path.exists(f"data/{args.dataset}/dict_users_noniid_dir{args.alpha_dirichlet}_CorruptedNum{args.corrupted_num}.npy"):
            dict_users = np.load(f"data/{args.dataset}/dict_users_noniid_dir{args.alpha_dirichlet}_CorruptedNum{args.corrupted_num}.npy", allow_pickle=True).item()
        else:
            dict_users = non_iid_dirichlet_sampling(y_train, args.n_classes, p=1.0, num_users=args.n_clients, seed=2024, alpha_dirichlet=args.alpha_dirichlet, corrupted_num=args.corrupted_num)
            np.save(f"data/{args.dataset}/dict_users_noniid_dir{args.alpha_dirichlet}_CorruptedNum{args.corrupted_num}.npy", dict_users, allow_pickle=True)

    # Check 'dict_users' for Federated Learning
    assert len(dict_users.keys()) == args.n_clients
    items = []
    for key in dict_users.keys():
        items += list(dict_users[key])
    assert len(items) == len(set(items)) == len(y_train), f"{len(items)}, {len(set(items))}, {len(y_train)}"
    assert set(items) == set(list(range(len(items))))

    print("\n########################")
    print("######### Dataset is ready #########")
    print("########################\n")

    return train_dataset, test_datasets, dict_users



