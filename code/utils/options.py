import argparse

def args_parser():
    parser = argparse.ArgumentParser()

    # system setting
    parser.add_argument('--deterministic', type=int,  default=1,
                        help='whether use deterministic training')
    parser.add_argument('--seed', type=int,  default=0, help='random seed')
    parser.add_argument('--gpu', type=str,  default='1', help='GPU to use')

    # basic setting
    parser.add_argument('--alg', type=str,
                        default='Fed', help='algorithm name')
    parser.add_argument('--runs', type=int,
                        default=3, help='run times')
    parser.add_argument('--dataset', type=str,
                        default='ICH', help='dataset name')
    parser.add_argument('--model', type=str,
                        default='Resnet18', help='model name')
    parser.add_argument('--pretrained', action="store_false", 
                        help='use pretrained model')
    parser.add_argument('--batch_size', type=int,
                        default=32, help='batch_size per gpu')
    parser.add_argument('--base_lr', type=float,  default=3e-4,
                        help='base learning rate')
    parser.add_argument('--noise', type=int,
                        default=0, help='noise strength')
    parser.add_argument('--attempts', type=int,
                        default=0, help='times of attempts')
    
    # for FL
    parser.add_argument('--iid', action="store_true", help="i.i.d. or non-i.i.d.")
    parser.add_argument('--alpha_dirichlet', type=float,
                        default=1.0, help='parameter for non-iid')
    parser.add_argument('--local_ep', type=int, default=1, help='local epoch')
    parser.add_argument('--rounds', type=int,  default=300, help='rounds')
    parser.add_argument('--beta', type=float,  default=0.5, help='beta for EMA')
    
    parser.add_argument('--corrupted_num', type=int,  default=10, help='corrupted clients num')


    # for GSAM
    parser.add_argument('--gsam_rho', type=float,  default=0.05, help='rho for GSAM')
    parser.add_argument('--gsam_alpha', type=float,  default=0.1, help='alpha for GSAM')
    parser.add_argument('--p_rho_curve', type=float, default=0.5, help="p_rho_curve")


    # for FedCE
    parser.add_argument('--fedce_mode', type=str, default="times", help="mode of FedCE")

    # for q-fedavg
    parser.add_argument('--q', type=float, default=1., help="q of q-fedavg")


    args = parser.parse_args()
    return args