import argparse
import sys

# argv = sys.argv
# dataset = argv[1]


def cora_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="cora")
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--p', type=float, default="0.02")
    parser.add_argument('--epochtimes', type=int, default=20)  
    parser.add_argument('--clusters', type=int, default=100) 
    parser.add_argument('--cuda_id', type=int, default=0)
    parser.add_argument('--alpha', type=float, default=0.9)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--w_ib', type=float, default=0.05)
    parser.add_argument('--IB', type=str, default="IB")
    parser.add_argument('--attack_iters', type=int, default=1)
    parser.add_argument('--step_size', type=float, default=0.02)
    parser.add_argument('--mask_type', type=str, default='adv')
    parser.add_argument('--prompt_type', type=str, default='dynamic')
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--warm_up', type=int, default=0)
    parser.add_argument('--club_opt_lr', type=float, default=1e-3)
    parser.add_argument('--hclub', type=int, default=64)
    parser.add_argument('--lclub', type=int, default=1)
    parser.add_argument('--lp', type=float, default="0.2")
    parser.add_argument('--club_epoch', type=int, default=5)    
    args, _ = parser.parse_known_args()
    return args


def citeseer_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="citeseer")
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--p', type=float, default="0.02")
    parser.add_argument('--epochtimes', type=int, default=20)  
    parser.add_argument('--clusters', type=int, default=100) 
    parser.add_argument('--cuda_id', type=int, default=0)
    parser.add_argument('--alpha', type=float, default=0.95)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--w_ib', type=float, default=0.1)
    parser.add_argument('--IB', type=str, default="IB")
    parser.add_argument('--attack_iters', type=int, default=2)
    parser.add_argument('--step_size', type=float, default=0.05)
    parser.add_argument('--mask_type', type=str, default='adv')
    parser.add_argument('--prompt_type', type=str, default='dynamic')
    parser.add_argument('--k', type=int, default=20)
    parser.add_argument('--warm_up', type=int, default=0)
    parser.add_argument('--club_opt_lr', type=float, default=5e-3)
    parser.add_argument('--hclub', type=int, default=64)
    parser.add_argument('--lclub', type=int, default=1)
    parser.add_argument('--lp', type=float, default="0.2")
    parser.add_argument('--club_epoch', type=int, default=1)    
    args, _ = parser.parse_known_args()
    return args


def photo_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="A-photo")
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--cuda_id', type=int, default=0)
    parser.add_argument('--alpha', type=float, default=0.9)
    parser.add_argument('--step_size', type=float, default=0.01)
    parser.add_argument('--attack_iters', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=250)
    parser.add_argument('--IB', type=str, default="IB")
    parser.add_argument('--p', type=float, default="0.02")
    parser.add_argument('--lp', type=float, default="0.3")
    parser.add_argument('--mask_type', type=str, default='adv')
    parser.add_argument('--prompt_type', type=str, default='x')
    parser.add_argument('--warm_up', type=int, default=0)
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--w_ib', type=float, default=0.03)
    parser.add_argument('--hclub', type=int, default=64)
    parser.add_argument('--lclub', type=int, default=1)
    parser.add_argument('--club_opt_lr', type=float, default=0.0001)
    parser.add_argument('--club_epoch', type=int, default=1)
    args, _ = parser.parse_known_args()
    return args
    
def computer_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="A-computer")
    parser.add_argument('--dropout', type=float, default=0.8)
    parser.add_argument('--cuda_id', type=int, default=0)
    parser.add_argument('--alpha', type=float, default=0.9)
    parser.add_argument('--step_size', type=float, default=0.01)
    parser.add_argument('--attack_iters', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--w_ib', type=float, default=0.03)
    parser.add_argument('--IB', type=str, default="IB")
    parser.add_argument('--p', type=float, default="0.02")
    parser.add_argument('--mask_type', type=str, default='adv')
    parser.add_argument('--prompt_type', type=str, default='dynamic')
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--warm_up', type=int, default=30)
    parser.add_argument('--club_opt_lr', type=float, default=5e-3)
    parser.add_argument('--hclub', type=int, default=64)
    parser.add_argument('--lclub', type=int, default=1)
    parser.add_argument('--lp', type=float, default="0.2")
    parser.add_argument('--club_epoch', type=int, default=1)    
    args, _ = parser.parse_known_args()
    return args



def set_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="cora")
    args, _ = parser.parse_known_args()
    dataset = args.dataset
    
    if dataset == "cora":
        args = cora_params()
    elif dataset == "citeseer":
        args = citeseer_params()
    elif dataset == "A-photo":
        args = photo_params()
    elif dataset == "A-computer":
        args = computer_params()
    return args
    