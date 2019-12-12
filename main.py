import argparse
import torch
import numpy as np
from kl_mnist import *

def check_args(args):
    # Checks to make sure that all args inputted are of correct type / value. Returns True if so, else returns False

    # Check norm
    if args.norm == '2':
        args.norm = 2
    elif args.norm == 'inf':
        args.norm = np.inf
    else:
        print("Error: Norm must either be '2' or 'inf'!")
        return False

    return True


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_deltas", action='store_true', help='If set, evaluates the deltas across the MNIST dataset')
    parser.add_argument("--eval_psis", action='store_true',
                        help='If set, evaluates the psis across the MNIST dataset (assumes deltas have already been evaluated')
    parser.add_argument("--run_attack", action='store_true',
                        help='If set, Will run the specified attack with the --attack arg')
    parser.add_argument("--attack", type=str, default='noise',
                        help="Type of attack. Can be 'noise', 'fgm', or 'pixel'")
    parser.add_argument("--eps", type=float, default=0,
                        help='Strength of attack perturbation')
    parser.add_argument("--norm", type=str, default='2',
                        help='Type of norm if using fgm attack')
    parser.add_argument("--num_images", type=int, default=1000,
                        help='Number of images to evaluate on test dataset')
    parser.add_argument("--num_delta_SD", type=int, default=1.8,
                        help='sigma_detect threshold value for detecting adversarial / anomalous examples')
    parser.add_argument("--num_psi_SD", type=int, default=0.75,
                        help='Sigma_detect threshold value for detecting adversarial / anomalous examples')
    parser.add_argument("--num_false_threshold", type=int, default=1,
                        help='Min number of delta values for a given image that must exceed sigma_detect to be detected')
    parser.add_argument("--print_every", type=int, default=100,
                        help='Min number of delta values for a given image that must exceed sigma_detect to be detected')
    parser.add_argument("--stop_idx", nargs='*', default=[],
                        help='specific indexes within the test dataset to pause at with detailed printouts')
    parser.add_argument("--check_all", type=bool, action='store_true',
                        help='If set, will pause at every image within the test dataset')
    parser.add_argument("--use_printouts", type=bool, action='store_true',
                        help='If set, will pause at every anomalous / successful adversarial image in the dataset')

    cuda = True if torch.cuda.is_available() else False
    device = 'cuda' if cuda else 'cpu'

    ####################################
    ####################################
    ### TODO: NEED TO LOAD YOUR SPECFIC MODEL HERE BELOW! ###
    '''
    model (nn.Module): The model to be used for testing. It is expected to have the following attributes:
        .name (str): Name of the model
        .encoder (nn.Module): Neural network with callable, implemented forward function -- must expect inputs of size 784
        .classifier (nn.Module): Neural network with callable, implemented forward function -- must expect inputs of size 784
        .z_prior (torch.nn.Parameter): length-2 tuple that holds the z_prior mean(s) in [0] and z_prior variances in [1]
    '''
    model = None
    #####################################
    #####################################

    args = parser.parse_args()

    if check_args(args):
        if args.eval_deltas:
            determine_deltas_mnist(model, device)
        if args.eval_psis:
            evaluate_attack_mnist(model, device, args.attack, eps=0, get_psi=True)
        if args.run_attack:
            evaluate_attack_mnist(model=model,
                                  device=device,
                                  attack=args.attack,
                                  eps=args.eps,
                                  norm=args.norm,
                                  num_SD=args.num_delta_SD,
                                  num_summed_SD=args.num_psi_SD,
                                  num_false=args.num_false_threshold,
                                  num_imgs=args.num_imgs,
                                  print_every=args.print_every,
                                  stop_idx=args.stop_idx,
                                  CheckAll=args.check_all,
                                  use_printouts=args.use_printouts)
