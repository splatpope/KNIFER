import argparse
# TODO : command line utility to supersede the GUI
# kinda done, just needs a parser really
import torch
from training import TrainingManager, KNIFER_ARCHS

from metrics import FID

parser = argparse.ArgumentParser(
    description="Experimentation platform for KNIFER GAN", 
    epilog="Use in interactive mode. You get a 'tm' variable which is an instance of TrainingManager.",
)
# Experiment parameters
parser.add_argument('--arch', choices=KNIFER_ARCHS.keys(), help="GAN architecture", required=True)
parser.add_argument('--dataset', type=str, help="Path to dataset to load as an ImageFolder", required=True)
parser.add_argument('--experiment', type=str, help="Experiment name. Determines output folders", required=True)
parser.add_argument('--num_workers', type=int, default=0, help="Amount of CPU workers to use for data loading")
parser.add_argument('-p', '--parallel', action="store_true", help="Enable distributed GPU")
parser.add_argument('--img_size', type=int, default=64, help="Size of images involved")
parser.add_argument('--batch_size', default=16, type=int, help='Batch size')
parser.add_argument('--latent_size', default=100, type=int, help='Latent space dimensionality')
parser.add_argument('--n_features', type=int, help="Base amount of features for both models.")
# Common training parameters
parser.add_argument('--lr_g', default=1e-4, type=float, metavar="learning_rate_G", help="Generator learning rate")
parser.add_argument('--lr_d', default=4e-4, type=float, metavar="learning_rate_D", help="Discriminator learning rate")
parser.add_argument('--b1', default=0.5, type=float, help="First order ADAM momentum")
parser.add_argument('--b2', default=0.999, type=float, help="Second order ADAM momentum")
# WGAN related parameters
parser.add_argument('--critic_iters', default=5, type=int, help="(WGAN) Discriminator update factor")
parser.add_argument('--lambda_gp', default=10, type=int, help="(WGAN+GP) Lambda factor for gradient penalty")
# SAGAN related parameters
parser.add_argument('--attn_spots', type=int, nargs="*", help="(SAGAN) Middle layers after which a self attention module should be added")


def run_manager_n_times(manager: TrainingManager, n: int, n_epochs: int, save_step: int = 1, viz_step:int = 1):
    experiment = manager.experiment_name
    saves_dest = "./savestates/" + experiment
    viz_dest = "./viz/" + experiment
    for i in range(n):
        manager.simple_train_loop(n_epochs)
        if (i+1) % save_step == 0:
            manager.save(saves_dest)
        if (i+1) % viz_step == 0:
            manager.synthetize_viz(viz_dest)

def build_manager(args):
    params = dict()
    params.update({
        "arch": args.arch,
        "img_size": args.img_size,
        "batch_size": args.batch_size,
        "latent_size": args.latent_size,
        "lr_g": args.lr_g,
        "lr_d": args.lr_d,
        "b1": args.b1,
        "b2": args.b2,
        "critic_iters": args.critic_iters,
        "lambda_gp": args.lambda_gp,
    })

    if args.attn_spots:
        params.update({
            "attn_spots": args.attn_spots,
        })

    if args.n_features:
        params.update({
            "features": args.n_features,
        })
    else:
        params.update({
            "features": args.img_size,
        })

    manager = TrainingManager(args.experiment, debug=True, parallel=args.parallel)
    manager.set_dataset_folder(args.dataset)
    manager.set_trainer(params, num_workers=args.num_workers)
    return manager

if __name__ == '__main__':
    args = parser.parse_args()
    print(args)

    tm = build_manager(args)
