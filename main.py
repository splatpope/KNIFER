import argparse
import json
import sys

# TODO : command line utility to supersede the GUI
# kinda done, just needs a parser really
from training import TrainingManager

def run_manager_n_times(manager: TrainingManager, n: int, n_epochs: int, 
        save_step: int = sys.maxsize, 
        viz_step: int = sys.maxsize, 
        FID_step: int = sys.maxsize,
    ):
    for i in range(n):
        manager.simple_train_loop(n_epochs)
        if (i+1) % save_step == 0:
            manager.save()
        if (i+1) % viz_step == 0:
            manager.synth_fixed()
        if (i+1) % FID_step == 0:
            manager.produce_FID()

def build_manager(args, params):
    manager = TrainingManager(args.experiment, args.output, debug=True, parallel=args.parallel)
    manager.set_dataset_folder(args.dataset)
    manager.set_trainer(params, num_workers=args.num_workers)
    return manager

def arch_from_cfg(path):
    with open(path, 'r') as f:
        return json.load(f)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Experimentation platform for KNIFER GAN", 
        epilog="Use in interactive mode. You get a 'tm' variable which is an instance of TrainingManager.",
    )
# Experiment parameters
    parser.add_argument("-c", "--config_file", type=str, help="Path to config file describing arch to use", required=True)
    parser.add_argument('--dataset', type=str, help="Path to dataset to load as an ImageFolder", required=True)
    parser.add_argument("--output", default="./experiments", type=str, help="Path to which output is saved (i.e. experiment results : runs, checkpoints, samples")
    parser.add_argument('--experiment', type=str, help="Experiment name. Determines output folders", required=True)
    parser.add_argument('--num_workers', type=int, default=0, help="Amount of CPU workers to use for data loading")
    parser.add_argument('-p', '--parallel', action="store_true", help="Enable distributed GPU")
    parser.add_argument('--batch_size', default=16, type=int, help='Batch size')
    parser.add_argument('--latent_size', default=100, type=int, help='Latent space dimensionality')
    # Common training parameters
    parser.add_argument('--lr_g', default=1e-4, type=float, metavar="learning_rate_G", help="Generator learning rate")
    parser.add_argument('--lr_d', default=4e-4, type=float, metavar="learning_rate_D", help="Discriminator learning rate")
    parser.add_argument('--b1', default=0.5, type=float, help="First order ADAM momentum")
    parser.add_argument('--b2', default=0.999, type=float, help="Second order ADAM momentum")
    # WGAN related parameters
    parser.add_argument('--critic_iters', default=5, type=int, help="(WGAN) Discriminator update factor")
    parser.add_argument('--lambda_gp', default=10, type=int, help="(WGAN+GP) Lambda factor for gradient penalty")
    parser.add_argument('--use_sr', action="store_true", help="Use Spectral Regularization (warning : makes liberal use of SVD")

    args = parser.parse_args()
    p = {
        "batch_size": args.batch_size,
        "latent_size": args.latent_size,
        "lr_g": args.lr_g,
        "lr_d": args.lr_d,
        "b1": args.b1,
        "b2": args.b2,
        "critic_iters": args.critic_iters,
        "lambda_gp": args.lambda_gp,
        "use_sr": args.use_sr,
    }

    print(args)
    p.update(arch_from_cfg(args.config_file))

    tm = build_manager(args, p)
    print(p)
