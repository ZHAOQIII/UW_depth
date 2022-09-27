import os
import argparse
from solver import Solver
from data_loader import get_loader
from torch.backends import cudnn


def str2bool(v):
    return v.lower() in ('true')



def main(config):
    # For fast training.
    cudnn.benchmark = True

    # Create directories if not exist.
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)
    if not os.path.exists(config.model_save_dir):
        os.makedirs(config.model_save_dir)
    if not os.path.exists(config.sample_dir):
        os.makedirs(config.sample_dir)
    if not os.path.exists(config.result_dir):
        os.makedirs(config.result_dir)
    if not os.path.exists(config.images_dir_train):
        os.makedirs(config.images_dir_train)
    if not os.path.exists(config.images_save_dir):
        os.makedirs(config.images_save_dir)
    if not os.path.exists(config.images_dir_pixpix):
        os.makedirs(config.images_dir_pixpix)


    data_loaderA,data_loaderB = get_loader(config.image_dir, config.data_crop_size,
                             config.image_size, config.batch_size,
                             config.mode, config.num_workers)

    solver = Solver(data_loaderA,data_loaderB, config)

    if config.mode == 'train':
        solver.train()
    elif config.mode == 'test':
        solver.test()
    elif config.mode == 'test_pix2pix':
        solver.test_pix2pix()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model configuration.
    parser.add_argument('--c_dim', type=int, default=3, help='dimension of domain labels (1st dataset)')
    parser.add_argument('--data_crop_size', type=int, default=128, help='crop size for the RaFD dataset')
    parser.add_argument('--image_size', type=int, default=128, help='image resolution')
    parser.add_argument('--g_conv_dim', type=int, default=64, help='number of conv filters in the first layer of G')
    parser.add_argument('--d_conv_dim', type=int, default=64, help='number of conv filters in the first layer of D')
    parser.add_argument('--g_repeat_num', type=int, default=6, help='number of residual blocks in G')
    parser.add_argument('--d_repeat_num', type=int, default=6, help='number of strided conv layers in D')
    parser.add_argument('--lambda_cls', type=float, default=5, help='weight for domain classification loss')
    parser.add_argument('--lambda_rec', type=float, default=10, help='weight for reconstruction loss')
    parser.add_argument('--lambda_ls', type=float, default=1, help='weight for gradient penalty')
    parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
    # Training configuration.
    parser.add_argument('--batch_size', type=int, default=1, help='mini-batch size')
    parser.add_argument('--num_iters', type=int, default=100000, help='number of total iterations for training D')
    parser.add_argument('--num_iters_decay', type=int, default=50000, help='number of iterations for decaying lr')
    parser.add_argument('--g_lr', type=float, default=0.00005, help='learning rate for G')
    parser.add_argument('--d_lr', type=float, default=0.0002, help='learning rate for D')
    parser.add_argument('--lr_pix2pix', type=float, default=0.00005, help='initial learning rate for adam')
    parser.add_argument('--d_lr_pix2pix', type=float, default=0.0002, help='initial learning rate for adam')
    parser.add_argument('--n_critic', type=int, default=5, help='number of D updates per each G update')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
    parser.add_argument('--resume_iters', type=int, default=0, help='resume training from this step')
    parser.set_defaults(pool_size=0, no_lsgan=True)
    parser.add_argument('--init_type', type=str, default='normal',
                        help='network initialization [normal | xavier | kaiming | orthogonal]')
    parser.add_argument('--init_gain', type=float, default=0.02,
                        help='scaling factor for normal, xavier and orthogonal.')

    # Test configuration.
    parser.add_argument('--test_iters', type=int, default=100000, help='test model from this step')
    parser.add_argument('--model', type=str, default='pix2pix',
                        help='chooses which model to use. cycle_gan, pix2pix, test')
    parser.add_argument('--gpu_ids', type=str, default='1', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    # Miscellaneous.
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--mode', type=str, default='test_pix2pix', choices=['train', 'test','test_pix2pix'])
    parser.add_argument('--use_tensorboard', type=str2bool, default=True)
    # Directories.
    parser.add_argument('--image_dir', type=str, default='data/')
    parser.add_argument('--log_dir', type=str, default='result/logs')
    parser.add_argument('--model_save_dir', type=str, default='result/models')
    parser.add_argument('--sample_dir', type=str, default='result/samples')
    parser.add_argument('--result_dir', type=str, default='result/results')
    parser.add_argument('--result_pix2pix_dir', type=str, default='result/results_pix2pix')
    parser.add_argument('--images_dir_train', type=str, default='result/train')
    parser.add_argument('--images_dir_pixpix', type=str, default='result/pix2pix')
    parser.add_argument('--images_save_dir', type=str, default='result/depths')
    parser.add_argument('--channels', type=int, default=6, help='number of image channels')
    # Step size.
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--sample_step', type=int, default=1000)
    parser.add_argument('--model_save_step', type=int, default=10000)
    parser.add_argument('--lr_update_step', type=int, default=1000)

    config = parser.parse_args()
    print(config)
    main(config)

