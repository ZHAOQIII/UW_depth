from model import Generator
from model import NLayerDiscriminator,NLayerDiscriminator1
import torch
import torch.nn.functional as F
import numpy as np
import os
import torch.nn as nn
import os.path
from loss import GANLoss,VGGLoss1
import datetime
from fcn import fcdense56_nodrop
from loss import PerceptualLoss
from torchvision.utils import save_image
import time

from __init__ import init_net
depth_dir="./result/water_depth"

class Solver(object):
    """Solver for training and testing StarGAN."""

    def __init__(self, data_loaderA,data_loaderB,config):
        """Initialize configurations."""

        # Data loader.
        self.data_loaderA = data_loaderA
        self.data_loaderB = data_loaderB
        self.channels = config.channels

        # Model configurations.
        self.c_dim = config.c_dim
        self.image_size = config.image_size
        self.g_conv_dim = config.g_conv_dim
        self.d_conv_dim = config.d_conv_dim
        self.g_repeat_num = config.g_repeat_num
        self.d_repeat_num = config.d_repeat_num
        self.lambda_cls = config.lambda_cls
        self.lambda_rec = config.lambda_rec
        self.lambda_ls = config.lambda_ls
        self.criterionL1 = torch.nn.L1Loss()
        self.lambda_L1=config.lambda_L1

        # Training configurations.
        self.batch_size = config.batch_size
        self.num_iters = config.num_iters
        self.num_iters_decay = config.num_iters_decay
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.n_critic = config.n_critic
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.resume_iters = config.resume_iters
        self.image_dir=config.image_dir
        # Test configurations.
        self.test_iters = config.test_iters
        self.gpu_ids=config.gpu_ids
        # Miscellaneous.
        self.use_tensorboard = config.use_tensorboard
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        # Directories.
        self.log_dir = config.log_dir
        self.sample_dir = config.sample_dir
        self.model_save_dir = config.model_save_dir
        self.result_dir = config.result_dir
        self.images_dir_train=config.images_dir_train
        self.images_save_dir=config.images_save_dir
        self.images_dir_pixpix = config.images_dir_pixpix
        self.result_pix2pix_dir = config.result_pix2pix_dir
        self.init_type = config.init_type
        self.init_gain = config.init_gain
        # Step size.
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step
        self.lr_update_step = config.lr_update_step
        self.lr_pix2pix = config.lr_pix2pix
        self.d_lr_pix2pix=config.d_lr_pix2pix        # Build the model and tensorboard.
        self.build_model()
        if self.use_tensorboard:
            self.build_tensorboard()
        self.model=config.model
        self.isTrain = True
        self.criterionGAN = GANLoss(use_lsgan=True).to(self.device)

        self.pool_size = config.pool_size
        self.per = PerceptualLoss(nn.L1Loss())
        self.vgg1=VGGLoss1(self.gpu_ids)

    def build_model(self):
        """Create a generator and a discriminator."""
        self.G = Generator(self.g_conv_dim, self.c_dim,self.g_repeat_num)
        self.D = NLayerDiscriminator(self.c_dim)
        self.G_pix2pix = fcdense56_nodrop()
        self.D_pix2pix = NLayerDiscriminator1()


        # Initialize generator and discriminator
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, [self.beta1, self.beta2])
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), self.d_lr, [self.beta1, self.beta2])
        self.g_pix2pix_optimizer = torch.optim.Adam(self.G_pix2pix.parameters(), self.lr_pix2pix, [self.beta1, self.beta2])
        self.d_pix2pix_optimizer = torch.optim.Adam(self.D_pix2pix.parameters(), self.d_lr_pix2pix, [self.beta1, self.beta2])


        self.G.to(self.device)
        self.D.to(self.device)

        self.G_pix2pix.to(self.device)
        self.D_pix2pix.to(self.device)


    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

    def restore_model(self, resume_iters):
        """Restore the trained generator and discriminator."""
        print('Loading the trained models from step {}...'.format(resume_iters))
        G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(resume_iters))
        D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(resume_iters))
        self.G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
        self.D.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))
    def restore_model_pix2pix(self, resume_iters):
        """Restore the trained generator and discriminator."""
        print('Loading the trained models from step {}...'.format(resume_iters))
        G_pix2pix_path = os.path.join(self.images_dir_pixpix, '{}-G.pth'.format(resume_iters))
        D_pix2pix_path = os.path.join(self.images_dir_pixpix, '{}-D.pth'.format(resume_iters))
        self.G_pix2pix.load_state_dict(torch.load(G_pix2pix_path, map_location=lambda storage, loc: storage))
        self.D_pix2pix.load_state_dict(torch.load(D_pix2pix_path, map_location=lambda storage, loc: storage))
    def restore_model_all(self, resume_iters):
        """Restore the trained generator and discriminator."""
        print('Loading the trained models from step {}...'.format(resume_iters))
        G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(resume_iters))
        D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(resume_iters))
        G_pix2pix_path = os.path.join(self.images_dir_pixpix, '{}-G.pth'.format(resume_iters))
        D_pix2pix_path = os.path.join(self.images_dir_pixpix, '{}-D.pth'.format(resume_iters))
        self.G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
        self.D.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))
        self.G_pix2pix.load_state_dict(torch.load(G_pix2pix_path, map_location=lambda storage, loc: storage))
        self.D_pix2pix.load_state_dict(torch.load(D_pix2pix_path, map_location=lambda storage, loc: storage))
    def build_tensorboard(self):
        """Build a tensorboard logger."""
        from logger import Logger
        self.logger = Logger(self.log_dir)

    def update_lr(self, g_lr, d_lr,lr_pix2pix,d_lr_pix2pix):
        """Decay learning rates of the generator and discriminator."""
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = g_lr
        for param_group in self.d_optimizer.param_groups:
            param_group['lr'] = d_lr
        for param_group in self.g_pix2pix_optimizer.param_groups:
            param_group['lr'] = lr_pix2pix
        for param_group in self.d_pix2pix_optimizer.param_groups:
            param_group['lr'] = d_lr_pix2pix

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()
    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def denorm(self, x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out.clamp_(0, 1)


    def label2onehot(self, labels, dim):
        """Convert label indices to one-hot vectors."""
        batch_size = labels.size(0)
        out = torch.zeros(batch_size, dim)
        out[np.arange(batch_size), labels.long()] = 1
        return out

    def create_labels(self, c_org, c_dim=2):
        """Generate target domain labels for debugging and testing."""
        # Get hair color indices.
        c_trg_list = []
        for i in range(c_dim):
            c_trg = self.label2onehot(torch.ones(c_org.size(0))*i, c_dim)
            c_trg_list.append(c_trg.to(self.device))
        return c_trg_list

    def classification_loss(self, logit, target):
        """Compute binary or softmax cross entropy loss."""
        return F.cross_entropy(logit, target)
    def backward_D(self):
        # Fake
        pred_fake = self.D_pix2pix(self.fake_B)
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
    
        # Real
        pred_real = self.D_pix2pix(self.real_B)
        self.loss_D_real =self.criterionGAN(pred_real, True)

        # Combined loss
        self.loss_D = (self.loss_D_fake + self.loss_D_real)*0.5

        self.loss_D.backward(retain_graph=True)
        self.d_pix2pix_optimizer.step()

    def backward_G(self):
        # First, G(A) should fake the discriminator
        pred_fake = self.D_pix2pix(self.fake_B)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)


        self.loss_G_L1 = 50*(self.vgg1(self.fake_B, self.real_B)+self.criterionL1(self.fake_B, self.real_B))


        self.loss_G = self.loss_G_GAN+self.loss_G_L1

        self.loss_G.backward()

    def train(self):



        """Train StarGAN within a single dataset."""
        # Set data loader.
        data_loaderA = self.data_loaderA
        data_loaderB = self.data_loaderB
        if not os.path.exists(depth_dir):
            os.makedirs(depth_dir)


        # Fetch fixed inputs for debugging.
        data_iterA = iter(data_loaderA)
        data_iterB = iter(data_loaderB)
        x_fixed, c_org = next(data_iterA)
        x_fixed = x_fixed.to(self.device)
        c_fixed_list = self.create_labels(c_org, self.c_dim)

        # Learning rate cache for decaying.
        g_lr = self.g_lr
        d_lr = self.d_lr
        lr_pix2pix = self.lr_pix2pix
        d_lr_pix2pix=self.d_lr_pix2pix
        # Start training from scratch or resume training.
        start_iters = 0
        if self.resume_iters:
            start_iters = self.resume_iters
            self.restore_model_all(self.resume_iters)
        # Start training.
        print('Start training...')
        start_time = time.time()

        for i in range(start_iters, self.num_iters):
            # Fetch real images and labels.
            try:
                x_orgA, label_orgA = next(data_iterA)
                x_orgB, label_orgB = next(data_iterB)
            except:
                data_iterA = iter(data_loaderA)
                x_orgA, label_orgA = next(data_iterA)
                data_iterB = iter(data_loaderB)
                x_orgB, label_orgB = next(data_iterB)
            x_orgB = x_orgB.to(self.device)
            x_orgA = x_orgA.to(self.device)
            x_orgB[:,3:,:,:]=self.G_pix2pix(x_orgB[:,:3,:,:])

            label_orgB=torch.add(label_orgB,1)
            rand_idx = torch.randperm(label_orgA.size(0))
            label_trgA = label_orgB[rand_idx]

            c_org = self.label2onehot(label_orgA, self.c_dim)
            c_trg = self.label2onehot(label_trgA, self.c_dim)
            c_orgb = self.label2onehot(label_orgB, self.c_dim)

            c_org = c_org.to(self.device)             # Original domain labels.
            c_trg = c_trg.to(self.device)
            c_orgb = c_orgb.to(self.device)
            label_orgA = label_orgA.to(self.device)
            label_trgA = label_trgA.to(self.device)
            label_orgB = label_orgB.to(self.device)

            # =================================================================================== #
            #                             2. Train the discriminator                              #
            # =================================================================================== #

            # Compute loss with real images.

            out_src,out_cls = self.D(x_orgB[:, :3, :, :])
            
            
            d_loss_cls = self.classification_loss(out_cls, label_orgB)
            loss_D_real = self.criterionGAN(out_src, True)#


            x_fake = self.G(x_orgA, c_trg)
            out_src, out_cls= self.D(x_fake.detach())

            loss_D_fake = self.criterionGAN(out_src, False)
            loss_D = (loss_D_real + loss_D_fake)*0.5
            
            # Backward and optimize.
            d_loss =  loss_D+ self.lambda_cls * d_loss_cls
            self.reset_grad()
            d_loss.backward()
            self.d_optimizer.step()
            # Logging.
            loss = {}
            loss['D/cls'] = d_loss_cls.item()
            loss['D/ls'] = loss_D.item()


            # =================================================================================== #
            #                               3. Train the generator                                #
            # =================================================================================== #

            self.real_A = x_fake[:, :3, :, :].clone()
            self.real_B = x_orgA[:, 3:, :, :].clone()
            self.fake_B = self.G_pix2pix(self.real_A)
            self.set_requires_grad(self.D_pix2pix, True)
            self.d_pix2pix_optimizer.zero_grad()
            self.backward_D()
            if (i + 1) % 1000 == 0:
                real_depth = torch.cat((self.real_A, self.fake_B), dim=3)
                real_depth_target = torch.cat((real_depth, self.real_B), dim=3)
                result_path = os.path.join(self.images_save_dir, '{}-images.jpg'.format(i + 1))
                save_image(self.denorm(real_depth_target.data.cpu()), result_path, nrow=1, padding=0)
            if (i + 1) % 1000 == 0:
                origin_depth = torch.cat((x_orgB[:,:3,:,:], x_orgB[:,3:,:,:]), dim=3)
                depth_path = os.path.join(depth_dir, '{}-images.jpg'.format(i + 1))
                save_image(self.denorm(origin_depth.data.cpu()), depth_path, nrow=1, padding=0)

            # Original-to-target domain.
            x_fake = self.G(x_orgA, c_trg)#
            self.set_requires_grad(self.D_pix2pix, False)
            self.g_pix2pix_optimizer.zero_grad()
            self.backward_G()
            self.g_pix2pix_optimizer.step()

            out_src,out_cls= self.D(x_fake)
            g_loss_fake = self.criterionGAN(out_src, True)
            g_loss_cls = self.classification_loss(out_cls, label_trgA)
            # Target-to-original domain.
            x_fake1 = torch.cat((x_fake[:, :3, :, :], x_orgA[:, 3:, :, :]), dim=1)
            x_reconst = self.G(x_fake1, c_org)

            g_loss_rec = torch.mean(torch.abs(x_orgA[:, :3, :, :] - x_reconst))
            perA = self.per.get_loss(x_fake[:, :3, :, :], x_orgA[:, :3, :, :])

            # Backward and optimize.
            g_loss = g_loss_fake + self.lambda_cls * g_loss_cls +self.lambda_rec * g_loss_rec+perA*0.1
            self.reset_grad()
            g_loss.backward()
            self.g_optimizer.step()
            # Logging.
            loss['G/loss_fake'] = g_loss_fake.item()
            loss['G/loss_rec'] = g_loss_rec.item()
            loss['G/cls'] = g_loss_cls.item()
            loss['G/perA'] = perA.item()
            loss['G/L1_loss'] = self.loss_G_L1.item()

            # =================================================================================== #
            #                                 4. Miscellaneous                                    #
            # =================================================================================== #

            # Print out training information.
            if (i+1) % self.log_step == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration [{}/{}]".format(et, i+1, self.num_iters)
                for tag, value in loss.items():
                    log += ", {}: {:.7f}".format(tag, value)
                print(log)

                if self.use_tensorboard:
                    for tag, value in loss.items():
                        self.logger.scalar_summary(tag, value, i+1)

            # Translate fixed images for debugging.
            if (i + 1) % self.sample_step == 0:
                with torch.no_grad():
                    x_fake_list = [x_fixed[:,:3,:,:]]
                    for c_fixed in c_fixed_list:
                        x_fake_list.append(self.G(x_fixed, c_fixed))
                    x_concat = torch.cat(x_fake_list, dim=3)
                    x_concat = x_concat[:,:3,:,:]
                    sample_path = os.path.join(self.sample_dir, '{}-images.jpg'.format(i + 1))
                    save_image(self.denorm(x_concat.data.cpu()), sample_path, nrow=1, padding=0)
                    print('Saved real and fake images into {}...'.format(sample_path))

            # Save model checkpoints.
            if (i+1) % self.model_save_step == 0:
                G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(i+1))
                D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(i+1))
                G_pix2pix_path = os.path.join(self.images_dir_pixpix, '{}-G.pth'.format(i + 1))
                D_pix2pix_path = os.path.join(self.images_dir_pixpix, '{}-D.pth'.format(i + 1))
                torch.save(self.G.state_dict(), G_path)
                torch.save(self.D.state_dict(), D_path)
                torch.save(self.G_pix2pix.state_dict(), G_pix2pix_path)
                torch.save(self.D_pix2pix.state_dict(), D_pix2pix_path)
                print('Saved model checkpoints into {}...'.format(self.model_save_dir))

            # Decay learning rates.
            if (i+1) % self.lr_update_step == 0 and (i+1) > (self.num_iters - self.num_iters_decay):
                g_lr -= (self.g_lr / float(self.num_iters_decay))
                lr_pix2pix -= (self.lr_pix2pix / float(self.num_iters_decay))
                d_lr -= (self.d_lr / float(self.num_iters_decay))
                d_lr_pix2pix -= (self.d_lr_pix2pix / float(self.num_iters_decay))
                self.update_lr(g_lr, d_lr,lr_pix2pix,d_lr_pix2pix)
                print('Decayed learning rates, g_lr: {}, d_lr: {}, lr_pix2pix: {},d_lr_pix2pix:{}.'.format(g_lr, d_lr, lr_pix2pix,d_lr_pix2pix))

    def test(self):
        path_list = os.walk(self.result_dir)
        for root, _, image_list in path_list:
            for ele in image_list:
                os.remove(os.path.join(root, ele))

        """Translate images using StarGAN trained on a single dataset."""
        # Load the trained generator.
        self.restore_model(self.test_iters)
        
        # Set data loader.
        data_loader = self.data_loaderA

        with torch.no_grad():
            for i, (x_real1, c_org) in enumerate(data_loader):
                x_real_name = data_loader.batch_sampler.sampler.data_source.imgs[i][0]
                basename = os.path.basename(x_real_name)
                basename = basename[0:-6]
                x_real = x_real1.to(self.device)
                c_trg_list = self.create_labels(c_org, self.c_dim)

                # Translate images.
                x_fake_list = [x_real[:,:3,:,:]]
                for c_trg in c_trg_list:
                    aa = self.G(x_real, c_trg)
                    list = aa[:,:3,:,:]
                    x_fake_list.append(list)
                result_path = os.path.join(self.result_dir, '{}.png'.format(basename))

                x_concat = torch.cat(x_fake_list, dim=3)
                save_image(self.denorm(x_concat.data.cpu()), result_path, nrow=1, padding=0)

                print('Saved real and fake images into {}...'.format(i))

    def test_pix2pix(self):
        if not os.path.exists(self.result_pix2pix_dir):
            os.makedirs(self.result_pix2pix_dir)
        path_list = os.walk(self.result_pix2pix_dir)
        for root, _, image_list in path_list:
            for ele in image_list:
                os.remove(os.path.join(root, ele))

        """Translate images using StarGAN trained on a single dataset."""
        # Load the trained generator.
        self.restore_model_pix2pix(self.test_iters)

        # Set data loader.
        data_loader = self.data_loaderA

        with torch.no_grad():
            for i, (x_real, c_org) in enumerate(data_loader):
                x_real_name = data_loader.batch_sampler.sampler.data_source.imgs[i][0]
                basename = os.path.basename(x_real_name)
                basename =  basename[:-4]#basename[:8]#
                print(basename)
                # Prepare input images and target domain labels.
                x_real = x_real[:, :3, :, :]
                x_real = x_real.to(self.device)


                # Translate images.
                x_fake_list = [x_real]
                x_real=self.G_pix2pix(x_real)
                x_fake_list.append(x_real)
                # Save the translated images.
                x_concat = torch.cat(x_fake_list, dim=3)
                result_path = os.path.join(self.result_pix2pix_dir, '{}.png'.format(basename))
                save_image(self.denorm(x_concat.data.cpu()), result_path, nrow=1, padding=0)
                print('Saved real and fake images into {}.'.format(result_path))

