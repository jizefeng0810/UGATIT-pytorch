import time, itertools
from dataset import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
from networks import *
from utils import *
from glob import glob
from distributed import *
import torch
import torch.distributed as dist

class UGATIT(object) :
    def __init__(self, args):
        self.light = False
        if args.model == 'UGATIT':
            self.light = args.light

            if self.light :
                self.model_name = 'UGATIT_light'
            else :
                self.model_name = 'UGATIT'
        else:
            self.model_name = args.model

        self.result_dir = args.result_dir
        self.dataset = args.dataset
        self.exp_name = args.exp_name + '_' + self.model_name

        self.iteration = args.iteration
        self.decay_flag = args.decay_flag

        self.batch_size = args.batch_size
        self.print_freq = args.print_freq
        self.save_freq = args.save_freq

        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.ch = args.ch

        """ Weight """
        self.adv_weight = args.adv_weight
        self.cycle_weight = args.cycle_weight
        self.identity_weight = args.identity_weight
        self.cam_weight = args.cam_weight

        """ Generator """
        self.n_res = args.n_res

        """ Discriminator """
        self.n_dis = args.n_dis

        self.img_size = args.img_size
        self.img_ch = args.img_ch

        self.device = args.device
        self.benchmark_flag = args.benchmark_flag
        self.resume = args.resume

        # ddp
        self.num_worker = args.num_worker
        self.local_rank = args.local_rank

        if torch.backends.cudnn.enabled and self.benchmark_flag:
            print('set benchmark !')
            torch.backends.cudnn.benchmark = True
        
        self.world_size = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
        self.distributed = args.distributed
        assert torch.cuda.device_count() > self.local_rank
        if self.distributed:
            torch.cuda.set_device(self.local_rank)
            torch.distributed.init_process_group(backend="nccl", init_method="env://")
            # synchronize()

        print()

        print("##### Information #####")
        print("# light : ", self.light)
        print("# dataset : ", self.dataset)
        print("# batch_size : ", self.batch_size)
        print("# iteration per epoch : ", self.iteration)

        print()

        print("##### Generator #####")
        print("# residual blocks : ", self.n_res)

        print()

        print("##### Discriminator #####")
        print("# discriminator layer : ", self.n_dis)

        print()

        print("##### Weight #####")
        print("# adv_weight : ", self.adv_weight)
        print("# cycle_weight : ", self.cycle_weight)
        print("# identity_weight : ", self.identity_weight)
        print("# cam_weight : ", self.cam_weight)

    ##################################################################################
    # Model
    ##################################################################################

    def build_model(self):
        """ DataLoader """
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize((self.img_size + 30, self.img_size+30)),
            transforms.RandomCrop(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        test_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

        self.trainA = ImageFolder(self.dataset, 'trainA', train_transform)
        self.trainB = ImageFolder(self.dataset, 'trainB', train_transform)
        self.testA = ImageFolder(self.dataset, 'testA', test_transform)
        self.testB = ImageFolder(self.dataset, 'testB', test_transform)
        if self.distributed:
            trainA_sampler = torch.utils.data.distributed.DistributedSampler(self.trainA, shuffle=True)
            trainB_sampler = torch.utils.data.distributed.DistributedSampler(self.trainB, shuffle=True)
        else:
            trainA_sampler = None
            trainB_sampler = None
        self.trainA_loader = DataLoader(self.trainA, batch_size=self.batch_size, shuffle=(trainA_sampler is None), sampler=trainA_sampler, num_workers=self.num_worker)
        self.trainB_loader = DataLoader(self.trainB, batch_size=self.batch_size, shuffle=(trainB_sampler is None), sampler=trainB_sampler, num_workers=self.num_worker)
        self.testA_loader = DataLoader(self.testA, batch_size=1, shuffle=False)
        self.testB_loader = DataLoader(self.testB, batch_size=1, shuffle=False)

        """ Define Generator, Discriminator """
        if self.model_name == 'UGATIT' or self.model_name == 'UGATIT_light':
            self.genA2B = ResnetGenerator(input_nc=3, output_nc=3, ngf=self.ch, n_blocks=self.n_res, img_size=self.img_size, light=self.light).to(self.device)
            self.genB2A = ResnetGenerator(input_nc=3, output_nc=3, ngf=self.ch, n_blocks=self.n_res, img_size=self.img_size, light=self.light).to(self.device)
        elif self.model_name == 'UNet':
            self.genA2B = Generator(in_c=3, type='unet').to(self.device)
            self.genB2A = Generator(in_c=3, type='unet').to(self.device)
        
        self.disGA = Discriminator(input_nc=3, ndf=self.ch, n_layers=7).to(self.device)
        self.disGB = Discriminator(input_nc=3, ndf=self.ch, n_layers=7).to(self.device)
        self.disLA = Discriminator(input_nc=3, ndf=self.ch, n_layers=5).to(self.device)
        self.disLB = Discriminator(input_nc=3, ndf=self.ch, n_layers=5).to(self.device)

        if self.distributed:
            self.genA2B = nn.parallel.DistributedDataParallel(self.genA2B, device_ids=[self.local_rank], output_device=self.local_rank, broadcast_buffers=False,)
            self.genB2A = nn.parallel.DistributedDataParallel(self.genB2A, device_ids=[self.local_rank], output_device=self.local_rank, broadcast_buffers=False,)
            self.disGA = nn.parallel.DistributedDataParallel(self.disGA, device_ids=[self.local_rank], output_device=self.local_rank, broadcast_buffers=False,)
            self.disGB = nn.parallel.DistributedDataParallel(self.disGB, device_ids=[self.local_rank], output_device=self.local_rank, broadcast_buffers=False,)
            self.disLA = nn.parallel.DistributedDataParallel(self.disLA, device_ids=[self.local_rank], output_device=self.local_rank, broadcast_buffers=False,)
            self.disLB = nn.parallel.DistributedDataParallel(self.disLB, device_ids=[self.local_rank], output_device=self.local_rank, broadcast_buffers=False,)


        """ Define Loss """
        self.L1_loss = nn.L1Loss().to(self.device)
        self.MSE_loss = nn.MSELoss().to(self.device)
        self.BCE_loss = nn.BCEWithLogitsLoss().to(self.device)

        """ Trainer """
        self.G_optim = torch.optim.Adam(itertools.chain(self.genA2B.parameters(), self.genB2A.parameters()), lr=self.lr, betas=(0.5, 0.999), weight_decay=self.weight_decay)
        self.D_optim = torch.optim.Adam(itertools.chain(self.disGA.parameters(), self.disGB.parameters(), self.disLA.parameters(), self.disLB.parameters()), lr=self.lr, betas=(0.5, 0.999), weight_decay=self.weight_decay)

        """ Define Rho clipper to constraint the value of rho in AdaILN and ILN"""
        self.Rho_clipper = RhoClipper(0, 1)

    def train(self):
        if self.distributed:
            self.genA2B, self.genB2A, self.disGA, self.disGB, self.disLA, self.disLB = self.genA2B.module, self.genB2A.module, self.disGA.module, self.disGB.module, self.disLA.module, self.disLB.module
        self.genA2B.train(), self.genB2A.train(), self.disGA.train(), self.disGB.train(), self.disLA.train(), self.disLB.train()

        start_iter = 1
        if self.resume:
            model_list = glob(os.path.join(self.result_dir, self.exp_name, 'model', '*.pt'))
            if not len(model_list) == 0:
                model_list.sort()
                start_iter = int(model_list[-1].split('_')[-1].split('.')[0])
                self.load(os.path.join(self.result_dir, self.exp_name, 'model'), start_iter)
                print(" [*] Load SUCCESS")
                if self.decay_flag and start_iter > (self.iteration // 2):
                    self.G_optim.param_groups[0]['lr'] -= (self.lr / (self.iteration // 2)) * (start_iter - self.iteration // 2)
                    self.D_optim.param_groups[0]['lr'] -= (self.lr / (self.iteration // 2)) * (start_iter - self.iteration // 2)

        # training loop
        print('training start !')
        start_time = time.time()
        for step in range(start_iter, self.iteration + 1):
            if self.decay_flag and step > (self.iteration // 2):
                self.G_optim.param_groups[0]['lr'] -= (self.lr / (self.iteration // 2))
                self.D_optim.param_groups[0]['lr'] -= (self.lr / (self.iteration // 2))

            try:
                real_A, _ = trainA_iter.next()
            except:
                trainA_iter = iter(self.trainA_loader)
                real_A, _ = trainA_iter.next()

            try:
                real_B, _ = trainB_iter.next()
            except:
                trainB_iter = iter(self.trainB_loader)
                real_B, _ = trainB_iter.next()

            real_A, real_B = real_A.to(self.device), real_B.to(self.device)

            # Update D
            self.D_optim.zero_grad()

            if self.model_name == 'UGATIT' or self.model_name == 'UGATIT_light':
                fake_A2B, _, _ = self.genA2B(real_A)
                fake_B2A, _, _ = self.genB2A(real_B)
            elif self.model_name == 'UNet':
                fake_A2B = self.genA2B(real_A)
                fake_B2A = self.genB2A(real_B)

            real_GA_logit, real_GA_cam_logit, _ = self.disGA(real_A)
            real_LA_logit, real_LA_cam_logit, _ = self.disLA(real_A)
            real_GB_logit, real_GB_cam_logit, _ = self.disGB(real_B)
            real_LB_logit, real_LB_cam_logit, _ = self.disLB(real_B)

            fake_GA_logit, fake_GA_cam_logit, _ = self.disGA(fake_B2A)
            fake_LA_logit, fake_LA_cam_logit, _ = self.disLA(fake_B2A)
            fake_GB_logit, fake_GB_cam_logit, _ = self.disGB(fake_A2B)
            fake_LB_logit, fake_LB_cam_logit, _ = self.disLB(fake_A2B)

            D_ad_loss_GA = self.MSE_loss(real_GA_logit, torch.ones_like(real_GA_logit).to(self.device)) + self.MSE_loss(fake_GA_logit, torch.zeros_like(fake_GA_logit).to(self.device))
            D_ad_cam_loss_GA = self.MSE_loss(real_GA_cam_logit, torch.ones_like(real_GA_cam_logit).to(self.device)) + self.MSE_loss(fake_GA_cam_logit, torch.zeros_like(fake_GA_cam_logit).to(self.device))
            D_ad_loss_LA = self.MSE_loss(real_LA_logit, torch.ones_like(real_LA_logit).to(self.device)) + self.MSE_loss(fake_LA_logit, torch.zeros_like(fake_LA_logit).to(self.device))
            D_ad_cam_loss_LA = self.MSE_loss(real_LA_cam_logit, torch.ones_like(real_LA_cam_logit).to(self.device)) + self.MSE_loss(fake_LA_cam_logit, torch.zeros_like(fake_LA_cam_logit).to(self.device))
            D_ad_loss_GB = self.MSE_loss(real_GB_logit, torch.ones_like(real_GB_logit).to(self.device)) + self.MSE_loss(fake_GB_logit, torch.zeros_like(fake_GB_logit).to(self.device))
            D_ad_cam_loss_GB = self.MSE_loss(real_GB_cam_logit, torch.ones_like(real_GB_cam_logit).to(self.device)) + self.MSE_loss(fake_GB_cam_logit, torch.zeros_like(fake_GB_cam_logit).to(self.device))
            D_ad_loss_LB = self.MSE_loss(real_LB_logit, torch.ones_like(real_LB_logit).to(self.device)) + self.MSE_loss(fake_LB_logit, torch.zeros_like(fake_LB_logit).to(self.device))
            D_ad_cam_loss_LB = self.MSE_loss(real_LB_cam_logit, torch.ones_like(real_LB_cam_logit).to(self.device)) + self.MSE_loss(fake_LB_cam_logit, torch.zeros_like(fake_LB_cam_logit).to(self.device))

            D_loss_A = self.adv_weight * (D_ad_loss_GA + D_ad_cam_loss_GA + D_ad_loss_LA + D_ad_cam_loss_LA)
            D_loss_B = self.adv_weight * (D_ad_loss_GB + D_ad_cam_loss_GB + D_ad_loss_LB + D_ad_cam_loss_LB)

            Discriminator_loss = D_loss_A + D_loss_B
            Discriminator_loss.backward()
            self.D_optim.step()

            # Update G
            self.G_optim.zero_grad()

            if self.model_name == 'UGATIT' or self.model_name == 'UGATIT_light':
                fake_A2B, fake_A2B_cam_logit, _ = self.genA2B(real_A)
                fake_B2A, fake_B2A_cam_logit, _ = self.genB2A(real_B)

                fake_A2B2A, _, _ = self.genB2A(fake_A2B)
                fake_B2A2B, _, _ = self.genA2B(fake_B2A)

                fake_A2A, fake_A2A_cam_logit, _ = self.genB2A(real_A)
                fake_B2B, fake_B2B_cam_logit, _ = self.genA2B(real_B)
            elif self.model_name == 'UNet':
                fake_A2B = self.genA2B(real_A)
                fake_B2A = self.genB2A(real_B)

                fake_A2B2A = self.genB2A(fake_A2B)
                fake_B2A2B = self.genA2B(fake_B2A)

                fake_A2A = self.genB2A(real_A)
                fake_B2B = self.genA2B(real_B)

            fake_GA_logit, fake_GA_cam_logit, _ = self.disGA(fake_B2A)
            fake_LA_logit, fake_LA_cam_logit, _ = self.disLA(fake_B2A)
            fake_GB_logit, fake_GB_cam_logit, _ = self.disGB(fake_A2B)
            fake_LB_logit, fake_LB_cam_logit, _ = self.disLB(fake_A2B)

            G_ad_loss_GA = self.MSE_loss(fake_GA_logit, torch.ones_like(fake_GA_logit).to(self.device))
            G_ad_cam_loss_GA = self.MSE_loss(fake_GA_cam_logit, torch.ones_like(fake_GA_cam_logit).to(self.device))
            G_ad_loss_LA = self.MSE_loss(fake_LA_logit, torch.ones_like(fake_LA_logit).to(self.device))
            G_ad_cam_loss_LA = self.MSE_loss(fake_LA_cam_logit, torch.ones_like(fake_LA_cam_logit).to(self.device))
            G_ad_loss_GB = self.MSE_loss(fake_GB_logit, torch.ones_like(fake_GB_logit).to(self.device))
            G_ad_cam_loss_GB = self.MSE_loss(fake_GB_cam_logit, torch.ones_like(fake_GB_cam_logit).to(self.device))
            G_ad_loss_LB = self.MSE_loss(fake_LB_logit, torch.ones_like(fake_LB_logit).to(self.device))
            G_ad_cam_loss_LB = self.MSE_loss(fake_LB_cam_logit, torch.ones_like(fake_LB_cam_logit).to(self.device))

            G_recon_loss_A = self.L1_loss(fake_A2B2A, real_A)
            G_recon_loss_B = self.L1_loss(fake_B2A2B, real_B)

            G_identity_loss_A = self.L1_loss(fake_A2A, real_A)
            G_identity_loss_B = self.L1_loss(fake_B2B, real_B)

            if self.model_name == 'UGATIT' or self.model_name == 'UGATIT_light':
                G_cam_loss_A = self.BCE_loss(fake_B2A_cam_logit, torch.ones_like(fake_B2A_cam_logit).to(self.device)) + self.BCE_loss(fake_A2A_cam_logit, torch.zeros_like(fake_A2A_cam_logit).to(self.device))
                G_cam_loss_B = self.BCE_loss(fake_A2B_cam_logit, torch.ones_like(fake_A2B_cam_logit).to(self.device)) + self.BCE_loss(fake_B2B_cam_logit, torch.zeros_like(fake_B2B_cam_logit).to(self.device))

            if self.model_name == 'UGATIT' or self.model_name == 'UGATIT_light':
                G_loss_A =  self.adv_weight * (G_ad_loss_GA + G_ad_cam_loss_GA + G_ad_loss_LA + G_ad_cam_loss_LA) + self.cycle_weight * G_recon_loss_A + self.identity_weight * G_identity_loss_A + self.cam_weight * G_cam_loss_A
                G_loss_B = self.adv_weight * (G_ad_loss_GB + G_ad_cam_loss_GB + G_ad_loss_LB + G_ad_cam_loss_LB) + self.cycle_weight * G_recon_loss_B + self.identity_weight * G_identity_loss_B + self.cam_weight * G_cam_loss_B
            elif self.model_name == 'UNet':
                G_loss_A =  self.adv_weight * (G_ad_loss_GA + G_ad_cam_loss_GA + G_ad_loss_LA + G_ad_cam_loss_LA) + self.cycle_weight * G_recon_loss_A + self.identity_weight * G_identity_loss_A
                G_loss_B = self.adv_weight * (G_ad_loss_GB + G_ad_cam_loss_GB + G_ad_loss_LB + G_ad_cam_loss_LB) + self.cycle_weight * G_recon_loss_B + self.identity_weight * G_identity_loss_B

            Generator_loss = G_loss_A + G_loss_B
            Generator_loss.backward()
            self.G_optim.step()

            # clip parameter of AdaILN and ILN, applied after optimizer step
            self.genA2B.apply(self.Rho_clipper)
            self.genB2A.apply(self.Rho_clipper)

            if self.distributed == False or dist.get_rank()==0 :
                if step % 10 == 0:
                    print("[%5d/%5d] time: %4.4f d_loss: %.8f, g_loss: %.8f" % (step, self.iteration, time.time() - start_time, Discriminator_loss, Generator_loss))
                if step % self.print_freq == 0:
                    train_sample_num = 5
                    test_sample_num = 5
                    if self.model_name == 'UGATIT' or self.model_name == 'UGATIT_light':
                        A2B = np.zeros((self.img_size * 7, 0, 3))
                        B2A = np.zeros((self.img_size * 7, 0, 3))
                    elif self.model_name == 'UNet':
                        A2B = np.zeros((self.img_size * 4, 0, 3))
                        B2A = np.zeros((self.img_size * 4, 0, 3))

                    self.genA2B.eval(), self.genB2A.eval(), self.disGA.eval(), self.disGB.eval(), self.disLA.eval(), self.disLB.eval()
                    for _ in range(train_sample_num):
                        try:
                            real_A, _ = trainA_iter.next()
                        except:
                            trainA_iter = iter(self.trainA_loader)
                            real_A, _ = trainA_iter.next()

                        try:
                            real_B, _ = trainB_iter.next()
                        except:
                            trainB_iter = iter(self.trainB_loader)
                            real_B, _ = trainB_iter.next()
                        real_A, real_B = real_A.to(self.device), real_B.to(self.device)

                        if self.model_name == 'UGATIT' or self.model_name == 'UGATIT_light':
                            fake_A2B, _, fake_A2B_heatmap = self.genA2B(real_A)
                            fake_B2A, _, fake_B2A_heatmap = self.genB2A(real_B)

                            fake_A2B2A, _, fake_A2B2A_heatmap = self.genB2A(fake_A2B)
                            fake_B2A2B, _, fake_B2A2B_heatmap = self.genA2B(fake_B2A)

                            fake_A2A, _, fake_A2A_heatmap = self.genB2A(real_A)
                            fake_B2B, _, fake_B2B_heatmap = self.genA2B(real_B)

                            A2B = np.concatenate((A2B,  np.concatenate((RGB2BGR(tensor2numpy(denorm(real_A[0])), 'A2A', self.img_size),
                                                                        cam(tensor2numpy(fake_A2A_heatmap[0]), self.img_size),
                                                                        RGB2BGR(tensor2numpy(denorm(fake_A2A[0])), 'fake_A2A', self.img_size),
                                                                        cam(tensor2numpy(fake_A2B_heatmap[0]), self.img_size),
                                                                        RGB2BGR(tensor2numpy(denorm(fake_A2B[0])), 'fake_A2B', self.img_size),
                                                                        cam(tensor2numpy(fake_A2B2A_heatmap[0]), self.img_size),
                                                                        RGB2BGR(tensor2numpy(denorm(fake_A2B2A[0])), 'fake_A2B2A', self.img_size)), 0)), 1)

                            B2A = np.concatenate((B2A, np.concatenate((RGB2BGR(tensor2numpy(denorm(real_B[0])), 'real_B', self.img_size),
                                                                    cam(tensor2numpy(fake_B2B_heatmap[0]), self.img_size),
                                                                    RGB2BGR(tensor2numpy(denorm(fake_B2B[0])), 'fake_B2B', self.img_size),
                                                                    cam(tensor2numpy(fake_B2A_heatmap[0]), self.img_size),
                                                                    RGB2BGR(tensor2numpy(denorm(fake_B2A[0])), 'fake_B2A', self.img_size),
                                                                    cam(tensor2numpy(fake_B2A2B_heatmap[0]), self.img_size),
                                                                    RGB2BGR(tensor2numpy(denorm(fake_B2A2B[0])), 'fake_B2A2B', self.img_size)), 0)), 1)
                        elif self.model_name == 'UNet':
                            fake_A2B = self.genA2B(real_A)
                            fake_B2A = self.genB2A(real_B)

                            fake_A2B2A = self.genB2A(fake_A2B)
                            fake_B2A2B = self.genA2B(fake_B2A)

                            fake_A2A = self.genB2A(real_A)
                            fake_B2B = self.genA2B(real_B)

                            A2B = np.concatenate((A2B,  np.concatenate((RGB2BGR(tensor2numpy(denorm(real_A[0])), 'A2A', self.img_size),
                                                                        RGB2BGR(tensor2numpy(denorm(fake_A2A[0])), 'fake_A2A', self.img_size),
                                                                        RGB2BGR(tensor2numpy(denorm(fake_A2B[0])), 'fake_A2B', self.img_size),
                                                                        RGB2BGR(tensor2numpy(denorm(fake_A2B2A[0])), 'fake_A2B2A', self.img_size)), 0)), 1)

                            B2A = np.concatenate((B2A, np.concatenate((RGB2BGR(tensor2numpy(denorm(real_B[0])), 'real_B', self.img_size),
                                                                    RGB2BGR(tensor2numpy(denorm(fake_B2B[0])), 'fake_B2B', self.img_size),
                                                                    RGB2BGR(tensor2numpy(denorm(fake_B2A[0])), 'fake_B2A', self.img_size),
                                                                    RGB2BGR(tensor2numpy(denorm(fake_B2A2B[0])), 'fake_B2A2B', self.img_size)), 0)), 1)

                    for _ in range(test_sample_num):
                        try:
                            real_A, _ = testA_iter.next()
                        except:
                            testA_iter = iter(self.testA_loader)
                            real_A, _ = testA_iter.next()

                        try:
                            real_B, _ = testB_iter.next()
                        except:
                            testB_iter = iter(self.testB_loader)
                            real_B, _ = testB_iter.next()
                        real_A, real_B = real_A.to(self.device), real_B.to(self.device)

                        if self.model_name == 'UGATIT' or self.model_name == 'UGATIT_light':
                            fake_A2B, _, fake_A2B_heatmap = self.genA2B(real_A)
                            fake_B2A, _, fake_B2A_heatmap = self.genB2A(real_B)

                            fake_A2B2A, _, fake_A2B2A_heatmap = self.genB2A(fake_A2B)
                            fake_B2A2B, _, fake_B2A2B_heatmap = self.genA2B(fake_B2A)

                            fake_A2A, _, fake_A2A_heatmap = self.genB2A(real_A)
                            fake_B2B, _, fake_B2B_heatmap = self.genA2B(real_B)

                            A2B = np.concatenate((A2B, np.concatenate((RGB2BGR(tensor2numpy(denorm(real_A[0])), 'real_A', self.img_size),
                                                                    cam(tensor2numpy(fake_A2A_heatmap[0]), self.img_size),
                                                                    RGB2BGR(tensor2numpy(denorm(fake_A2A[0])), 'fake_A2A', self.img_size),
                                                                    cam(tensor2numpy(fake_A2B_heatmap[0]), self.img_size),
                                                                    RGB2BGR(tensor2numpy(denorm(fake_A2B[0])), 'fake_A2B', self.img_size),
                                                                    cam(tensor2numpy(fake_A2B2A_heatmap[0]), self.img_size),
                                                                    RGB2BGR(tensor2numpy(denorm(fake_A2B2A[0])), 'fake_A2B2A', self.img_size)), 0)), 1)

                            B2A = np.concatenate((B2A, np.concatenate((RGB2BGR(tensor2numpy(denorm(real_B[0])), 'real_B', self.img_size),
                                                                    cam(tensor2numpy(fake_B2B_heatmap[0]), self.img_size),
                                                                    RGB2BGR(tensor2numpy(denorm(fake_B2B[0])), 'fake_B2B', self.img_size),
                                                                    cam(tensor2numpy(fake_B2A_heatmap[0]), self.img_size),
                                                                    RGB2BGR(tensor2numpy(denorm(fake_B2A[0])), 'fake_B2A', self.img_size),
                                                                    cam(tensor2numpy(fake_B2A2B_heatmap[0]), self.img_size),
                                                                    RGB2BGR(tensor2numpy(denorm(fake_B2A2B[0])), 'fake_B2A2B', self.img_size)), 0)), 1)
                        elif self.model_name == 'UNet':
                            fake_A2B = self.genA2B(real_A)
                            fake_B2A = self.genB2A(real_B)

                            fake_A2B2A = self.genB2A(fake_A2B)
                            fake_B2A2B = self.genA2B(fake_B2A)

                            fake_A2A = self.genB2A(real_A)
                            fake_B2B = self.genA2B(real_B)

                            A2B = np.concatenate((A2B, np.concatenate((RGB2BGR(tensor2numpy(denorm(real_A[0])), 'real_A', self.img_size),
                                                                    RGB2BGR(tensor2numpy(denorm(fake_A2A[0])), 'fake_A2A', self.img_size),
                                                                    RGB2BGR(tensor2numpy(denorm(fake_A2B[0])), 'fake_A2B', self.img_size),
                                                                    RGB2BGR(tensor2numpy(denorm(fake_A2B2A[0])), 'fake_A2B2A', self.img_size)), 0)), 1)

                            B2A = np.concatenate((B2A, np.concatenate((RGB2BGR(tensor2numpy(denorm(real_B[0])), 'real_B', self.img_size),
                                                                    RGB2BGR(tensor2numpy(denorm(fake_B2B[0])), 'fake_B2B', self.img_size),
                                                                    RGB2BGR(tensor2numpy(denorm(fake_B2A[0])), 'fake_B2A', self.img_size),
                                                                    RGB2BGR(tensor2numpy(denorm(fake_B2A2B[0])), 'fake_B2A2B', self.img_size)), 0)), 1)

                    cv2.imwrite(os.path.join(self.result_dir, self.exp_name, 'img', 'A2B_%07d.png' % step), A2B * 255.0)
                    cv2.imwrite(os.path.join(self.result_dir, self.exp_name, 'img', 'B2A_%07d.png' % step), B2A * 255.0)
                    self.genA2B.train(), self.genB2A.train(), self.disGA.train(), self.disGB.train(), self.disLA.train(), self.disLB.train()

                if step % self.save_freq == 0:
                    self.save(os.path.join(self.result_dir, self.exp_name, 'model'), step)

                if step % 1000 == 0:
                    params = {}
                    params['genA2B'] = self.genA2B.state_dict()
                    params['genB2A'] = self.genB2A.state_dict()
                    params['disGA'] = self.disGA.state_dict()
                    params['disGB'] = self.disGB.state_dict()
                    params['disLA'] = self.disLA.state_dict()
                    params['disLB'] = self.disLB.state_dict()
                    torch.save(params, os.path.join(self.result_dir, self.exp_name, self.exp_name + '_params_latest.pt'))

    def save(self, dir, step):
        params = {}
        params['genA2B'] = self.genA2B.state_dict()
        params['genB2A'] = self.genB2A.state_dict()
        params['disGA'] = self.disGA.state_dict()
        params['disGB'] = self.disGB.state_dict()
        params['disLA'] = self.disLA.state_dict()
        params['disLB'] = self.disLB.state_dict()
        torch.save(params, os.path.join(dir, self.exp_name + '_params_%07d.pt' % step))

    def load(self, dir, step):
        params = torch.load(os.path.join(dir, self.exp_name + '_params_%07d.pt' % step))
        self.genA2B.load_state_dict(params['genA2B'])
        self.genB2A.load_state_dict(params['genB2A'])
        self.disGA.load_state_dict(params['disGA'])
        self.disGB.load_state_dict(params['disGB'])
        self.disLA.load_state_dict(params['disLA'])
        self.disLB.load_state_dict(params['disLB'])

    def test(self):
        model_list = glob(os.path.join(self.result_dir, self.exp_name, 'model', '*.pt'))
        if not len(model_list) == 0:
            model_list.sort()
            iter = int(model_list[-1].split('_')[-1].split('.')[0])
            self.load(os.path.join(self.result_dir, self.exp_name, 'model'), iter)
            print(" [*] Load SUCCESS")
        else:
            print(" [*] Load FAILURE")
            return

        self.genA2B.eval(), self.genB2A.eval()
        for n, (real_A, _) in enumerate(self.testA_loader):
            real_A = real_A.to(self.device)

            fake_A2B, _, fake_A2B_heatmap = self.genA2B(real_A)

            fake_A2B2A, _, fake_A2B2A_heatmap = self.genB2A(fake_A2B)

            fake_A2A, _, fake_A2A_heatmap = self.genB2A(real_A)

            A2B = np.concatenate((RGB2BGR(tensor2numpy(denorm(real_A[0]))),
                                  cam(tensor2numpy(fake_A2A_heatmap[0]), self.img_size),
                                  RGB2BGR(tensor2numpy(denorm(fake_A2A[0]))),
                                  cam(tensor2numpy(fake_A2B_heatmap[0]), self.img_size),
                                  RGB2BGR(tensor2numpy(denorm(fake_A2B[0]))),
                                  cam(tensor2numpy(fake_A2B2A_heatmap[0]), self.img_size),
                                  RGB2BGR(tensor2numpy(denorm(fake_A2B2A[0])))), 0)

            cv2.imwrite(os.path.join(self.result_dir, self.exp_name, 'test', 'A2B_%d.png' % (n + 1)), A2B * 255.0)

        for n, (real_B, _) in enumerate(self.testB_loader):
            real_B = real_B.to(self.device)

            fake_B2A, _, fake_B2A_heatmap = self.genB2A(real_B)

            fake_B2A2B, _, fake_B2A2B_heatmap = self.genA2B(fake_B2A)

            fake_B2B, _, fake_B2B_heatmap = self.genA2B(real_B)

            B2A = np.concatenate((RGB2BGR(tensor2numpy(denorm(real_B[0]))),
                                  cam(tensor2numpy(fake_B2B_heatmap[0]), self.img_size),
                                  RGB2BGR(tensor2numpy(denorm(fake_B2B[0]))),
                                  cam(tensor2numpy(fake_B2A_heatmap[0]), self.img_size),
                                  RGB2BGR(tensor2numpy(denorm(fake_B2A[0]))),
                                  cam(tensor2numpy(fake_B2A2B_heatmap[0]), self.img_size),
                                  RGB2BGR(tensor2numpy(denorm(fake_B2A2B[0])))), 0)

            cv2.imwrite(os.path.join(self.result_dir, self.exp_name, 'test', 'B2A_%d.png' % (n + 1)), B2A * 255.0)
