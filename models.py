import torch
from network import Encoder,Decoder
from utils import weights_init, get_model_list, vgg_preprocess, load_vgg16, get_scheduler,ssim
import torch.nn as nn
import os
from torch.autograd import Variable

class Models(nn.Module):
    def __init__(self,hyperparameters):
        super(Models, self).__init__()
        lr = 0.0001


        self.enc = Encoder()
        self.dec = Decoder()

        # Setup the optimizers

        gen_params = list(self.enc.parameters())+list(self.dec.parameters())
        self.gen_opt = torch.optim.Adam([p for p in gen_params if p.requires_grad],lr=lr)
        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters)

        # Network weight initialization
        self.apply(weights_init(hyperparameters['init']))

        self.l1 = nn.L1Loss()
        self.l2 = nn.MSELoss()

        self.enc =nn.DataParallel(self.enc)
        self.dec = nn.DataParallel(self.dec)


    def l1_loss(self, input, target,mask):
        return self.l1(torch.mul(input,mask),torch.mul(target,mask))




    def recon_criterion_rmse(self, input, target, mask, denorm=False):
        if (denorm):
            input = (input * 0.5 + 0.5)
            target = (target * 0.5 + 0.5)
        out = 0
        psnr_ = 0
        ssim_ = 0
        if (len(input.shape) == 3):
            tmp = torch.sum((torch.mul(input, mask) - torch.mul(target, mask)) ** 2)
            tmp /= torch.sum(mask)
            tmp = tmp ** 0.5
            psnr = 20 * torch.log10(1 / tmp)
            img1 = torch.mul(input, mask) + torch.mul(target, 1 - mask)
            img1 = torch.unsqueeze(img1, dim=0)
            img2 = torch.unsqueeze(target, dim=0)
            ssim_loss = ssim(img1, img2)
            # ssim_loss = pytorch_ssim.SSIM(window_size=11)
            return tmp.item(), psnr.item(), ssim_loss.item()
        else:
            for i in range(len(input)):
                tmp = torch.sum((torch.mul(input[i], mask[i]) - torch.mul(target[i], mask[i])) ** 2)
                tmp /= torch.sum(mask[i])
                tmp = tmp ** 0.5
                out += tmp
                psnr_ += 20 * torch.log10(1 / tmp)

                img1 = torch.mul(input[i], mask[i]) + torch.mul(target[i], 1 - mask[i])
                img1 = torch.unsqueeze(img1, dim=0)
                img2 = torch.unsqueeze(target[i], dim=0)
                ssim_ += ssim(img1, img2)

            return (out / len(input)).item(), (psnr_ / len(input)).item(), (ssim_ / len(input)).item()

    def gen_update_rotate(self,Xa_out, Xb_out, light_a_label, light_b_label, r_Xb_out, mask, diff):
        self.gen_opt.zero_grad()
        # encode
        light_a, s_embed = self.enc.forward(Xa_out)


        #rotation
        r_light = light_b_label.clone()
        for i in range(len(r_light)):
            r_light[i] = torch.roll(r_light[i], round((1-int(diff[i])/12.0)*32.0), 2)


        # decode
        dec_relight = self.dec.forward(light_b_label, s_embed)
        dec_rec = self.dec.forward(light_a, s_embed)
        dec_r_relight = self.dec.forward(r_light, s_embed)

        # loss
        self.loss_light = self.l1(light_a, light_a_label)
        self.loss_relight = self.l1_loss(dec_relight, Xb_out, mask)
        self.loss_r_relight = self.l1_loss(dec_r_relight, r_Xb_out, mask)
        self.loss_rec = self.l1_loss(dec_rec, Xa_out, mask)



        self.loss_gen_total = 1.0 * self.loss_relight + 0.8 * self.loss_light + 1.0 * (
                    self.loss_rec + self.loss_r_relight)

        self.loss_gen_total.backward()
        self.gen_opt.step()

        dec_relight = torch.mul(dec_relight, mask)
        dec_rec = torch.mul(dec_rec, mask)

        image_anchor = Xa_out[0:1].detach().cpu()[:3]
        image_recons = dec_rec[0:1].detach().cpu()[:3]
        image_x_b = dec_relight[0:1].detach().cpu()[:3]
        image_gt = Xb_out[0:1].detach().cpu()[:3]

        self.image_display = torch.cat((image_anchor, image_recons, image_gt, image_x_b), dim=3)



    def test_output(self,I_s,I_t_gt,I_refer,mask,L_s,L_t,solid_mask):
        # encode
        en_s_l, en_s_embed = self.gen.encode(I_s)
        en_t_l, en_t_embed = self.gen.encode(I_refer)

        # decode
        dec_relight = self.gen.decode(L_t, en_s_embed)
        dec_relight = torch.mul(dec_relight, mask)

        # decode ref
        #move_pix = 1
        #_hdr = Variable(torch.zeros((en_t_l.size(0), 3, 16, 32)).cuda())
        #_hdr[:, :, :, :32 - int(move_pix)] = en_t_l[:, :, :, int(move_pix):]
        #_hdr[:, :, :, 32 - int(move_pix):] = en_t_l[:, :, :, :int(move_pix)]

        _hdr = en_t_l

        ref_relight = self.gen.decode(_hdr, en_s_embed)
        ref_relight = torch.mul(ref_relight, mask)

        # output
        image_anchor = I_s[0:1].detach().cpu()[:3]
        image_recons = dec_relight[0:1].detach().cpu()[:3]
        image_relight = I_t_gt[0:1].detach().cpu()[:3]
        image_relight_target = I_refer[0:1].detach().cpu()[:3]
        ref_relight = ref_relight[0:1].detach().cpu()[:3]



        self.image_display = torch.cat((image_anchor,image_relight_target,image_relight, image_recons,ref_relight),
                                       dim=3)
        self.gt_L = L_t[0:1].detach().cpu()[:3]
        self.ref_L = en_t_l[0:1].detach().cpu()[:3]


    def test_forward(self,Xa_out,Xb_out,light_b_label,mask):
        # encode
        en_s_l, en_s_embed = self.gen.encode(Xa_out)
        # decode
        dec_relight = self.gen.decode(light_b_label, en_s_embed)
        dec_relight = torch.mul(dec_relight,mask)

        ref_relight = torch.mul(dec_relight,mask) + torch.mul(Xb_out,1-mask)

        xb_rmse, xb_psnr, xb_ssim = self.recon_criterion_rmse(ref_relight, Xb_out, mask)

        image_anchor = Xa_out[0:1].detach().cpu()[:3]
        image_recons = ref_relight[0:1].detach().cpu()[:3]
        image_relight = Xb_out[0:1].detach().cpu()[:3]

        self.image_display = torch.cat((image_anchor, image_recons, image_relight),
                                       dim=3)
        return xb_rmse, xb_psnr, xb_ssim




    def resume(self, checkpoint_dir,need_opt=True,path=None):
        # Load generators
        if(path==None):
            last_model_name = get_model_list(checkpoint_dir, "gen")
        else:
            last_model_name=path
        state_dict = torch.load(last_model_name)
        self.enc.module.load_state_dict(state_dict['enc'])
        self.dec.module.load_state_dict(state_dict['edc'])
        iterations = int(last_model_name[-11:-3])
        if(need_opt):
            state_dict = torch.load(os.path.join(checkpoint_dir, 'optimizer_'+last_model_name[-11:-3]+'.pt'))
            self.gen_opt.load_state_dict(state_dict['gen'])
        print('Resume from iteration %d' % iterations)
        return iterations

    def save(self, snapshot_dir, iterations):
        # Save generators, discriminators, and optimizers
        gen_name = os.path.join(snapshot_dir, 'gen_%08d.pt' % (iterations + 1))
        opt_name = os.path.join(snapshot_dir, 'optimizer_%08d.pt'% (iterations + 1))
        torch.save({'enc': self.enc.module.state_dict(),'dec':self.dec.module.state_dict()}, gen_name)
        torch.save({'gen': self.gen_opt.state_dict()}, opt_name)

