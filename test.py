from utils import test_write_images,get_config,prepare_sub_folder
from torch.utils.data import DataLoader
from models import Models
from dataset import My3DDataset
import torch.backends.cudnn as cudnn
import torch
import os
from torch.autograd import Variable
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
class Tester():
    def __init__(self):
        cudnn.benchmark = True
        # Load experiment setting
        config = get_config('configs/OT3_rotate.yaml')
        self.trainer = Models(config)
        self.trainer.cuda()

        # Setup logger and output folders

        self.trainer.resume('outputs/OT3_rotate/checkpoints', need_opt=False)#,path='pretrain_model/gen_00000000.pt')
        self.trainer.eval()
        self.config = config

        self.dataset = My3DDataset(opts=self.config, is_Train=False)
        self.test_loader = DataLoader(dataset=self.dataset, batch_size=self.config['batch_size']*5, shuffle=False,
                                      num_workers=self.config['nThreads'])

    def eval_all(self):
        xb_rmse, xb_psnr, xb_ssim=0,0,0
        count=0

        if(not os.path.exists('outputs/%s/images/test_sample' % (self.name_))):
            os.mkdir('outputs/%s/images/test_sample' % (self.name_))

        with torch.no_grad():
            for it, out_data in enumerate(self.test_loader):
                for j in range(len(out_data)):
                    out_data[j] = out_data[j].cuda().detach()
                Xa_out, Xb_out, light_a_label, light_b_label, r_Xb_out, mask, diff = out_data

                tmp_xb_rmse, tmp_xb_psnr, tmp_xb_ssim = self.trainer.test_forward(Xa_out,Xb_out,light_b_label,mask)

                xb_rmse += tmp_xb_rmse
                xb_psnr += tmp_xb_psnr
                xb_ssim += tmp_xb_ssim
                count+=1
                print(it)
                if (it % 20 == 0):
                   test_write_images(self.trainer.image_display, 1,
                                     'outputs/%s/images/test_sample/%d.jpg' % (self.name_,it / 20))
            print('final xb_rmse = %f' %(xb_rmse/count))
            print('final xb_psnr = %f' %(xb_psnr/count))
            print('final xb_ssim = %f' %(xb_ssim/count))







    def upload(self,img, mask):
        img = torch.unsqueeze(img, dim=0)
        mask = torch.unsqueeze(mask, dim=0)
        img = img.cuda(self.config['cuda_device']).detach()
        mask = mask.cuda(self.config['cuda_device']).detach()
        return img, mask

    def test_real_rotation(self):
        self.dataset.scale = 2
        with torch.no_grad():
            for j in range(60):
                for i in range(60):
                    if(j!=12 or i!=57):
                        continue
                    source_img, source_mask = self.dataset.getReal2(j)
                    source_img, source_mask = self.upload(source_img, source_mask)

                    # add
                    target_img_, target_mask_ = self.dataset.getReal2(i)
                    target_img_, target_mask_ = self.upload(target_img_, target_mask_)

                    ref_path = '/media/guoxian/D/Real_Image/' + self.dataset.real_files[i] + '.jpg'
                    ref_mask_path = '/media/guoxian/D/Real_Image/' + self.dataset.real_files[i] + '.tif'
                    ref_bg = self.dataset.inpaint2(0, ref_path, ref_mask_path)
                    ref_bg = torch.unsqueeze(ref_bg, dim=0)
                    ref_bg = ref_bg.cuda().detach()

                    en_s_l, en_s_embed = self.trainer.gen.encode(source_img)
                    en_t_l, en_t_embed = self.trainer.gen.encode(target_img_)
                    out = [source_img[0:1].detach().cpu()[:3], target_img_[0:1].detach().cpu()[:3]]
                    for s in range(12):
                        _hdr = Variable(torch.zeros((en_t_l.size(0), 3, 16, 32)).cuda())
                        move_pix = abs(16 - int(s * 16 / 6))
                        if (s < 6):
                            _hdr[:, :, :, :move_pix] = en_t_l[:, :, :, 32 - move_pix:]
                            _hdr[:, :, :, move_pix:] = en_t_l[:, :, :, :32 - move_pix]
                        elif (s == 6):
                            move_pix = 1
                            _hdr[:, :, :, :32 - int(move_pix)] = en_t_l[:, :, :, int(move_pix):]
                            _hdr[:, :, :, 32 - int(move_pix):] = en_t_l[:, :, :, :int(move_pix)]
                        else:
                            _hdr[:, :, :, :32 - int(move_pix)] = en_t_l[:, :, :, int(move_pix):]
                            _hdr[:, :, :, 32 - int(move_pix):] = en_t_l[:, :, :, :int(move_pix)]

                        ref_relight = self.trainer.gen.decode(_hdr, en_s_embed)
                        if(s==6):
                            ref_relight = torch.mul(ref_relight, source_mask)+ torch.mul(ref_bg, 1 - source_mask)
                        else:
                            ref_relight = torch.mul(ref_relight, source_mask)
                        out.append(ref_relight[0:1].detach().cpu()[:3])

                    img = torch.cat(tuple(out), dim=3)

                    test_write_images(img, 1, 'tmp/real_rotation/' + str(j) + '_' + str(i) + '.png')



    def save_synthetic(self):
        with torch.no_grad():
            for i in range(240):
                out_data = self.dataset.__getitem__(i*100)
                data=[]
                for j in range(len(out_data)):
                    data.append(torch.unsqueeze(out_data[j].cuda(self.config['cuda_device']).detach(),dim=0))
                I_s,I_t_gt,I_refer,mask,_,L_s,L_t,solid_mask = data
                _,_,_,tmp_xb_rmse, tmp_xb_psnr, tmp_xb_ssim = \
                    self.trainer.test_forward(I_s, I_t_gt, I_refer, mask, L_s, L_t, solid_mask)
                test_write_images(self.trainer.image_display, 1,'tmp/synthetic_pair/%d.jpg' % (i))

tester = Tester()
tester.eval_all()
