import torch
import torch.nn as nn


def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )


class Conv2dBlock(nn.Module):
    def __init__(self, input_dim ,output_dim, kernel_size=3, stride=1,
                 padding=1, norm='gn', activation='prelu', pad_type='zero'):
        super(Conv2dBlock, self).__init__()
        self.use_bias = True
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'gn':
            self.norm = nn.GroupNorm(norm_dim,norm_dim)
        elif norm == 'none' or norm == 'sn':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'softplus':
            self.activation = nn.Softplus()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias)

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x

# input_dim ,output_dim, kernel_size, stride,
# padding=0, norm='none', activation='relu', pad_type='zero'):

class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_down1_1 = Conv2dBlock(3, 29,7,padding=3)
        self.conv_down1_2 = Conv2dBlock(32, 64,stride=2)
        self.conv_down2_1 = Conv2dBlock(64, 64)
        self.conv_down2_2 = Conv2dBlock(64, 128,stride=2)
        self.conv_down3_1 = Conv2dBlock(128, 128)
        self.conv_down3_2 = Conv2dBlock(128, 256,stride=2)
        self.conv_down4_1 = Conv2dBlock(256, 256)
        self.conv_down4_2 = Conv2dBlock(256, 512,stride=2)

        self.conv_down5_1 = Conv2dBlock(512, 512)
        self.conv_down5_2 = Conv2dBlock(512, 512)
        self.conv_down5_3 = Conv2dBlock(512, 512)
        self.conv_down5_4 = Conv2dBlock(512, 512, stride=2,activation='softplus')

        self.adapool = nn.AdaptiveAvgPool2d(2)


        self.upsample = nn.Upsample(scale_factor=2)

        self.conv_up5_3 = Conv2dBlock(512,256)
        self.conv_up5_2 = Conv2dBlock(512+256, 512)
        self.conv_up5_1 = Conv2dBlock(512+512, 512)

        self.conv_up4_2 = nn.Sequential(*[nn.Upsample(scale_factor=2),Conv2dBlock(512+512, 256)])
        self.conv_up4_1 = Conv2dBlock(256+256, 256)

        self.conv_up3_2 = nn.Sequential(*[nn.Upsample(scale_factor=2),Conv2dBlock(256+256, 128)])
        self.conv_up3_1 = Conv2dBlock(128+128, 128)

        self.conv_up2_2 = nn.Sequential(*[nn.Upsample(scale_factor=2),Conv2dBlock(128+128, 64)])
        self.conv_up2_1 = Conv2dBlock(64+64, 64)

        self.conv_up1_2 = nn.Sequential(*[nn.Upsample(scale_factor=2),Conv2dBlock(64 + 64, 32)])
        self.conv_up1_1 = Conv2dBlock(32 + 32, 3,activation='sigmoid')


    def encode(self,x):

        conv1_1 = torch.cat((self.conv_down1_1(x),x),dim=1)
        conv1_2 = self.conv_down1_2(conv1_1)

        conv2_1 = self.conv_down2_1(conv1_2)
        conv2_2 = self.conv_down2_2(conv2_1)

        conv3_1 = self.conv_down3_1(conv2_2)
        conv3_2 = self.conv_down3_2(conv3_1)

        conv4_1 = self.conv_down4_1(conv3_2)
        conv4_2 = self.conv_down4_2(conv4_1)

        conv5_1 = self.conv_down5_1(conv4_2)
        conv5_2 = self.conv_down5_2(conv5_1)
        conv5_3 = self.conv_down5_3(conv5_2)
        conv5_4 = self.conv_down5_4(conv5_3)

        adapool_out = self.adapool(conv5_4)

        embed = [conv1_1,conv1_2,conv2_1,conv2_2,conv3_1,conv3_2,conv4_1,conv4_2,
                 conv5_1,conv5_2]
        adapool_out = adapool_out.view(adapool_out.size(0),adapool_out.size(1),-1)

        source_light  = adapool_out[:,:,:3]*adapool_out[:,:,3].unsqueeze(-1)
        source_light = source_light.permute((0,2,1))
        return source_light.view(source_light.size(0),3,16,32)  ,embed

    def decode(self,source_light,embed): # b x 3 x 16 x 32
        #tilling
        source_light = source_light.view(source_light.size(0),3,16*32).permute((0,2,1)) # b, 512,3
        source_light =source_light.repeat((1,1,342))[:,:,:1024]
        source_light = source_light.view(source_light.size(0),512,32,32)

        dconv_up5_3 = torch.cat((embed[9],self.conv_up5_3(source_light)),dim=1)
        dconv_up5_2 = torch.cat((embed[8], self.conv_up5_2(dconv_up5_3)), dim=1)
        dconv_up5_1 = torch.cat((embed[7], self.conv_up5_1(dconv_up5_2)), dim=1)

        dconv_up4_2 = torch.cat((embed[6], self.conv_up4_2(dconv_up5_1)), dim=1)
        dconv_up4_1 = torch.cat((embed[5], self.conv_up4_1(dconv_up4_2)), dim=1)

        dconv_up3_2 = torch.cat((embed[4], self.conv_up3_2(dconv_up4_1)), dim=1)
        dconv_up3_1 = torch.cat((embed[3], self.conv_up3_1(dconv_up3_2)), dim=1)

        dconv_up2_2 = torch.cat((embed[2], self.conv_up2_2(dconv_up3_1)), dim=1)
        dconv_up2_1 = torch.cat((embed[1], self.conv_up2_1(dconv_up2_2)), dim=1)

        dconv_up1_2 = torch.cat((embed[0], self.conv_up1_2(dconv_up2_1)), dim=1)
        dconv_up1_1 = self.conv_up1_1(dconv_up1_2)

        return dconv_up1_1





class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv_down1_1 = Conv2dBlock(3, 29, 7, padding=3)
        self.conv_down1_2 = Conv2dBlock(32, 64, stride=2)
        self.conv_down2_1 = Conv2dBlock(64, 64)
        self.conv_down2_2 = Conv2dBlock(64, 128, stride=2)
        self.conv_down3_1 = Conv2dBlock(128, 128)
        self.conv_down3_2 = Conv2dBlock(128, 256, stride=2)
        self.conv_down4_1 = Conv2dBlock(256, 256)
        self.conv_down4_2 = Conv2dBlock(256, 512, stride=2)

        self.conv_down5_1 = Conv2dBlock(512, 512)
        self.conv_down5_2 = Conv2dBlock(512, 512)
        self.conv_down5_3 = Conv2dBlock(512, 512)
        self.conv_down5_4 = Conv2dBlock(512, 512, stride=2, activation='softplus')
        self.adapool = nn.AdaptiveAvgPool2d(2)

    def forward(self, x):
        conv1_1 = torch.cat((self.conv_down1_1(x), x), dim=1)
        conv1_2 = self.conv_down1_2(conv1_1)

        conv2_1 = self.conv_down2_1(conv1_2)
        conv2_2 = self.conv_down2_2(conv2_1)

        conv3_1 = self.conv_down3_1(conv2_2)
        conv3_2 = self.conv_down3_2(conv3_1)

        conv4_1 = self.conv_down4_1(conv3_2)
        conv4_2 = self.conv_down4_2(conv4_1)

        conv5_1 = self.conv_down5_1(conv4_2)
        conv5_2 = self.conv_down5_2(conv5_1)
        conv5_3 = self.conv_down5_3(conv5_2)
        conv5_4 = self.conv_down5_4(conv5_3)

        adapool_out = self.adapool(conv5_4)

        embed = [conv1_1, conv1_2, conv2_1, conv2_2, conv3_1, conv3_2, conv4_1, conv4_2,
                 conv5_1, conv5_2]
        adapool_out = adapool_out.view(adapool_out.size(0), adapool_out.size(1), -1)

        source_light = adapool_out[:, :, :3] * adapool_out[:, :, 3].unsqueeze(-1)
        source_light = source_light.permute((0, 2, 1))
        return source_light.view(source_light.size(0), 3, 16, 32), embed

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2)
        self.conv_up5_3 = Conv2dBlock(512,256)
        self.conv_up5_2 = Conv2dBlock(512+256, 512)
        self.conv_up5_1 = Conv2dBlock(512+512, 512)

        self.conv_up4_2 = nn.Sequential(*[nn.Upsample(scale_factor=2),Conv2dBlock(512+512, 256)])
        self.conv_up4_1 = Conv2dBlock(256+256, 256)

        self.conv_up3_2 = nn.Sequential(*[nn.Upsample(scale_factor=2),Conv2dBlock(256+256, 128)])
        self.conv_up3_1 = Conv2dBlock(128+128, 128)

        self.conv_up2_2 = nn.Sequential(*[nn.Upsample(scale_factor=2),Conv2dBlock(128+128, 64)])
        self.conv_up2_1 = Conv2dBlock(64+64, 64)

        self.conv_up1_2 = nn.Sequential(*[nn.Upsample(scale_factor=2),Conv2dBlock(64 + 64, 32)])
        self.conv_up1_1 = Conv2dBlock(32 + 32, 3,activation='sigmoid')


    def forward(self, source_light, embed):  # b x 3 x 16 x 32
        # tilling
        source_light = source_light.view(source_light.size(0), 3, 16 * 32).permute((0, 2, 1))  # b, 512,3
        source_light = source_light.repeat((1, 1, 342))[:, :, :1024]
        source_light = source_light.view(source_light.size(0), 512, 32, 32)

        dconv_up5_3 = torch.cat((embed[9], self.conv_up5_3(source_light)), dim=1)
        dconv_up5_2 = torch.cat((embed[8], self.conv_up5_2(dconv_up5_3)), dim=1)
        dconv_up5_1 = torch.cat((embed[7], self.conv_up5_1(dconv_up5_2)), dim=1)

        dconv_up4_2 = torch.cat((embed[6], self.conv_up4_2(dconv_up5_1)), dim=1)
        dconv_up4_1 = torch.cat((embed[5], self.conv_up4_1(dconv_up4_2)), dim=1)

        dconv_up3_2 = torch.cat((embed[4], self.conv_up3_2(dconv_up4_1)), dim=1)
        dconv_up3_1 = torch.cat((embed[3], self.conv_up3_1(dconv_up3_2)), dim=1)

        dconv_up2_2 = torch.cat((embed[2], self.conv_up2_2(dconv_up3_1)), dim=1)
        dconv_up2_1 = torch.cat((embed[1], self.conv_up2_1(dconv_up2_2)), dim=1)

        dconv_up1_2 = torch.cat((embed[0], self.conv_up1_2(dconv_up2_1)), dim=1)
        dconv_up1_1 = self.conv_up1_1(dconv_up1_2)

        return dconv_up1_1