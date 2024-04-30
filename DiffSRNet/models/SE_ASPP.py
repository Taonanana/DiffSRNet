import torch
import torch.nn as nn
import torch.nn.functional as F

class SE_Block(nn.Module):                         # Squeeze-and-Excitation block
    def __init__(self, in_planes):
        super(SE_Block, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv1 = nn.Conv2d(in_planes, in_planes // 16, kernel_size=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_planes // 16, in_planes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # print('SE', x.shape)
        x = self.avgpool(x)
        # print('SE_G_x', x.shape)
        x = self.conv1(x)
        # print('SE_con1_x', x.shape)
        x = self.relu(x)
        x = self.conv2(x)
        # print('SE_con2_x', x.shape)
        out = self.sigmoid(x)
        # print('SE_sigmoid_x', x.shape)
        return out

class SE_ASPP(nn.Module):                       ##加入通道注意力机制
    def __init__(self, dim_in, dim_out, rate=1, bn_mom=0.1):
        super(SE_ASPP, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 1, 1, padding=0, dilation=rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=6 * rate, dilation=6 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=12 * rate, dilation=12 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=18 * rate, dilation=18 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch5_conv = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=True)
        self.branch5_bn = nn.BatchNorm2d(dim_out, momentum=bn_mom)
        self.branch5_relu = nn.ReLU(inplace=True)

        self.conv_cat = nn.Sequential(
            nn.Conv2d(dim_out * 5, dim_out, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        print('dim_in:',dim_in)
        print('dim_out:',dim_out)
        self.senet=SE_Block(in_planes=dim_out*5)

    def forward(self, x):
        [b, c, row, col] = x.size()
        # print('x', x.shape)
        conv1x1 = self.branch1(x)
        conv3x3_1 = self.branch2(x)
        conv3x3_2 = self.branch3(x)
        conv3x3_3 = self.branch4(x)
        global_feature = torch.mean(x, 2, True)
        # print('global_feature_mean1', global_feature.shape)
        
        global_feature = torch.mean(global_feature, 3, True)
        # print('global_feature_mean2', global_feature.shape)
        global_feature = self.branch5_conv(global_feature)
        # print('global_feature_conv', global_feature.shape)
        global_feature = self.branch5_bn(global_feature)
        # print('global_feature_bn', global_feature.shape)
        global_feature = self.branch5_relu(global_feature)
        # print('global_feature_relu', global_feature.shape)
        global_feature = F.interpolate(global_feature, (row, col), None, 'bilinear', True)
        # print('global_feature_F', global_feature.shape)

        feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature], dim=1)
        # print('conv1x1', conv1x1.shape)
        # print('conv3x3_1', conv3x3_1.shape)
        # print('conv3x3_2', conv3x3_2.shape)
        # print('conv3x3_3', conv3x3_3.shape)
        
        # print('global_feature', global_feature.shape)
        # print('feature_cat:',feature_cat.shape)
        seaspp1=self.senet(feature_cat)             
        # print('seaspp1:',seaspp1.shape)
        se_feature_cat=seaspp1*feature_cat
        # print('se_feature_cat', se_feature_cat.shape)
        result = self.conv_cat(se_feature_cat)
        # result = self.conv_cat(feature_cat)
        print('result:',result.shape)
        return result

