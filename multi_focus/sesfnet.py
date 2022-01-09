import os
import torch
import torch.nn as nn
import numpy as np
import skimage
import PIL.Image
import torchvision.transforms as transforms
import torch.nn.functional as f
from skimage import morphology
from skimage.color import rgb2gray

class SESF_Fuse():
    def __init__(self, attention='cse'):
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.model = SESFuseNet(attention)
        self.model_path = os.path.join(os.getcwd(), "parameters", "model_best.pkl")
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        self.mean_value = 0.4500517361627943
        self.std_value = 0.26465333914691797
        self.data_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([self.mean_value], [self.std_value])
        ])

        self.kernel_radius = 5
        self.area_ratio = 0.01
        self.ks = 5
        self.gf_radius = 4
        self.eps = 0.1

    @staticmethod
    def box_filter(img,r):
        """
        Definition imDst(x, y)=sum(sum(imSrc(x-r:x+r,y-r:y+r)));
        :param imgSrc: np.array, image
        :param r: int, radius
        :return: imDst: np.array. result of calculation
        """
        h, w = img.shape[:2]
        imDst = np.zeros(img.shape)
        imCum = np.cumsum(img, axis=0)

        imDst[0: r + 1] = imCum[r: 2 * r + 1]
        imDst[r + 1: h - r, :] = imCum[2 * r + 1: h, :] - imCum[0: h - 2 * r - 1, :]
        imDst[h - r: h, :] = np.tile(imCum[h - 1, :], [r, 1, 1]) - imCum[h - 2 * r - 1: h - r - 1, :]

        # cumulative sum over w axis
        imCum = np.cumsum(imDst, axis=1)

        # difference over w axis
        imDst[:, 0: r + 1] = imCum[:, r: 2 * r + 1]
        imDst[:, r + 1: w - r] = imCum[:, 2 * r + 1: w] - imCum[:, 0: w - 2 * r - 1]
        imDst[:, w - r: w] = np.tile(np.expand_dims(imCum[:, w - 1], axis=1), [1, r, 1]) - imCum[:, w - 2 * r - 1: w - r - 1]
        return imDst

    def guided_filter(self, I, p, r, eps=0.1):
        """
        Guided Filter
        :param I: np.array, guided image
        :param p: np.array, input image
        :param r: int, radius
        :param eps: float
        :return: np.array, filter result
        """
        h,w = I.shape[:2]
        N = self.box_filter(np.ones((h,w,1)),r)

        mean_I, mean_p, mean_Ip = self.box_filter(I, r) / N, self.box_filter(p, r) / N, self.box_filter(I * p, r) / N
        cov_Ip = mean_Ip - mean_I * mean_p
        mean_II = self.box_filter(I * I, r) / N
        var_I = mean_II - mean_I * mean_I
        a = cov_Ip / (var_I + eps)

        b = mean_p - np.expand_dims(np.sum((a * mean_I), 2), 2)
        mean_a = self.box_filter(a, r) / N
        mean_b = self.box_filter(b, r) / N
        q = np.expand_dims(np.sum(mean_a * I, 2), 2) + mean_b
        return q
    def fuse(self, img1, img2):
        dim = img1.ndim
        img1_gray = rgb2gray(img1)
        img2_gray = rgb2gray(img2)
        img1_gray_pil = PIL.Image.fromarray(img1_gray)
        img2_gray_pil = PIL.Image.fromarray(img2_gray)
        img1_tensor = self.data_transforms(img1_gray_pil).unsqueeze(0).to(self.device)
        img2_tensor = self.data_transforms(img2_gray_pil).unsqueeze(0).to(self.device)

        dm = self.model.forward("fuse", img1_tensor, img2_tensor, kernel_radius=self.kernel_radius)
        # Morphology filter and Small region removal
        h, w = img1.shape[:2]
        se = skimage.morphology.disk(self.ks)  # 'disk' kernel with ks size for structural element
        dm = skimage.morphology.binary_opening(dm, se)
        dm = morphology.remove_small_holes(dm == 0, self.area_ratio * h * w)
        dm = np.where(dm, 0, 1)
        dm = skimage.morphology.binary_closing(dm, se)
        dm = morphology.remove_small_holes(dm == 1, self.area_ratio * h * w)
        dm = np.where(dm, 1, 0)
        # guided filter
        if dim == 3:
            dm = np.expand_dims(dm, axis=2)

        temp_fused = img1 * dm + img2 * (1 - dm)
        dm = self.guided_filter(temp_fused, dm, self.gf_radius, eps=self.eps)
        fused = img1 * 1.0 * dm + img2 * 1.0 * (1 - dm)
        fused = np.clip(fused, 0, 255).astype(np.uint8)
        return fused


class SESFuseNet(nn.Module):
    def __init__(self, attention = "cse"):
        super().__init__()
        """
        layer1: input with one channel -> output 16 channel feature map
        layer2: input 16 -> output 16
        layer3: input 32 -> output 16
        layer4: input 48 -> output 16
        """
        # encoder
        self.features = self.conv_block(in_channels=1, out_channels=16)
        self.conv_encode_1 = self.conv_block(16,16)
        self.conv_encode_2 = self.conv_block(32,16)
        self.conv_encode_3 = self.conv_block(48,16)
        if attention == "cse":
            self.se_f = CSELayer(16,8)
            self.se_1 = CSELayer(16,8)
            self.se_2 = CSELayer(16,8)
            self.se_3 = CSELayer(16,8)
        
        # decoder
        self.conv_decode_1 = self.conv_block(64,64)
        self.conv_decode_2 = self.conv_block(64,32)
        self.conv_decode_3 = self.conv_block(32,16)
        self.conv_decode_4 = self.conv_block(16,1)



    @staticmethod
    def conv_block(in_channels, out_channels, kernel_size=3):
        args = [in_channels, out_channels, kernel_size, 1, 1]
        block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels)
        )
        return block
    
    @staticmethod
    def concat(f1, f2):
        return torch.cat((f1,f2),1)
    
    def forward(self, phase, img1, img2=None, kernel_radius=5):
        if phase == "fuse":
            with torch.no_grad():
                # Image1
                features_1 = self.features(img1)
                se_features_1 = self.se_f(features_1)
                encode_block1_1 = self.conv_encode_1(se_features_1)
                se_encode_block1_1 = self.se_1(encode_block1_1)
                se_cat1_1 = self.concat(se_features_1, se_encode_block1_1)
                encode_block2_1 = self.conv_encode_2(se_cat1_1)
                se_encode_block2_1 = self.se_2(encode_block2_1)
                se_cat2_1 = self.concat(se_cat1_1, se_encode_block2_1)
                encode_block3_1 = self.conv_encode_3(se_cat2_1)
                se_encode_block3_1 = self.se_3(encode_block3_1)
                se_cat3_1 = self.concat(se_cat2_1, se_encode_block3_1)

                # Image2
                features_2 = self.features(img2)
                se_features_2 = self.se_f(features_2)
                encode_block1_2 = self.conv_encode_1(se_features_2)
                se_encode_block1_2 = self.se_2(encode_block1_2)
                se_cat1_2 = self.concat(se_features_2, se_encode_block1_2)
                encode_block2_2 = self.conv_encode_2(se_cat1_2)
                se_encode_block2_2 = self.se_2(encode_block2_2)
                se_cat2_2 = self.concat(se_cat1_2, se_encode_block2_2)
                encode_block3_2 = self.conv_encode_3(se_cat2_2)
                se_encode_block3_2 = self.se_3(encode_block3_2)
                se_cat3_2 = self.concat(se_cat2_2, se_encode_block3_2)
            output = self.fusion_channel_sf(se_cat3_1, se_cat3_2, kernel_radius=kernel_radius)
        else:
            print("Check error")
        return output
    @staticmethod
    def fusion_channel_sf(fo1, fo2, kernel_radius=3):
        device = fo1.device
        b, c, h, w = fo1.shape
        r_shift_kernel = torch.Tensor([[0, 0, 0], [1, 0, 0], [0, 0, 0]])\
            .reshape((1, 1, 3, 3)).repeat(c, 1, 1, 1)
        b_shift_kernel = torch.Tensor([[0, 1, 0], [0, 0, 0], [0, 0, 0]])\
            .reshape((1, 1, 3, 3)).repeat(c, 1, 1, 1)
        f1_r_shift = f.conv2d(fo1, r_shift_kernel, padding=1, groups=c)
        f1_b_shift = f.conv2d(fo1, b_shift_kernel, padding=1, groups=c)
        f2_r_shift = f.conv2d(fo2, r_shift_kernel, padding=1, groups=c)
        f2_b_shift = f.conv2d(fo2, b_shift_kernel, padding=1, groups=c)

        f1_grad = torch.pow((f1_r_shift - fo1), 2) + torch.pow((f1_b_shift - fo1), 2)
        f2_grad = torch.pow((f2_r_shift - fo2), 2) + torch.pow((f2_b_shift - fo2), 2)

        kernel_size = kernel_radius * 2 + 1
        add_kernel = torch.ones((c, 1, kernel_size, kernel_size)).float()
        kernel_padding = kernel_size // 2
        f1_sf = torch.sum(f.conv2d(f1_grad, add_kernel, padding=kernel_padding, groups=c), dim=1)
        f2_sf = torch.sum(f.conv2d(f2_grad, add_kernel, padding=kernel_padding, groups=c), dim=1)
        weight_zeros = torch.zeros(f1_sf.shape)
        weight_ones = torch.ones(f1_sf.shape)

        # get decision map
        dm_tensor = torch.where(f1_sf > f2_sf, weight_ones, weight_zeros)
        dm_np = dm_tensor.squeeze().cpu().numpy().astype(np.int)
        return dm_np


"""
three versions of the SE module, spatial squeeze and channel excitation (sSE), 
channel squeeze and spatial excitation (cSE), and concurrent spatial and 
channel squeeze and channel excitation (scSE), which lead the network to learn more 
meaningful spatial and/or channel- wise feature maps
"""
class CSELayer(nn.Module):
    """
    cSE uses a global average pooling layer to embed the global spatial information in a vector,
    which passes through two fully connected layers to acquire a new vector. 
    This encodes the channel-wise dependencies, which can be used to recali- brate the original feature map in the channel direction.
    """
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    def forward(self,x):
        b, c, _, _ = x.size()
        y = self.avg(x).view(b,c)
        y = self.fc(y).view(b,c,1,1)
        return x * y.expand_as(x)

class SSELayer(nn.Module):
    """
    sSE uses a convolutional layer with one 1 x 1 kernel to acquire a projection tensor.
    Each unit of the projection refers to the combined representation for all channels C at a 
    spatial location, and is used to spatially recalibrate the original feature map.
    """
    def __init__(self, channel):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel, 1, bias=False),
            nn.Sigmoid()
        )
    def forward(self,x):
        y = self.fc(x)
        return x * y

class SCSELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.CSE = CSELayer(channel, reduction=reduction)
        self.SSE = SSELayer(channel)
    
    def forward(self,x):
        SSE = self.SSE(x)
        CSE = self.CSE(x)
        return SSE+CSE
