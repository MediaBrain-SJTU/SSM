import torch
from torch import nn
import torch.nn.functional as F
from torchsummary import summary


def conv(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)


def upconv2x2(in_channels, out_channels, mode='transpose'):
    if mode == 'transpose':
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
    else:
        return conv(in_channels, out_channels, kernel_size=1, stride=1, padding=0)


class UNetDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(UNetDownBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.conv1 = conv(self.in_channels, self.out_channels, kernel_size=self.kernel_size, stride=self.stride)
        self.bn1 = nn.BatchNorm2d(self.out_channels)
        self.relu1 = nn.ReLU()

        self.conv2 = conv(self.out_channels, self.out_channels)
        self.bn2 = nn.BatchNorm2d(self.out_channels)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))

        return x


class UNetUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, merge_mode='concat', up_mode='transpose'):
        super(UNetUpBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.merge_mode = merge_mode
        self.up_mode = up_mode

        self.upconv = upconv2x2(self.in_channels, self.out_channels, mode=self.up_mode)

        if self.merge_mode == 'concat':
            self.conv1 = conv(2 * self.out_channels, self.out_channels)
        else:
            self.conv1 = conv(self.out_channels, self.out_channels)
        self.bn1 = nn.BatchNorm2d(self.out_channels)
        self.relu1 = nn.ReLU()
        self.conv2 = conv(self.out_channels, self.out_channels)
        self.bn2 = nn.BatchNorm2d(self.out_channels)
        self.relu2 = nn.ReLU()

    def forward(self, from_up, from_down):
        if self.up_mode != 'transpose':
            from_up = F.interpolate(from_up, scale_factor=2, mode='nearest')
        from_up = self.upconv(from_up)

        if self.merge_mode == 'concat':
            x = torch.cat((from_up, from_down), 1)
        else:
            x = from_up + from_down
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))

        return x

class UNetMaskBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetMaskBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.conv2 = conv(self.in_channels+1, self.out_channels)
        self.bn2 = nn.BatchNorm2d(self.out_channels)
        self.relu2 = nn.ReLU()
        self.conv3 = conv(self.out_channels, self.out_channels)
        #self.bn2 = nn.BatchNorm2d(self.out_channels)
        #self.relu2 = nn.ReLU()

    def forward(self, from_up, mask):
        B, C, H, W = from_up.shape
        mask = F.interpolate(mask, size=[H,W], mode='nearest')
        x = self.bn1(from_up)
        x = torch.cat((x, mask), 1)
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.conv3(x)
        x = mask * x
        x = x + from_up
        return x


class UNet(nn.Module):
    def __init__(self, n_channels=3, merge_mode='concat', up_mode='transpose'):
        super(UNet, self).__init__()
        self.n_chnnels = n_channels
        self.merge_mode = merge_mode
        self.up_mode = up_mode

        self.down1 = UNetDownBlock(self.n_chnnels, 64, 3, 1, 1)
        self.down2 = UNetDownBlock(64, 128, 4, 2, 1)
        self.down3 = UNetDownBlock(128, 256, 4, 2, 1)
        self.down4 = UNetDownBlock(256, 512, 4, 2, 1)
        self.down5 = UNetDownBlock(512, 512, 4, 2, 1)

        self.up1 = UNetUpBlock(512, 512, merge_mode=self.merge_mode, up_mode=self.up_mode)
        self.up2 = UNetUpBlock(512, 256, merge_mode=self.merge_mode, up_mode=self.up_mode)
        self.up3 = UNetUpBlock(256, 128, merge_mode=self.merge_mode, up_mode=self.up_mode)
        self.up4 = UNetUpBlock(128, 64, merge_mode=self.merge_mode, up_mode=self.up_mode)

        self.mask1 = UNetMaskBlock(512, 512)
        self.mask2 = UNetMaskBlock(512, 512)
        self.mask3 = UNetMaskBlock(256, 256)
        self.mask4 = UNetMaskBlock(128, 128)
        self.mask5 = UNetMaskBlock(64, 64)

        self.conv_final = nn.Sequential(conv(64, 4, 3, 1, 1), nn.Tanh())

    def forward(self, x):
        mask = x[:,-1,:,:].unsqueeze(1)
        img = x[:,:3,:,:]
        x1 = self.down1(img)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)

        x5 = self.mask1(x5,mask)
        x = self.up1(x5, x4)
        x = self.mask2(x,mask)
        x = self.up2(x, x3)
        x = self.mask3(x,mask)
        x = self.up3(x, x2)
        x = self.mask4(x,mask)
        x = self.up4(x, x1)
        x = self.mask5(x,mask)
        x = self.conv_final(x)
        return x[:,:3,:,:], x[:,3:,:,:]


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet().to(device)
    summary(model, (3, 256, 256))
