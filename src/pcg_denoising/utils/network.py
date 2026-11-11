"""
PyTorch port of unet_architecture.py (Keras / TensorFlow 2.3).

Tensors are channels-first: (N, 1, H, W).  H and W must be divisible by 64
(six stride-2 stages); the Keras model had the same hidden requirement, it
just failed with a Concatenate shape error instead of a clear message.

Details kept identical to the Keras model:
  * Conv2D(k=5, stride=2, padding='same')  -> TF-style asymmetric padding
    (1 before, 2 after), implemented with ZeroPad2d.
  * Conv2DTranspose(k=5, stride=2, padding='same') -> full transposed conv,
    then crop 1 from the start and 2 from the end (exact adjoint of the above).
  * BatchNorm: Keras momentum 0.99 / eps 1e-3  ->  PyTorch momentum 0.01 / eps 1e-3.
  * First decoder LeakyReLU uses Keras' default slope 0.3, the rest use 0.2.
  * Skip connections concatenate the *pre-BN conv outputs* (U5..U1), as in the original.
  * he_uniform init (fan_in), zero biases.
  * Final sigmoid conv has 2 output filters, multiplied with the 1-channel input,
    so the network outputs 2 channels (N, 2, H, W).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

K = 5
OUT_CHANNELS = 1


def _bn(channels):
    return nn.BatchNorm2d(channels, eps=1e-3, momentum=0.01)


class ConvDown(nn.Module):
    """Keras Conv2D(k=5, strides=2, padding='same') for even input sizes."""

    def __init__(self, cin, cout):
        super().__init__()
        self.pad = nn.ZeroPad2d((1, 2, 1, 2))  # left, right, top, bottom
        self.conv = nn.Conv2d(cin, cout, K, stride=2)

    def forward(self, x):
        return self.conv(self.pad(x))


class ConvUp(nn.Module):
    """Keras Conv2DTranspose(k=5, strides=2, padding='same'): output = 2 * input."""

    def __init__(self, cin, cout):
        super().__init__()
        self.conv = nn.ConvTranspose2d(cin, cout, K, stride=2)

    def forward(self, x):
        y = self.conv(x)  # size 2n + 3
        return y[:, :, 1:-2, 1:-2]  # TF SAME crop -> size 2n


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        chans = [1, 16, 32, 64, 128, 256, 512]
        self.down = nn.ModuleList([ConvDown(chans[i], chans[i + 1]) for i in range(6)])
        self.down_bn = nn.ModuleList([_bn(c) for c in chans[1:]])

        # decoder input channels include the concatenated skip connection
        up_in = [512, 512, 256, 128, 64, 32]
        up_out = [256, 128, 64, 32, 16, 1]
        self.up = nn.ModuleList([ConvUp(i, o) for i, o in zip(up_in, up_out)])
        self.up_bn = nn.ModuleList([_bn(o) for o in up_out])
        self.slopes = [0.3, 0.2, 0.2, 0.2, 0.2, 0.2]
        self.dropout = nn.Dropout(0.5)  # only active in train() mode

        # Conv2D(2, (4,4), dilation_rate=2, padding='same'): effective kernel 7 -> pad 3/3
        self.mask = nn.Conv2d(1, OUT_CHANNELS, 4, dilation=2, padding=3)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                # == Keras he_uniform: U(-sqrt(6/fan_in), +sqrt(6/fan_in))
                nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x):
        h, w = x.shape[-2:]
        if h % 64 or w % 64:
            raise ValueError(
                f"Input spatial size must be divisible by 64, got {h}x{w}."
            )

        skips = []
        a = x
        for conv, bn in zip(self.down, self.down_bn):
            u = conv(a)
            skips.append(u)  # U1..U6 (pre-BN)
            a = F.relu(bn(u))

        d = a
        for i in range(6):
            u = self.up[i](d)
            a = F.leaky_relu(u, self.slopes[i])
            b = self.up_bn[i](a)
            if i < 3:  # D7, D8, D9 have dropout; B10..B12 do not
                b = self.dropout(b)
            if i < 5:
                d = torch.cat([skips[4 - i], b], dim=1)  # U5, U4, U3, U2, U1
            else:
                d = b

        mask = torch.sigmoid(self.mask(d))  # (N, 2, H, W)
        return mask * x  # broadcasts over the single input channel


def build_model(input_shape=None):
    """Kept for API compatibility. The network is fully convolutional, so
    input_shape is not needed."""
    return UNet()