import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Frontend(nn.Module):
    def __init__(self, L, rms_norm=True):
        super().__init__()
        self.L, self.rms_norm = L, rms_norm

    def forward(self, x):
        x = x[..., :self.L]
        if self.rms_norm:
            x = x * torch.rsqrt(x.pow(2).mean(dim=(1, 2), keepdim=True) + 1e-8)
        return x


def conv_bn(c_in, c_out, k, stride=1, dilation=1):
    return nn.Sequential(
        nn.Conv1d(c_in, c_out, k, stride=stride, padding=dilation * (k - 1) // 2, dilation=dilation, bias=False),
        nn.BatchNorm1d(c_out), nn.ReLU(inplace=True))


def dense_head(d_in, n_classes, fc=128, dropout=0.3):
    return nn.Sequential(nn.Flatten(), nn.Linear(d_in, fc), nn.SELU(), nn.AlphaDropout(dropout),
                         nn.Linear(fc, fc), nn.SELU(), nn.AlphaDropout(dropout), nn.Linear(fc, n_classes))


def run_rnn(rnn, h):
    with torch.autocast(device_type='cuda' if h.is_cuda else 'cpu', enabled=False):
        out, _ = rnn(h.float())
    return out


class MLP(nn.Module):
    def __init__(self, L, n_classes, hidden=512, depth=2, dropout=0.3):
        super().__init__()
        layers, d = [nn.Flatten()], 2 * L
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.BatchNorm1d(hidden), nn.ReLU(inplace=True), nn.Dropout(dropout)]
            d = hidden
        self.net = nn.Sequential(*layers, nn.Linear(d, n_classes))

    def forward(self, x):
        return self.net(x)


class SimpleCNN(nn.Module):
    def __init__(self, L, n_classes, width=64, n_conv=2, kernel=7, fc=128, dropout=0.5):
        super().__init__()
        layers, c = [], 2
        for _ in range(n_conv):
            layers += [conv_bn(c, width, kernel), nn.MaxPool1d(4)]
            c = width
        self.features = nn.Sequential(*layers, nn.AdaptiveAvgPool1d(8))
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(8 * c, fc), nn.ReLU(inplace=True),
                                  nn.Dropout(dropout), nn.Linear(fc, n_classes))

    def forward(self, x):
        return self.head(self.features(x))


class VGG1D(nn.Module):
    def __init__(self, L, n_classes, width=64, n_blocks=7, fc=128, dropout=0.3):
        super().__init__()
        n_blocks = min(n_blocks, int(math.log2(L)) - 1)
        layers, c = [], 2
        for _ in range(n_blocks):
            layers += [conv_bn(c, width, 3), nn.MaxPool1d(2)]
            c = width
        self.features = nn.Sequential(*layers)
        self.head = dense_head(c * (L >> n_blocks), n_classes, fc, dropout)

    def forward(self, x):
        return self.head(self.features(x))


class ResUnit(nn.Module):
    def __init__(self, c, k=3, dilation=1):
        super().__init__()
        pad = dilation * (k - 1) // 2
        self.body = nn.Sequential(
            nn.Conv1d(c, c, k, padding=pad, dilation=dilation, bias=False), nn.BatchNorm1d(c), nn.ReLU(inplace=True),
            nn.Conv1d(c, c, k, padding=pad, dilation=dilation, bias=False), nn.BatchNorm1d(c))

    def forward(self, x):
        return F.relu(x + self.body(x))


class ResNet1D(nn.Module):
    def __init__(self, L, n_classes, width=32, n_stacks=6, fc=128, dropout=0.3):
        super().__init__()
        n_stacks = min(n_stacks, int(math.log2(L)) - 1)
        layers, c = [], 2
        for _ in range(n_stacks):
            layers += [nn.Conv1d(c, width, 1), ResUnit(width), ResUnit(width), nn.MaxPool1d(2)]
            c = width
        self.features = nn.Sequential(*layers)
        self.head = dense_head(c * (L >> n_stacks), n_classes, fc, dropout)

    def forward(self, x):
        return self.head(self.features(x))


class DilatedCNN(nn.Module):
    def __init__(self, L, n_classes, width=64, dilations=(1, 2, 4, 8, 16, 32), dropout=0.3):
        super().__init__()
        self.features = nn.Sequential(conv_bn(2, width, 7, stride=2), *[ResUnit(width, 3, d) for d in dilations],
                                      nn.AdaptiveAvgPool1d(1))
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(width, n_classes))

    def forward(self, x):
        return self.head(self.features(x))


class InceptionBlock(nn.Module):
    def __init__(self, c_in, c_out, kernels=(3, 9, 19)):
        super().__init__()
        cb = c_out // (len(kernels) + 1)
        self.branches = nn.ModuleList([nn.Conv1d(c_in, cb, k, padding=k // 2, bias=False) for k in kernels])
        self.pool = nn.Sequential(nn.MaxPool1d(3, stride=1, padding=1),
                                  nn.Conv1d(c_in, c_out - cb * len(kernels), 1, bias=False))
        self.bn = nn.BatchNorm1d(c_out)
        self.skip = nn.Identity() if c_in == c_out else nn.Conv1d(c_in, c_out, 1, bias=False)

    def forward(self, x):
        y = torch.cat([b(x) for b in self.branches] + [self.pool(x)], dim=1)
        return F.relu(self.bn(y) + self.skip(x))


class InceptionNet(nn.Module):
    def __init__(self, L, n_classes, width=64, n_blocks=4, dropout=0.3):
        super().__init__()
        layers = [conv_bn(2, width, 7)]
        for _ in range(n_blocks):
            layers += [InceptionBlock(width, width), nn.MaxPool1d(2)]
        self.features = nn.Sequential(*layers, nn.AdaptiveAvgPool1d(1))
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(width, n_classes))

    def forward(self, x):
        return self.head(self.features(x))


class CLDNN(nn.Module):
    def __init__(self, L, n_classes, width=64, n_conv=3, hidden=128, dropout=0.3):
        super().__init__()
        layers, c = [], 2
        for _ in range(n_conv):
            layers += [conv_bn(c, width, 7), nn.MaxPool1d(2)]
            c = width
        self.conv = nn.Sequential(*layers)
        self.rnn = nn.LSTM(width, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden, 128), nn.ReLU(inplace=True), nn.Dropout(dropout),
                                  nn.Linear(128, n_classes))

    def forward(self, x):
        h = run_rnn(self.rnn, self.conv(x).transpose(1, 2))
        return self.head(h[:, -1])


class LSTMAP(nn.Module):
    def __init__(self, L, n_classes, hidden=128, layers=2, dropout=0.2):
        super().__init__()
        self.rnn = nn.LSTM(2, hidden, num_layers=layers, batch_first=True, dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Linear(hidden, n_classes)

    def forward(self, x):
        amp = torch.sqrt(x[:, 0] ** 2 + x[:, 1] ** 2 + 1e-12)
        phase = torch.atan2(x[:, 1], x[:, 0]) / math.pi
        h = run_rnn(self.rnn, torch.stack([amp, phase], dim=-1))
        return self.head(h[:, -1])


class ConvGRU(nn.Module):
    def __init__(self, L, n_classes, width=64, stride=4, hidden=96, dropout=0.3):
        super().__init__()
        self.conv = nn.Sequential(conv_bn(2, width, 2 * stride + 1, stride=stride), conv_bn(width, width, 5))
        self.rnn = nn.GRU(width, hidden, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * hidden, n_classes))

    def forward(self, x):
        h = run_rnn(self.rnn, self.conv(x).transpose(1, 2))
        return self.head(h.mean(dim=1))


class PatchTransformer(nn.Module):
    def __init__(self, L, n_classes, patch=16, d_model=128, n_layers=4, n_heads=4, dropout=0.1):
        super().__init__()
        self.embed = nn.Conv1d(2, d_model, patch, stride=patch)
        self.pos = nn.Parameter(0.02 * torch.randn(1, L // patch, d_model))
        layer = nn.TransformerEncoderLayer(d_model, n_heads, 2 * d_model, dropout, activation='gelu',
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x):
        h = self.encoder(self.embed(x).transpose(1, 2) + self.pos)
        return self.head(self.norm(h.mean(dim=1)))


ARCHS = {'mlp': MLP, 'cnn': SimpleCNN, 'vgg': VGG1D, 'resnet': ResNet1D, 'tcn': DilatedCNN,
         'inception': InceptionNet, 'cldnn': CLDNN, 'lstm': LSTMAP, 'gru': ConvGRU,
         'transformer': PatchTransformer}


def build_model(arch, L, n_classes=24, rms_norm=True, **kw):
    assert L & (L - 1) == 0, 'input length L must be a power of two'
    return nn.Sequential(Frontend(L, rms_norm), ARCHS[arch](L, n_classes, **kw))
