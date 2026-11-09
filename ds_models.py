import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
POS_SCALE = 50.0


def position_features(pos, scen, scenarios):
    x, y = pos[:, 0].astype(np.float64), pos[:, 1].astype(np.float64)
    theta = np.arctan2(y, x)
    one_hot = (np.asarray(scen)[:, None] == np.asarray(scenarios)[None, :]).astype(np.float64)
    feats = np.column_stack([x / POS_SCALE, y / POS_SCALE, np.hypot(x, y) / POS_SCALE, np.sin(theta), np.cos(theta),
                             one_hot])
    return feats.astype(np.float32)


def backbone(arch, pretrained):
    try:
        import torchvision
    except ImportError as e:
        raise SystemExit('torchvision is required for the vision / fusion models (see README)') from e
    weights = torchvision.models.get_model_weights(arch).DEFAULT if pretrained else None
    net = torchvision.models.get_model(arch, weights=weights)
    name, last = [(n, m) for n, m in net.named_modules() if isinstance(m, nn.Linear)][-1]
    parent, _, attr = name.rpartition('.')
    setattr(net.get_submodule(parent) if parent else net, attr, nn.Identity())
    return net, last.in_features


def color_jitter(x, strength=0.2):
    B = x.shape[0]
    r = lambda: 1 + strength * (2 * torch.rand(B, 1, 1, 1, device=x.device) - 1)
    gray = x.mean(1, keepdim=True)
    x = gray + (x - gray) * r()
    mean = x.mean((1, 2, 3), keepdim=True)
    x = mean + (x - mean) * r()
    return (x * r()).clamp(0, 1)


class VisionNet(nn.Module):
    def __init__(self, arch, n_beams, size, pretrained=True, pos_dim=0, pos_hidden=64, jitter=True):
        super().__init__()
        self.body, feat = backbone(arch, pretrained)
        self.size, self.jitter = tuple(size), jitter
        self.register_buffer('mean', torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))
        self.pos = None
        if pos_dim:
            self.pos = nn.Sequential(nn.Linear(pos_dim, pos_hidden), nn.ReLU(), nn.Linear(pos_hidden, pos_hidden),
                                     nn.ReLU())
            feat += pos_hidden
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(feat, n_beams))

    def forward(self, img, pos=None):
        x = img.permute(0, 3, 1, 2).float().div_(255)
        if tuple(x.shape[-2:]) != self.size:
            x = F.interpolate(x, size=self.size, mode='bilinear', align_corners=False, antialias=True)
        if self.training and self.jitter:
            x = color_jitter(x)
        z = self.body((x - self.mean) / self.std)
        if self.pos is not None:
            z = torch.cat([z, self.pos(pos)], 1)
        return self.head(z)


class PosMLP(nn.Module):
    def __init__(self, in_dim, n_beams, hidden=256, depth=3, dropout=0.1):
        super().__init__()
        layers, d = [], in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(dropout)]
            d = hidden
        self.net = nn.Sequential(*layers, nn.Linear(d, n_beams))

    def forward(self, img, pos):
        return self.net(pos)


def build_neural(run, n_beams, pos_dim, pretrained):
    if run['family'] == 'mlp':
        return PosMLP(pos_dim, n_beams, **run['kw'])
    return VisionNet(run['arch'], n_beams, run['size'], pretrained=pretrained,
                     pos_dim=pos_dim if run['family'] == 'fusion' else 0, **run['kw'])


def angle(pos):
    return np.arctan2(pos[:, 1], pos[:, 0])


class AngleLUT:
    def __init__(self, bins, n_beams):
        self.bins, self.n_beams = bins, n_beams

    def fit(self, X, y):
        a = angle(X)
        self.edges = np.linspace(a.min(), a.max() + 1e-9, self.bins + 1)
        b = np.clip(np.searchsorted(self.edges, a, side='right') - 1, 0, self.bins - 1)
        counts = np.zeros((self.bins, self.n_beams))
        np.add.at(counts, (b, y), 1)
        filled = np.flatnonzero(counts.sum(1) > 0)
        nearest = filled[np.abs(np.arange(self.bins)[:, None] - filled[None, :]).argmin(1)]
        self.table = counts[nearest] / counts[nearest].sum(1, keepdims=True)
        self.classes_ = np.arange(self.n_beams)
        return self

    def predict_proba(self, X):
        b = np.clip(np.searchsorted(self.edges, angle(X), side='right') - 1, 0, self.bins - 1)
        return self.table[b]


def classic_features(method, pos):
    if method in ('knn', 'gnb', 'lut'):
        return pos
    a = angle(pos)
    if method == 'logreg':
        return np.column_stack([np.sin(a), np.cos(a), a, np.hypot(*pos.T) / POS_SCALE])
    return np.column_stack([pos, np.hypot(*pos.T), a])


def make_classic(method, kw, seed, n_beams, n_train, n_jobs):
    from sklearn import ensemble, linear_model, naive_bayes, neighbors, pipeline, preprocessing
    if method == 'knn':
        k = min(kw['k'], n_train)
        return neighbors.KNeighborsClassifier(n_neighbors=k, weights='distance' if k > 1 else 'uniform')
    if method == 'lut':
        return AngleLUT(kw['bins'], n_beams)
    if method == 'rf':
        return ensemble.RandomForestClassifier(n_estimators=200, max_depth=kw.get('depth'), n_jobs=n_jobs,
                                               random_state=seed)
    if method == 'et':
        return ensemble.ExtraTreesClassifier(n_estimators=200, n_jobs=n_jobs, random_state=seed)
    if method == 'hgb':
        return ensemble.HistGradientBoostingClassifier(max_iter=100, learning_rate=0.1, random_state=seed)
    if method == 'logreg':
        return pipeline.make_pipeline(preprocessing.StandardScaler(), linear_model.LogisticRegression(max_iter=2000))
    if method == 'gnb':
        return naive_bayes.GaussianNB()
    raise ValueError(method)


class PerScenario:
    def __init__(self, method, kw, seed, n_beams, n_jobs=8):
        self.method, self.kw, self.seed, self.n_beams, self.n_jobs = method, kw, seed, n_beams, n_jobs

    def fit(self, pos, scen, y):
        y = np.asarray(y, dtype=np.int64)
        self.prior = np.bincount(y, minlength=self.n_beams) / len(y)
        self.models = {}
        for s in np.unique(scen):
            m = scen == s
            if len(np.unique(y[m])) == 1:
                self.models[s] = int(y[m][0])
                continue
            model = make_classic(self.method, self.kw, self.seed, self.n_beams, int(m.sum()), self.n_jobs)
            self.models[s] = model.fit(classic_features(self.method, pos[m]), y[m])
        return self

    def predict_proba(self, pos, scen):
        out = np.tile(self.prior, (len(scen), 1))
        for s, model in self.models.items():
            m = scen == s
            if not m.any():
                continue
            if isinstance(model, int):
                out[m] = np.eye(self.n_beams)[model]
            else:
                out[m] = 0
                out[np.ix_(np.flatnonzero(m), model.classes_)] = model.predict_proba(
                    classic_features(self.method, pos[m]))
        return out
