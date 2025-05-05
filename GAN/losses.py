import torch
import torch.nn as nn

class ContentLoss(nn.Module):
    def __init__(self, target, weight):
        super().__init__()
        self.weight = weight
        self.target = target.detach() * weight
        self.loss = 0.0

    def forward(self, input):
        self.loss = nn.functional.mse_loss(input * self.weight, self.target)
        return input

class StyleLoss(nn.Module):
    def __init__(self, target_feature, weight):
        super().__init__()
        self.weight = weight
        self.target = self.gram_matrix(target_feature).detach() * weight
        self.loss = 0.0

    def gram_matrix(self, input):
        b, c, h, w = input.size()
        features = input.view(b, c, h * w)
        G = torch.bmm(features, features.transpose(1, 2))
        return G.div(c * h * w)

    def forward(self, input):
        G = self.gram_matrix(input) * self.weight
        target = self.target.expand_as(G)
        self.loss = nn.functional.mse_loss(G, target)
        return input

class Normalization(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer('mean', torch.tensor(mean).view(-1,1,1))
        self.register_buffer('std',  torch.tensor(std).view(-1,1,1))

    def forward(self, img):
        return (img - self.mean) / self.std


def build_style_losses(cnn, norm_mean, norm_std, style_img, style_layers, device):
    normalization = Normalization(norm_mean, norm_std).to(device)
    model = nn.Sequential(normalization)
    style_losses = []
    i = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1; name = f'conv_{i}'
        elif isinstance(layer, nn.ReLU):
            name = f'relu_{i}'; layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = f'pool_{i}'
        elif isinstance(layer, nn.BatchNorm2d):
            name = f'bn_{i}'
        else:
            continue
        model.add_module(name, layer)
        if name in style_layers:
            target_feature = model(style_img).detach()
            sl = StyleLoss(target_feature, weight=1e3)
            style_losses.append((name, sl))
    return style_losses


def build_content_losses(cnn, norm_mean, norm_std, content_img, content_layers, device):
    normalization = Normalization(norm_mean, norm_std).to(device)
    model = nn.Sequential(normalization)
    content_losses = []
    i = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1; name = f'conv_{i}'
        elif isinstance(layer, nn.ReLU):
            name = f'relu_{i}'; layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = f'pool_{i}'
        elif isinstance(layer, nn.BatchNorm2d):
            name = f'bn_{i}'
        else:
            continue
        model.add_module(name, layer)
        if name in content_layers:
            target_feature = model(content_img).detach()
            cl = ContentLoss(target_feature, weight=1)
            content_losses.append((name, cl))
    return content_losses


def build_loss_model(cnn, norm_mean, norm_std,
                     style_losses, content_losses,
                     style_layers, content_layers, device):
    normalization = Normalization(norm_mean, norm_std).to(device)
    model = nn.Sequential(normalization)
    i = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1; name = f'conv_{i}'
        elif isinstance(layer, nn.ReLU):
            name = f'relu_{i}'; layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = f'pool_{i}'
        elif isinstance(layer, nn.BatchNorm2d):
            name = f'bn_{i}'
        else:
            continue
        model.add_module(name, layer)
        for sl_name, sl in style_losses:
            if sl_name == name:
                model.add_module(f"style_loss_{name}", sl)
        for cl_name, cl in content_losses:
            if cl_name == name:
                model.add_module(f"content_loss_{name}", cl)
    # trim after last loss module
    for j in range(len(model) - 1, -1, -1):
        if isinstance(model[j], ContentLoss) or isinstance(model[j], StyleLoss):
            break
    model = model[:j+1]
    return model