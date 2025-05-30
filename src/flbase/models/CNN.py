from ..model import Model
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torchvision.models as models
# from .ResNet import ResNet18, ResNet18NoNorm
from torchvision.models import vgg16, VGG16_Weights, resnet18, ResNet18_Weights


class Conv2Cifar(Model):
    def __init__(self, config):
        super().__init__(config)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.linear1 = nn.Linear(64 * 5 * 5, 384)
        self.linear2 = nn.Linear(384, 192)
        # intentionally remove the bias term for the last linear layer for fair comparison
        self.prototype = nn.Linear(192, config['num_classes'], bias=False)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 5 * 5)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        logits = self.prototype(x)
        return logits

    def get_embedding(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 5 * 5)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        logits = self.prototype(x)
        return x, logits


class Conv2CifarNH(Model):
    def __init__(self, config):
        super().__init__(config)
        self.return_embedding = config['FedNH_return_embedding']
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.linear1 = nn.Linear(64 * 5 * 5, 384)
        self.linear2 = nn.Linear(384, 192)
        temp = nn.Linear(192, config['num_classes'], bias=False).state_dict()['weight']
        self.prototype = nn.Parameter(temp)
        self.scaling = torch.nn.Parameter(torch.tensor([1.0]))

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 5 * 5)
        x = F.relu(self.linear1(x))
        feature_embedding = F.relu(self.linear2(x))
        feature_embedding_norm = torch.norm(feature_embedding, p=2, dim=1, keepdim=True).clamp(min=1e-12)
        feature_embedding = torch.div(feature_embedding, feature_embedding_norm)
        if self.prototype.requires_grad == False:
            normalized_prototype = self.prototype
        else:
            prototype_norm = torch.norm(self.prototype, p=2, dim=1, keepdim=True).clamp(min=1e-12)
            normalized_prototype = torch.div(self.prototype, prototype_norm)
        logits = torch.matmul(feature_embedding, normalized_prototype.T)
        logits = self.scaling * logits

        if self.return_embedding:
            return feature_embedding, logits
        else:
            return logits


class VGG16(Model):
    def __init__(self, config):
        super().__init__(config)
        backbone = vgg16(pretrained=True)
        modules = list(backbone.children())[:-2]
        self.backbone = torch.nn.Sequential(*modules)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(2 * 2 * 512, 128),
            nn.ReLU())
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128, 128),
            nn.ReLU())
        self.prototype = nn.Linear(128, config['num_classes'], bias=False)

    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        x = self.fc1(x)
        logits = self.prototype(x)
        return logits

    def get_embedding(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        x = self.fc1(x)
        logits = self.prototype(x)
        return x, logits


class VGG16NH(Model):
    def __init__(self, config):
        super().__init__(config)
        self.return_embedding = config['FedNH_return_embedding']
        backbone = vgg16(pretrained=True)
        modules = list(backbone.children())[:-2]
        self.backbone = torch.nn.Sequential(*modules)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(2 * 2 * 512, 128),
            nn.ReLU())
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128, 128),
            nn.ReLU())
        temp = nn.Linear(128, config['num_classes'], bias=False).state_dict()['weight']
        self.prototype = nn.Parameter(temp)
        self.scaling = torch.nn.Parameter(torch.tensor([1.0]))

    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        feature_embedding = self.fc1(x)
        feature_embedding_norm = torch.norm(feature_embedding, p=2, dim=1, keepdim=True).clamp(min=1e-12)
        feature_embedding = torch.div(feature_embedding, feature_embedding_norm)
        if self.prototype.requires_grad == False:
            normalized_prototype = self.prototype
        else:
            prototype_norm = torch.norm(self.prototype, p=2, dim=1, keepdim=True).clamp(min=1e-12)
            normalized_prototype = torch.div(self.prototype, prototype_norm)
        logits = torch.matmul(feature_embedding, normalized_prototype.T)
        logits = self.scaling * logits

        if self.return_embedding:
            return feature_embedding, logits
        else:
            return logits


class ResNet18(Model):
    def __init__(self, config):
        super().__init__(config)
        backbone = resnet18(pretrained=True)
        modules = list(backbone.children())[:-2]
        self.backbone = torch.nn.Sequential(*modules)
        self.fc = nn.Linear(2 * 2 * 512, 128)
        self.prototype = nn.Linear(128, config['num_classes'], bias=False)

    def forward(self, x):
        x = self.backbone(x)
        x = nn.Flatten()(x)
        x = self.fc(x)
        logits = self.prototype(x)
        return logits

    def get_embedding(self, x):
        x = self.backbone(x)
        x = nn.Flatten()(x)
        x = self.fc(x)
        logits = self.prototype(x)
        return x, logits


class ResNet18NH(Model):
    def __init__(self, config):
        super().__init__(config)
        self.return_embedding = config['FedNH_return_embedding']
        backbone = resnet18(pretrained=True)
        modules = list(backbone.children())[:-2]
        self.backbone = torch.nn.Sequential(*modules)
        self.fc = nn.Linear(2 * 2 * 512, 128)
        temp = nn.Linear(128, config['num_classes'], bias=False).state_dict()['weight']
        self.prototype = nn.Parameter(temp)
        self.scaling = torch.nn.Parameter(torch.tensor([1.0]))

    def forward(self, x):
        x = self.backbone(x)
        x = nn.Flatten()(x)
        feature_embedding = self.fc(x)
        feature_embedding_norm = torch.norm(feature_embedding, p=2, dim=1, keepdim=True).clamp(min=1e-12)
        feature_embedding = torch.div(feature_embedding, feature_embedding_norm)
        if self.prototype.requires_grad == False:
            normalized_prototype = self.prototype
        else:
            prototype_norm = torch.norm(self.prototype, p=2, dim=1, keepdim=True).clamp(min=1e-12)
            normalized_prototype = torch.div(self.prototype, prototype_norm)
        logits = torch.matmul(feature_embedding, normalized_prototype.T)
        logits = self.scaling * logits

        if self.return_embedding:
            return feature_embedding, logits
        else:
            return logits


class ResNet18Cifar(Model):
    def __init__(self, config):
        super().__init__(config)
        backbone = resnet18(pretrained=True)
        modules = list(backbone.children())[:-2]
        self.backbone = torch.nn.Sequential(*modules)
        self.fc = nn.Linear(1 * 1 * 512, 128)
        self.prototype = nn.Linear(128, config['num_classes'], bias=False)

    def forward(self, x):
        x = self.backbone(x)
        x = nn.Flatten()(x)
        x = self.fc(x)
        logits = self.prototype(x)
        return logits

    def get_embedding(self, x):
        x = self.backbone(x)
        x = nn.Flatten()(x)
        x = self.fc(x)
        logits = self.prototype(x)
        return x, logits


class ResNet18CifarNH(Model):
    def __init__(self, config):
        super().__init__(config)
        self.return_embedding = config['FedNH_return_embedding']
        backbone = resnet18(pretrained=True)
        modules = list(backbone.children())[:-2]
        self.backbone = torch.nn.Sequential(*modules)
        self.fc = nn.Linear(1 * 1 * 512, 128)
        temp = nn.Linear(128, config['num_classes'], bias=False).state_dict()['weight']
        self.prototype = nn.Parameter(temp)
        self.scaling = torch.nn.Parameter(torch.tensor([1.0]))

    def forward(self, x):
        x = self.backbone(x)
        x = nn.Flatten()(x)
        feature_embedding = self.fc(x)
        feature_embedding_norm = torch.norm(feature_embedding, p=2, dim=1, keepdim=True).clamp(min=1e-12)
        feature_embedding = torch.div(feature_embedding, feature_embedding_norm)
        if self.prototype.requires_grad == False:
            normalized_prototype = self.prototype
        else:
            prototype_norm = torch.norm(self.prototype, p=2, dim=1, keepdim=True).clamp(min=1e-12)
            normalized_prototype = torch.div(self.prototype, prototype_norm)
        logits = torch.matmul(feature_embedding, normalized_prototype.T)
        logits = self.scaling * logits

        if self.return_embedding:
            return feature_embedding, logits
        else:
            return logits


class MLP2(Model):
    def __init__(self, config):
        dim = 128
        super().__init__(config)
        self.fc1 = nn.Linear(784, dim * 8)
        self.fc2 = nn.Linear(dim * 8, dim * 4)
        self.fc3 = nn.Linear(dim * 4, dim * 2)
        self.fc4 = nn.Linear(dim * 2, dim)
        self.prototype = nn.Linear(dim, config['num_classes'], bias=False)

    def forward(self, x):
        x = nn.Flatten()(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        logits = self.prototype(x)
        return logits

    def get_embedding(self, x):
        x = nn.Flatten()(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        logits = self.prototype(x)
        return x, logits


class MLP2NH(Model):
    def __init__(self, config):
        dim = 128
        super().__init__(config)
        self.return_embedding = config['FedNH_return_embedding']
        self.fc1 = nn.Linear(784, dim * 8)
        self.fc2 = nn.Linear(dim * 8, dim * 4)
        self.fc3 = nn.Linear(dim * 4, dim * 2)
        self.fc4 = nn.Linear(dim * 2, dim)
        temp = nn.Linear(dim, config['num_classes'], bias=False).state_dict()['weight']
        self.prototype = nn.Parameter(temp)
        self.scaling = torch.nn.Parameter(torch.tensor([1.0]))

    def forward(self, x):
        x = nn.Flatten()(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        feature_embedding = F.relu(self.fc4(x))
        feature_embedding_norm = torch.norm(feature_embedding, p=2, dim=1, keepdim=True).clamp(min=1e-12)
        feature_embedding = torch.div(feature_embedding, feature_embedding_norm)
        if self.prototype.requires_grad == False:
            normalized_prototype = self.prototype
        else:
            prototype_norm = torch.norm(self.prototype, p=2, dim=1, keepdim=True).clamp(min=1e-12)
            normalized_prototype = torch.div(self.prototype, prototype_norm)
        logits = torch.matmul(feature_embedding, normalized_prototype.T)
        logits = self.scaling * logits

        if self.return_embedding:
            return feature_embedding, logits
        else:
            return logits