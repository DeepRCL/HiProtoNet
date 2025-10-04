import torch.nn as nn
import torch
from torchsummary import summary
import torch.utils.model_zoo as model_zoo
from torchvision import models

model_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth",
    "resnet152": "https://download.pytorch.org/models/resnet152-b121ed2d.pth",
    "resnet2p1d_18": "https://download.pytorch.org/models/r2plus1d_18-91a641e6.pth",
    "r3d_18": "https://download.pytorch.org/models/r3d_18-b3b3357e.pth",
}

model_dir = "./pretrained_models"


class DenseNet(nn.Module):
    """
    a simple multilabel image classifier using densenet
    """

    def __init__(
        self,
        num_classes: int = 3,
        cnn_dropout_p: float = 0.2,
        classifier_hidden_dim: int = 128,
        classifier_dropout_p: float = 0.5,
        **kwargs,
    ):
        """
        :param num_classes: int, number of classes
        :param cnn_dropout_p: float, dropout ratio of the CNN
        :param classifier_hidden_dim: int, the dimension of the hidden FC layer
        :param classifier_dropout_p: float, dropout ratio of the FC layer
        """
        super().__init__()

        # the backbone CNN(DenseNet), output shape (N, 1024)
        self.backbone = torch.hub.load("pytorch/vision:v0.10.0", "densenet121", pretrained=True)
        self.backbone.classifier = nn.Identity()

        # 2 dense blocks and transition blocks
        # output shape (N, 256, 14, 14)
        # self.backbone = nn.Sequential(*list(densenet.children())[0][:-4])

        # the FC layer applied to the output of convolutional network
        self.classifier: torch.nn.Sequential = nn.Sequential(
            nn.Linear(in_features=1024, out_features=classifier_hidden_dim),
            nn.BatchNorm1d(classifier_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=classifier_dropout_p),
            nn.Linear(in_features=classifier_hidden_dim, out_features=num_classes),
        )

    def forward(self, x):
        """
        :param x: torch.tensor, input torch.tensor of image frames,
                  normalized with imagenet's mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225]
                  shape (N, 3, img_size, img_size)
        :return:  multi-hot Vector of logits with shape of (N, num_classes)
        """
        x = self.backbone(x)  # shape (N, 1024)
        x = self.classifier(x)
        return x


class resnet2p1d_18(nn.Module):
    def __init__(self, pretrained=True, num_classes=15, **kwargs):
        """
        :param pretrained: to load the pretrained weights or not
        """
        super().__init__()
        r2p1d = models.video.r2plus1d_18(pretrained=False)  # output shape = NxD = Nx512
        if pretrained:
            my_dict = model_zoo.load_url(model_urls["resnet2p1d_18"], model_dir=model_dir)
            r2p1d.load_state_dict(my_dict, strict=False)
        self.backbone = nn.Sequential(*(list(r2p1d.children())[:-1]))
        self.classifier = nn.Linear(in_features=512, out_features=num_classes)

    def forward(self, x):
        """
        x(batch of tensors)   shape = (N, C, T, H, W)
        """
        x = self.backbone(x)
        # flatten the last 3 dimensions out of 5
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

if __name__ == "__main__":
    batch_size = 20
    model = DenseNet()
    summary(model, torch.rand((batch_size, 3, 224, 224)))  # (N,C,T,H,W)