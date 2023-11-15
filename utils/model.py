import torchvision.models as models
from torch import nn

def generate_resnet(num_classes=100, in_channels=1, model_name="ResNet18"):
    if model_name == "ResNet18":
        model = models.resnet18(pretrained=True)
    elif model_name == "ResNet34":
        model = models.resnet34(pretrained=True)
    elif model_name == "ResNet50":
        model = models.resnet50(pretrained=True)
    elif model_name == "ResNet101":
        model = models.resnet101(pretrained=True)
    elif model_name == "ResNet152":
        model = models.resnet152(pretrained=True)
    model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    fc_features = model.fc.in_features
    model.fc = nn.Linear(fc_features, num_classes)

    return model


class SimilarityModel(nn.Module):

    def __init__(self):
        super(SimilarityModel, self).__init__()

        # 定义一个卷积层，用于特征提取。
        self.conv = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=5),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(10, 20, kernel_size=5),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(20, 20, kernel_size=5),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
        )

        # 定义一个线性层，用来判断两张图片是否为同一类别
        self.sim = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2880, 100),
            nn.ReLU(),
            nn.Linear(100, 1),
            nn.Sigmoid(),
        )

    def forward(self, sample1, sample2):
        sample1_features = self.conv(sample1)
        sample2_features = self.conv(sample2)
        return self.sim(torch.abs(sample1_features - sample2_features))
