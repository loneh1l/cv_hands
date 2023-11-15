from utils.dataset_build import *
import torch
import torchvision.models as models
from torch import nn
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

root_dir = 'result//origin'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

'''
data=Pic_Data(root_dir,transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5), std=(0.5))
    ]
))
'''
data = Onehot_Pic_Data(root_dir, transforms.ToTensor())
train_dataset, test_dataset = train_test_split(data, test_size=0.5, shuffle=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=600, shuffle=True)


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


model = generate_resnet()
model = model.to(device)

model = model.train()
criteria = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
batch_size = 64
# 如果500次迭代loss都没有下降，那么就停止训练
early_stop = 1500
# 记录最小的一次loss值
min_loss = 100.
# 记录下上一次最小的loss是哪一次
last_episode = 0
# 无线更新参数，直到loss不再下降为止
#
for episode in range(101):
    # 使用sample_batch函数sample出一组数据
    for data in test_loader:
        sample, targets = data
        # 将数据送入模型，判断是否为同一类别
        outputs = model(sample.to(device))
        # 将模型的结果丢给CELoss计算损失
        loss = criteria(outputs, targets.to(device).float())
        loss.backward()
        # 更新模型参数
        optimizer.step()
        optimizer.zero_grad()

    # 如果本次损失比之前的小，那么记录一下
    if loss < min_loss:
        min_loss = loss.item()
        last_episode = episode
        torch.save(model, 'model/best_model.pt')
    # 如果连续{early_stop}次loss都没有减小，那么就停止训练
    if episode - last_episode > early_stop:
        break

    # 每50个episode打印一下日志
    if episode % 50 ==  0:
        print(f"episode {episode}, loss {loss}")

print("Finish Training.")

# model = torch.load("model/best_model.pt")
# model = model.to(device)

for data in test_loader:
    sample, targets = data
    y_pre = torch.argmax(model(sample.to(device)), 1).cpu()
    y_true = torch.argmax(targets, 1).cpu()
    print(accuracy_score(y_true, y_pre, normalize=True, sample_weight=None))
