from utils.dataset_build import *
import torch
from torch import nn
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from utils.model import generate_resnet
import matplotlib.pyplot as plt
import os

root_dir = 'result//gabor_bin'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def test(model, dataset):
    test_loader = DataLoader(dataset=dataset, batch_size=600, shuffle=True)
    for data in test_loader:
        sample, targets = data
        y_pre = torch.argmax(model(sample.to(device)), 1).cpu()
        y_true = torch.argmax(targets, 1).cpu()
        acc = accuracy_score(y_true, y_pre, normalize=True, sample_weight=None)
    return acc


def learn(dataset, testdataset):
    train_loader = DataLoader(dataset=dataset, batch_size=64, shuffle=True)
    model = generate_resnet()
    model = model.to(device)
    model = model.train()
    criteria = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    # 记录最小的一次loss值
    min_loss = 100.
    loss_dict = []
    acc_dict = []
    for episode in range(500):
        mean_loss = 0
        for data in train_loader:
            sample, targets = data
            outputs = model(sample.to(device))  # 将数据送入模型
            loss = criteria(outputs, targets.to(device).float())  # 将模型的结果丢给CELoss计算损失
            mean_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        acc = test(model, testdataset)
        acc_dict.append(acc)
        mean_loss /= 64
        loss_dict.append(mean_loss)
        # 如果本次损失比之前的小，那么记录一下
        if loss < min_loss:
            min_loss = loss.item()
            torch.save(model, 'model/best_model.pt')

        # 每50个episode打印一下日志
        if (episode+1) % 50 == 0:
            print(f"episode {episode+1}, loss {loss}")
    print("Finish Training.")
    return model, acc_dict, loss_dict


data = Pic_Data(root_dir, transforms.ToTensor(), onehot=True)  # 打标签
train_dataset, test_dataset = train_test_split(data, test_size=0.5, shuffle=True)  # 划分训练集和测试集

load_model = False
if load_model:
    model = torch.load("model/best_model.pt")
    model = model.to(device)
else:
    model, acc_dict, loss_dict = learn(train_dataset, test_dataset)

    # 画图
    if not os.path.exists("result/picture"):
        os.makedirs("result/picture")
    plt.plot(range(500), acc_dict)
    plt.savefig('result/picture/acc_resnet_gabor_bin.png')
    plt.show()
    plt.plot(range(500), loss_dict)
    plt.savefig('result/picture/loss_resnet_gabor_bin.png')
    plt.show()

acc = test(model, test_dataset)
print(acc)
