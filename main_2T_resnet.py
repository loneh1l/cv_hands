from utils.dataset_build import *
import torch
from torch import nn
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from utils.model import generate_resnet
import matplotlib.pyplot as plt
import os
import datetime
import numpy as np

current_time = datetime.datetime.now()
current_time_str = current_time.strftime('%Y_%m_%d_%H_%M_%S')
episode_num = 1000

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
    min_loss = 100.  # 记录最小的一次loss值
    max_acc = 0.  # 记录最大acc
    loss_dict = []
    acc_dict = []
    for episode in range(episode_num):
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
        loss_dict.append(np.log10(mean_loss))
        # 如果本次acc比之前的大，那么记录一下
        if acc > max_acc:
            max_acc = acc
            if not os.path.exists("model/" + current_time_str):
                os.makedirs("model/" + current_time_str)
            torch.save(model, 'model/'+current_time_str+'/best_model.pt')

        # 每50个episode打印一下日志
        if (episode+1) % 50 == 0:
            print(f"episode {episode+1}, loss {mean_loss}")
    print("Finish Training.")
    print('max_acc:', max_acc)
    with open("model/"+current_time_str+"/max_acc.txt", 'a') as f:
        f.write(str(max_acc))
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
    if not os.path.exists("result/picture/"+current_time_str):
        os.makedirs("result/picture/"+current_time_str)
    plt.plot(range(episode_num), acc_dict)
    plt.savefig('result/picture/'+current_time_str+'/acc_resnet_gabor_bin.png')
    plt.show()
    plt.plot(range(episode_num), loss_dict)
    plt.savefig('result/picture/'+current_time_str+'/loss_resnet_gabor_bin.png')
    plt.show()

acc = test(model, test_dataset)
print(acc)
