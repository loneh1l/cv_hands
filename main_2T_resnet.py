import random
from utils.dataset_build import *
import torch
import torchvision
import torchvision.models as models
from torch import nn
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm

root_dir = 'result//gabor_bin'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

'''
data=Pic_Data(root_dir,transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5), std=(0.5))
    ]
))
'''
data = Pic_Data(root_dir, transforms.ToTensor())
train_dataset, validation_dataset = train_test_split(data, test_size=0.5, shuffle=True)

all_targets = np.array([train_dataset.__getitem__(i)[1] for i in range(len(train_dataset))])
all_labels = np.array(list(set(all_targets)))



def sample_batch(batch_size):
    """
    从train_dataset中sample一些数据
    """
    batch = []

    labels = np.random.choice(all_labels, batch_size)
    for target in labels.reshape(-1, 1):
        sample = np.random.choice(np.argwhere(all_targets == target).flatten(), 1)
        label = np.eye(100)[target[0]-1]
        batch.append((sample.item(), label))

    random.shuffle(batch)

    sample_list = []
    target_list = []
    for sample, target in batch:
        sample_list.append(train_dataset.__getitem__(sample)[0])
        target_list.append(target)

    sample = torch.stack(sample_list)
    targets = torch.LongTensor(target_list)
    return sample, targets


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

for episode in range(1000):
    # 使用sample_batch函数sample出一组数据，包含一半正样本，一半负样本
    sample, targets = sample_batch(batch_size)
    # 将数据送入模型，判断是否为同一类别
    outputs = model(sample.to(device))
    # 将模型的结果丢给BCELoss计算损失
    # print(outputs)
    # print(targets)
    loss = criteria(outputs, targets.to(device).float())
    # print(loss)
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

#validation_dataset=train_dataset
model=torch.load("model/best_model.pt")
model=model.to(device)

support_set = []
validation_set = []
# 遍历所有的标签，每个标签选取前5个作为support set，后面的作为验证数据
for label in all_labels:
    label_indexes = np.argwhere(all_targets == label)
    support_set.append((label_indexes[:2].flatten().tolist()))
    validation_set += label_indexes[2:].flatten().tolist()


def predict(image):
    sim_list = [] # 存储image与每个类别的相似度
    # 一个类别一个类别的遍历，indexes存储的就是当前类别的5张图片的index
    for indexes in support_set:
        # 去validation_dataset中找出index对应的图片tensor
        tensor_list = []
        for i in indexes:
            tensor_list.append(validation_dataset[i][0])
        support_tensor = torch.stack(tensor_list)
        # 拿到该类别的5个图片后，就可以送给模型求image与它们的相似程度了，最后求个平均
        sim = model(image.repeat(2, 1,1,1).to(device), support_tensor.to(device)).mean()
        sim_list.append(sim)

    # 找出其中相似程度最高的那个，它就是预测结果
    #print(sim_list)
    result_index = torch.stack(sim_list).argmax().item()
    return all_labels[result_index]


total = 0
total_correct = 0

# 由于验证集太大，为了更快的看到效果，我们验证前，将validation_set打乱一下再验证
random.shuffle(validation_set)
progress = tqdm(validation_set)

for i in progress:
    image, label = validation_dataset.__getitem__(i)
    predict_label = predict(image)

    total += 1
    if predict_label == label:
        total_correct += 1

    progress.set_postfix({
            "accuracy": str("%.3f" % (total_correct / total))
        })