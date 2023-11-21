import random
from utils.dataset_build import *
import torch
from utils.model import SimilarityModel
from torch import nn
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm

root_dir = "result/gabor_bin"
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
train_dataset, validation_dataset = train_test_split(data,test_size=0.5, shuffle=True)

all_targets = np.array([train_dataset.__getitem__(i)[1] for i in range(len(train_dataset))])
all_labels = np.array(list(set(all_targets)))


def sample_batch(batch_size):
    """
    从train_dataset中sample一些数据对。一半正样本，一半负样本
    """

    # 选取二分之一个batch的labels作为正样本，这样就完成了正样本的构造。
    positive_labels = np.random.choice(all_labels, batch_size // 2)
    # 针对这些labels，每个选取两张相同类别的图片
    batch = []
    for label in positive_labels:
        labels_indexes = np.argwhere(all_targets == label)
        pair = np.random.choice(labels_indexes.flatten(), 2)
        batch.append((pair[0], pair[1], 1)) # 图片类别相同，所以target为1

    # 选取负样本，这次选取一个batch的labels，然后每个labels个选取一张图片。这样就完成了负样本的构造。
    negative_labels = np.random.choice(all_labels, batch_size)
    for sample1, sample2 in negative_labels.reshape(-1, 2):
        sample1 = np.random.choice(np.argwhere(all_targets == sample1).flatten(), 1)
        sample2 = np.random.choice(np.argwhere(all_targets == sample2).flatten(), 1)
        batch.append((sample1.item(), sample2.item(), 0)) # 图片类别不相同，所以target为0

    """
    完成上面的动作后，最终得到的batch如下：
        (734, 736, 1),
        (127, 132, 1),
        ...
        (859, 173, 0),
        ...
    其中前两个表示样本对对应在dataset中的index，1表示前两个样本是相同类别。0表示这两个样本为不同类别。
    接下来需要对其进行shuffle处理，然后从dataset中获取到对应数据，最终组成batch.
    """
    random.shuffle(batch)

    sample1_list = []
    sample2_list = []
    target_list = []
    for sample1, sample2, target in batch:
        sample1_list.append(train_dataset.__getitem__(sample1)[0])
        sample2_list.append(train_dataset.__getitem__(sample2)[0])
        target_list.append(target)
    sample1 = torch.stack(sample1_list)
    sample2 = torch.stack(sample2_list)
    targets = torch.LongTensor(target_list)
    return sample1, sample2, targets


def train():
    model = SimilarityModel()
    model = model.to(device)

    model = model.train()
    criteria = nn.BCELoss()
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
        sample1, sample2, targets = sample_batch(batch_size)
        # 将数据送入模型，判断是否为同一类别
        outputs = model(sample1.to(device), sample2.to(device))
        # 将模型的结果丢给BCELoss计算损失
        loss = criteria(outputs.flatten(), targets.to(device).float())
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
        if episode % 50 == 0:
            print(f"episode {episode}, loss {loss}")

    print("Finish Training.")
    return model


test_train_as_one = 1  # 调试测试：让测试集变得和训练集一样 1是一样 0是不一样
if test_train_as_one:
    validation_dataset = train_dataset

load_model = 1 #  直接从现有加载还是训练一个新的
if load_model:
    model = torch.load("model/best_model.pt")
    model = model.to(device)
else:
    model = train()

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