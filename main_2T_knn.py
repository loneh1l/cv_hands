from utils.dataset_build import *
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import cv2
from sklearn import svm


def load_data(directory):
    images = []
    labels = []

    # 遍历每个人的文件夹
    for file_path in os.listdir(directory):
        file_dir=directory+'//'+file_path
        image = cv2.imread(file_dir, cv2.IMREAD_GRAYSCALE)
        # 将图像调整为相同的大小，例如 (32, 32)
        image = cv2.resize(image, (128, 128))
        images.append(image.flatten())  # 将图像转换为一维数组
        labels.append(int(file_path[0:3]))

    return np.array(images), np.array(labels)

dataset_directory = 'result//origin'

X, y = load_data(dataset_directory)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

MODE = 2 # 1 for knn, 2 for svm
if MODE==1:
    clf = KNeighborsClassifier(n_neighbors=3)  # 可根据需要调整邻居数量
elif MODE ==2:
    clf = svm.SVC(kernel='rbf')

clf.fit(X_train, y_train)

# 预测并评估模型
y_pred = clf.predict(X_test)

accuracy = metrics.accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# 打印性能指标
print("Classification Report:")
print(metrics.classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(metrics.confusion_matrix(y_test, y_pred))