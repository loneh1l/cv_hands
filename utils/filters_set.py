import cv2
import numpy as np


def edge_detection(image, threshold1=30, threshold2=100):  # 边缘检测
    edges = cv2.Canny(image, threshold1, threshold2)
    return edges


def texture_enhancement(image, sigma=30.0):  # 增强纹理
    # 高斯滤波
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)
    # 纹理增强
    enhanced = cv2.addWeighted(image, 2, blurred, -1, 0)
    return enhanced


def contrast_linear(image, scal=1.5):  # 增强对比度（线性）
    o = image * float(scal)
    o[o > 255] = 255
    o = np.round(o)
    o = o.astype(np.uint8)
    return o


def contrast_exp(gabor_img, p=1.25):  # 增强对比度（指数）
    gabor_img = (gabor_img - np.min(gabor_img, axis=None)) ** p
    _max = np.max(gabor_img, axis=None)
    gabor_img = gabor_img / _max
    gabor_img = gabor_img * 255
    return gabor_img.astype(dtype=np.uint8)


def open_oper(image, size=2):  # 开运算
    kernel = np.ones((size, size), np.uint8)
    erosio1 = cv2.erode(image, kernel, iterations=1)
    dilate1 = cv2.dilate(erosio1, kernel, iterations=1)
    return dilate1


def close_oper(o, size=2):  # 闭运算
    kernel = np.ones((size, size), np.uint8)
    o = cv2.dilate(o, kernel, iterations=1)
    o = cv2.erode(o, kernel, iterations=1)
    return o


def medium(image, size=3):  # 中值滤波
    image = cv2.medianBlur(image, size)
    return image


def gabor_kernel(ksize, sigma, gamma, lamda, alpha, psi):
    sigma_x = sigma
    sigma_y = sigma / gamma

    ymax = xmax = ksize // 2  # 9//2
    xmin, ymin = -xmax, -ymax
    # print("xmin, ymin,xmin, ymin",xmin, ymin,ymax ,xmax)
    # X(第一个参数，横轴)的每一列一样，  Y（第二个参数，纵轴）的每一行都一样
    (y, x) = np.meshgrid(np.arange(ymin, ymax + 1), np.arange(xmin, xmax + 1))  # 生成网格点坐标矩阵
    x_alpha = x * np.cos(alpha) + y * np.sin(alpha)
    y_alpha = -x * np.sin(alpha) + y * np.cos(alpha)
    exponent = np.exp(-.5 * (x_alpha ** 2 / sigma_x ** 2 + y_alpha ** 2 / sigma_y ** 2))
    kernel = exponent * np.cos(2 * np.pi / lamda * x_alpha + psi)
    return kernel


def gabor(gray_img, ksize=9, sigma=1.0, gamma=0.5, lamda=5, psi=-np.pi/2):#gabor滤波
    filters = []
    for alpha in np.arange(0, np.pi, np.pi / 4):
        kern = gabor_kernel(ksize=ksize, sigma=sigma, gamma=gamma,lamda=lamda, alpha=alpha, psi=psi)
        filters.append(kern)

    gabor_img = np.zeros(gray_img.shape, dtype=np.uint8)

    i = 0
    for kern in filters:
        fimg = cv2.filter2D(gray_img, ddepth=cv2.CV_8U, kernel=kern)
        gabor_img = cv2.max(gabor_img, fimg)
        #cv2.imwrite("2." + str(i) + "gabor.jpg", gabor_img)
        i += 1

    return gabor_img


def deli(image):  # 删除连通分量低于平均值的部分
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image)

    # 计算平均面积
    areas = list()
    for i in range(num_labels):
        areas.append(stats[i][-1])

    area_avg = np.average(areas[1:-1])
    # 筛选超过平均面积的连通域
    image_filtered = np.zeros_like(image)
    for (i, label) in enumerate(np.unique(labels)):
        # 如果是背景，忽略
        if label == 0:
            continue
        if stats[i][-1] > area_avg:
            image_filtered[labels == i] = 255
    return image_filtered


def show(image, name='Fuck'):  # 图片展示
    cv2.imshow(name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()