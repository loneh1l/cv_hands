from utils.filters_set import *
'''
gabor滤波+伽马变换+高斯滤波+纹理增强+二值化+减去较小连通分量
'''
def extract_feature_1(image,swtich=-1):
    if swtich==-1:#展示中间步骤
        show(image)
        image = gabor(image)
        show(image)
        image = contrast_exp(image, p=1.25)
        show(image)
        image = texture_enhancement(image, 1.0)
        show(image)
        _, image = cv2.threshold(image, 50, 255, cv2.THRESH_BINARY)
        show(image)
        image = deli(image)
        show(image)
    else:
        image = gabor(image)
        image = contrast_exp(image, p=1.25)
        image = texture_enhancement(image, 1.0)
        _, image = cv2.threshold(image, 50, 255, cv2.THRESH_BINARY)
        image = deli(image)
    return image

'''
gabor滤波+伽马变换+高斯滤波+纹理增强
'''
def extract_feature_2(image,swtich=-1):
    if swtich==-1:#展示中间步骤
        show(image)
        image = gabor(image)
        show(image)
        image = contrast_exp(image, p=1.25)
        show(image)
        image = texture_enhancement(image, 1.0)
        show(image)
    else:
        image = gabor(image)
        image = contrast_exp(image, p=1.25)
        image = texture_enhancement(image, 1.0)
    return image
'''
canny函数边缘检测
'''
def extract_feature_3(image,swtich=-1):
    if swtich==-1:#展示中间步骤
        show(image)
        image=edge_detection(image)
        show(image)
    else:
        image=edge_detection(image)
    return image