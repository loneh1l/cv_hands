import cv2,os
import numpy as np
import matplotlib.pyplot as plt
#from canny import texture_enhancement
def edge_detection(image, threshold1=30, threshold2=100):
    # 边缘检测
    edges = cv2.Canny(image, threshold1, threshold2)

    return edges
def get_img(input_Path):
    img_paths = []
    for (path, dirs, files) in os.walk(input_Path):
        for filename in files:
            if filename.endswith(('.jpg','.png')):
                img_paths.append(path+'/'+filename)
    return img_paths


#构建Gabor滤波器
def build_filters():
     filters = []
     ksize = [7,9,11,13,15,17] # gabor尺度，6个
     lamda = np.pi/2.0         # 波长
     for theta in np.arange(0, np.pi, np.pi / 4): #gabor方向，0°，45°，90°，135°，共四个
         for K in range(6):
             kern = cv2.getGaborKernel((ksize[K], ksize[K]), 1.0, theta, lamda, 0.5, 0, ktype=cv2.CV_32F)
             kern /= 1.5*kern.sum()
             filters.append(kern)
     plt.figure(1)

     #用于绘制滤波器
     '''
     for temp in range(len(filters)):
         plt.subplot(4, 6, temp + 1)
         plt.imshow(filters[temp])
     plt.show()
'''
     return filters

#Gabor特征提取
def getGabor(img,filters):
    res = [] #滤波结果
    fig_0=None
    for i in range(len(filters)):
        # res1 = process(img, filters[i])
        accum = np.zeros_like(img)
        j=0
        for kern in filters[i]:
            j+=1
            fimg = cv2.filter2D(img, cv2.CV_8UC1, kern)
            scal = 2.0
            o = fimg * float(scal)

            o[o > 255] = 255
            o = np.round(o)
            o = o.astype(np.uint8)
            o=texture_enhancement(o)
            fimg = cv2.Laplacian(o, cv2.CV_8U, o, ksize=5)
            #_, fimg = cv2.threshold(fimg, 127, 255, cv2.THRESH_BINARY)
            #fimg = cv2.Canny(fimg, 30, 100)

            kernel = np.ones((2, 2), np.uint8)

            #fimg = cv2.bilateralFilter(fimg, 150, 80, 80)
            #fimg = edge_detection(fimg)
            #fimg = cv2.dilate(fimg, kernel, iterations=1)
            #fimg = cv2.erode(fimg, kernel, iterations=1)
            '''
            fimg = cv2.medianBlur(fimg, 3)
            fimg = cv2.dilate(fimg, kernel, iterations=1)
            fimg = cv2.erode(fimg, kernel, iterations=1)
            _, fimg = cv2.threshold(fimg, 150, 255, cv2.THRESH_BINARY)
            '''
            accum = np.maximum(accum, fimg, accum)
            if i==0 and j==4:
                fig_0=fimg
        res.append(np.asarray(accum))
        print(np.asarray(accum))
    #用于绘制滤波效果

    plt.figure(2)
    for temp in range(len(res)):
        plt.subplot(4,6,temp+1)
        plt.imshow(res[temp], cmap='gray' )
    plt.show()

    return fig_0  #返回滤波结果,结果为24幅图，按照gabor角度排列
def getGabor_pic(img,filters):
    kern=filters[4]
    fimg = cv2.filter2D(img, cv2.CV_8UC1, kern)
    '''
    scal = 2.0
    o = fimg * float(scal)

    o[o > 255] = 255
    o = np.round(o)
    o = o.astype(np.uint8)
    fimg = cv2.Laplacian(o, cv2.CV_8U, o, ksize=5)

    kernel = np.ones((2, 2), np.uint8)
    
    fimg = cv2.medianBlur(fimg, 3)
    fimg = cv2.dilate(fimg, kernel, iterations=1)
    fimg = cv2.erode(fimg, kernel, iterations=1)
    _, fimg = cv2.threshold(fimg, 127, 255, cv2.THRESH_BINARY)
    '''
    return fimg





if __name__ == '__main__':
    input_Path = "dataset/100/1_06_s.bmp"
    filters = build_filters()
    #img_paths = get_img(input_Path)

    img = cv2.imread(input_Path,0)
    cv2.imshow("res", getGabor(img, filters))
    cv2.waitKey(0)
    cv2.destroyAllWindows()



