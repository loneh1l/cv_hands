from utils.feature_extraction import *
import os
def extract_feature(dir='dataset/001/1_01_s.bmp',switch=-1, MODE_OPTION = 2):
    image = cv2.imread(dir, 0)
    if MODE_OPTION == 1:
        image = extract_feature_1(image, switch)
    if MODE_OPTION == 2:
        image = extract_feature_2(image, switch)
    return image



if __name__ == '__main__':
    path_in = "dataset"
    MODE_OPTION = 1  # 模式选择，0是不做任何处理，输出原图 1是二值化，2是不二值化
    if MODE_OPTION == 1:
        path_out = 'origin'
    elif MODE_OPTION == 1:
        path_out = 'gabor'
    elif MODE_OPTION == 2:
        path_out = 'gabor_bin'
    else:
        print('草泥马的选你妈呢，把你妈都选了')
        exit(0)
    extract_feature(MODE_OPTION=MODE_OPTION)  # 预览某一张的处理效果
    if not os.path.exists("result\\" + path_out):
        os.makedirs("result\\" + path_out)

    for i in os.listdir(path_in):
        path2 = path_in + "\\" + i
        for j in os.listdir(path2):
            path3 = path2 + "\\" + j
            image = extract_feature(path3, 0, MODE_OPTION)
            file_path = "result\\" + path_out + path3[7:11] + '.' + path3[12:]
            cv2.imwrite(file_path, image)