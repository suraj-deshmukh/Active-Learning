import cv2
import numpy as np
import h5py
import scipy.io as sio

def sliding_window(image):
    tmp = []
    for depth in image:
        for y in range(0, 343, 49):
            for x in range(0, 343, 49):
                window = depth[y:y + 49,x:x + 49]
                tmp.extend([np.mean(window),np.var(window)])
            # yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])
    return(tmp)

#image dataset link:- http://lamda.nju.edu.cn/files/miml-image-data.rar

target_path = "miml-image-data/miml data.mat"
image_path = "miml-image-data/original"

Y = sio.loadmat(target_path)
Y = Y['targets']
Y = Y.transpose()
Y = np.array([[elem if elem == 1 else 0 for elem in row]for row in Y])
X = []

for i in range(1,2001):
    print "reading image:"+ str(i) + ".jpg"
    img = image_path + "/" + str(i) + ".jpg"
    img = cv2.imread(img)
    img = cv2.cvtColor(img,cv2.COLOR_RGB2LUV)
    img = cv2.resize(img,(343,343))
    img = img.transpose((2,0,1))
    X.append(sliding_window(img))
    
X = np.array(X)
f = h5py.File("dataset_294.h5")
f['x'] = X
f['y'] = Y
f.close()
