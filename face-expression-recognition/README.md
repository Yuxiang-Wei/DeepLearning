# 人脸表情识别任务

### 数据集

FER2013，[点击下载](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data)

下载后放于根目录下。

### 依赖库

- skimage
- sklearn
- dlib

### 文件说明

#### preprocess.py

对下载到的csv文件进行预处理，需要最先运行

#### pca.py

利用PCA提取特征进行人脸表情识别

#### hog.py

只利用HOG特征进行人脸表情识别

#### landmarks.py

只利用landmarks特征进行人脸表情识别， 需先下载[shape_predictor_68_face_landmarks.dat](https://github.com/AKSHAYUBHAT/TensorFace/blob/master/openface/models/dlib/shape_predictor_68_face_landmarks.dat)，
放于根目录下。

#### hog_landmarks.py

利用HOG + landmarks特征共同进行人脸表情识别。

#### util.py

辅助函数文件


其余结果见报告。
