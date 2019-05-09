import numpy as np
import time
from skimage.feature import hog
from sklearn.svm import SVC
from util import load_training, load_test, evaluate, standard, get_landmarks
import warnings
import dlib

warnings.filterwarnings('ignore')

### 此处定义参数
pixels_per_cell = (16, 16)  # 直方图窗口大小
C = 1  # 软间隔系数
decision_function = 'ovr'  # 'ovo' for OneVsOne and 'ovr' for OneVsRest'
kernel = 'rbf'  # 核函数类型 'rbf', 'linear', 'poly' or 'sigmoid'
gamma = 1e-3  # 针对rbf, gamma越大，支持向量越少
#####

training_data, training_label = load_training()
test_data, test_label = load_test()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

print('training size: {}'.format(len(training_label)))
print('test size: {}'.format(len(test_label)))

print('start extract features ... ')
# 提取landmarks特征
face_rects = [dlib.rectangle(left=1, top=1, right=47, bottom=47)]

training_landmark_ft = [get_landmarks(image.astype(np.uint8), predictor, face_rects) for image in training_data]
test_landmark_ft = [get_landmarks(image.astype(np.uint8), predictor, face_rects) for image in test_data]

# 提取HOG特征
training_hog_ft = [hog(image, orientations=8, pixels_per_cell=pixels_per_cell,
                       cells_per_block=(1, 1), visualise=True)[0] for image in training_data]
test_hog_ft = [hog(image, orientations=8, pixels_per_cell=pixels_per_cell,
                       cells_per_block=(1, 1), visualise=True)[0] for image in test_data]

# 展开landmarks特征
training_landmark_ft = np.array([x.flatten() for x in training_landmark_ft])
training_label = np.array(training_label)
test_landmark_ft = np.array([x.flatten() for x in test_landmark_ft])
test_label = np.array(test_label)
x_dim, _, z_dim = np.shape(training_landmark_ft)
training_landmark_ft = training_landmark_ft.reshape((x_dim, z_dim))
x_dim, _, z_dim = np.shape(test_landmark_ft)
test_landmark_ft = test_landmark_ft.reshape((x_dim, z_dim))
# 展开hog特征
training_hog_ft = np.array([x.flatten() for x in training_hog_ft])
test_hog_ft = np.array([x.flatten() for x in test_hog_ft])

# 合并两个特征
training_ft = np.concatenate((training_landmark_ft, training_hog_ft), axis=1)
test_ft = np.concatenate((test_landmark_ft, test_hog_ft), axis=1)
# standard
training_ft = standard(training_ft)
test_ft = standard(test_ft)

model = SVC(C=C, random_state=0, max_iter=1000, kernel=kernel, decision_function_shape=decision_function,
            gamma=gamma)

print('start training ... ')
start_time = time.time()
model.fit(training_ft, training_label)
training_time = time.time() - start_time
print('training time = {0:.1f} sec. n_support={1}'.format(training_time, model.n_support_))
train_accuracy = evaluate(model, training_ft, training_label)
print('  - train accuracy = {0:.1f}'.format(train_accuracy * 100))

print('evaluating...')
test_accuracy = evaluate(model, test_ft, test_label)
print('  - test accuracy = {0:.1f}'.format(test_accuracy * 100))
