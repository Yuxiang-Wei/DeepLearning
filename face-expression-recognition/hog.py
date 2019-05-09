import numpy as np
import time
from skimage.feature import hog
from sklearn.svm import SVC
from util import load_training, load_test, evaluate, standard
import warnings

warnings.filterwarnings('ignore')

### 此处定义参数
pixels_per_cell = (16, 16)  # 直方图窗口大小
C = 1000  # 软间隔系数
decision_function = 'ovr'  # 'ovo' for OneVsOne and 'ovr' for OneVsRest'
kernel = 'rbf'  # 核函数类型 'rbf', 'linear', 'poly' or 'sigmoid'
gamma = 1e-1  # 针对rbf, gamma越大，支持向量越少
#####

training_data, training_label = load_training()
test_data, test_label = load_test()

print('training size: {}'.format(len(training_label)))
print('test size: {}'.format(len(test_label)))

print('start extract features ... ')
# 提取HOG特征
training_hog_ft = [hog(image, orientations=8, pixels_per_cell=pixels_per_cell,
                       cells_per_block=(1, 1), visualise=True)[0] for image in training_data]
test_hog_ft = [hog(image, orientations=8, pixels_per_cell=pixels_per_cell,
                       cells_per_block=(1, 1), visualise=True)[0] for image in test_data]

# 展开特征
training_hog_ft = np.array([x.flatten() for x in training_hog_ft])
training_label = np.array(training_label)
test_hog_ft = np.array([x.flatten() for x in test_hog_ft])
test_label = np.array(test_label)
# standard
training_hog_ft = standard(training_hog_ft)
test_hog_ft = standard(test_hog_ft)

model = SVC(C=C, random_state=0, max_iter=1000, kernel=kernel, decision_function_shape=decision_function,
            gamma=gamma)

print('start training ... ')
start_time = time.time()
model.fit(training_hog_ft, training_label)
training_time = time.time() - start_time
print('training time = {0:.1f} sec. n_support={1}'.format(training_time, model.n_support_))
train_accuracy = evaluate(model, training_hog_ft, training_label)
print('  - train accuracy = {0:.1f}'.format(train_accuracy * 100))

print('evaluating...')
test_accuracy = evaluate(model, test_hog_ft, test_label)
print('  - test accuracy = {0:.1f}'.format(test_accuracy * 100))
