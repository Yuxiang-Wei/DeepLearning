import numpy as np
import time
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from util import load_training, load_test, evaluate, standard
import warnings
warnings.filterwarnings('ignore')

### 此处定义参数
n_components = 100  # PCA降至的维数
C = 1  # 软间隔系数
decision_function = 'ovr'  # 'ovo' for OneVsOne and 'ovr' for OneVsRest'
kernel = 'rbf'  # 核函数类型 'rbf', 'linear', 'poly' or 'sigmoid'
gamma = 1e-5  # 针对rbf, gamma越大，支持向量越少
#####
training_data, training_label = load_training()
test_data, test_label = load_test()

print('training size: {}'.format(len(training_label)))
print('test size: {}'.format(len(test_label)))

# 展成一维
training_data = np.array([x.flatten() for x in training_data])
training_label = np.array(training_label)
test_data = np.array([x.flatten() for x in test_data])
test_label = np.array(test_label)

pca = PCA(n_components=n_components)
model = SVC(C=C, random_state=0, max_iter=1000, kernel=kernel, decision_function_shape=decision_function,
            gamma=gamma)

print('start extract features ... ')

training_pca_ft = pca.fit_transform(training_data)
test_pca_ft = pca.fit_transform(test_data)
# standard
training_pca_ft = standard(training_pca_ft)
test_pca_ft = standard(test_pca_ft)

print('start training ... ')
start_time = time.time()
model.fit(training_pca_ft, training_label)
training_time = time.time() - start_time
print('training time = {0:.1f} sec. n_support={1}'.format(training_time, model.n_support_))
train_accuracy = evaluate(model, training_pca_ft, training_label)
print("  - train accuracy = {0:.1f}".format(train_accuracy * 100))

print('evaluating...')
test_accuracy = evaluate(model, test_pca_ft, test_label)
print('  - test accuracy = {0:.1f}'.format(test_accuracy * 100))
