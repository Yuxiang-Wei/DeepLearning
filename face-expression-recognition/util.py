import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

def load_training():
    training = np.load('data/training.npy')
    np.random.shuffle(training) # 随机打乱数据
    training_data = [x[0] for x in training]
    training_label = [x[1] for x in training]
    return training_data, training_label

# TODO 考虑要不要复用

def load_test():
    test = np.load('data/test.npy')
    np.random.shuffle(test) # 随机打乱数据
    test_data = [x[0] for x in test]
    test_label = [x[1] for x in test]
    return test_data, test_label

def evaluate(model, X, Y):
    predicted_Y = model.predict(X)
    accuracy = accuracy_score(Y, predicted_Y)
    return accuracy

def standard(X):
    scaler = StandardScaler().fit(X)
    return scaler.transform(X)

def get_landmarks(image, predictor, rects):
    # this function have been copied from http://bit.ly/2cj7Fpq
    if len(rects) > 1:
        raise BaseException("TooManyFaces")
    if len(rects) == 0:
        raise BaseException("NoFaces")
    return np.matrix([[p.x, p.y] for p in predictor(image, rects[0]).parts()])