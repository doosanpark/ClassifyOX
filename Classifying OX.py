import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

class MultiClassNetwork:
    
    def __init__(self, units=10, batch_size=32, learning_rate=0.2, l1=0, l2=0):
        self.units = units         # 은닉층의 뉴런 개수
        self.batch_size = batch_size     # 배치 크기
        self.w1 = None             # 은닉층의 가중치
        self.b1 = None             # 은닉층의 절편
        self.w2 = None             # 출력층의 가중치
        self.b2 = None             # 출력층의 절편
        self.a1 = None             # 은닉층의 활성화 출력
        self.losses = []           # 훈련 손실
        self.val_losses = []       # 검증 손실
        self.lr = learning_rate    # 학습률
        self.l1 = l1               # L1 손실 하이퍼파라미터
        self.l2 = l2               # L2 손실 하이퍼파라미터

    # 은닉증, 출력층을 지난 z2 출력
    def forpass(self, img):
        z1 = np.dot(img, self.w1) + self.b1      # 첫 번째 층의 선형 식을 계산합니다. 행렬간의 곱셈. w1x1 + b1
        self.a1 = self.sigmoid(z1)               # 활성화 함수를 적용합니다.
        z2 = np.dot(self.a1, self.w2) + self.b2  # 두 번째 층의 선형 식을 계산합니다.
        return z2

    def backprop(self, img, err):
        m = len(img)       # 샘플 개수
        # 출력층의 가중치와 절편에 대한 그래디언트를 계산합니다.
        w2_grad = np.dot(self.a1.T, err) / m
        b2_grad = np.sum(err) / m
        # 시그모이드 함수까지 그래디언트를 계산합니다.
        err_to_hidden = np.dot(err, self.w2.T) * self.a1 * (1 - self.a1)
        # 은닉층의 가중치와 절편에 대한 그래디언트를 계산합니다.
        w1_grad = np.dot(img.T, err_to_hidden) / m
        b1_grad = np.sum(err_to_hidden, axis=0) / m
        return w1_grad, b1_grad, w2_grad, b2_grad
    
    def sigmoid(self, z):
        z = np.clip(z, -100, None)            # 안전한 np.exp() 계산을 위해
        a = 1 / (1 + np.exp(-z))              # 시그모이드 계산
        return a
    
    def softmax(self, z):
        # 소프트맥스 함수
        z = np.clip(z, -100, None)            # 안전한 np.exp() 계산을 위해
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z, axis=1).reshape(-1, 1)
 
    def init_weights(self, n_features, n_classes):
        #정규분포 난수를 이용하여 가중치 설정
        self.w1 = np.random.normal(0, 1, (n_features, self.units))  # (특성 개수 784, 은닉층의 크기 100)
        self.b1 = np.zeros(self.units)                              # 은닉층의 크기
        self.w2 = np.random.normal(0, 1, (self.units, n_classes))   # (은닉층의 크기, 클래스 개수)
        self.b2 = np.zeros(n_classes)
        
    def fit(self, img, label, epochs=100, img_val=None, label_val=None):
        np.random.seed(42)
        self.init_weights(img.shape[1], label.shape[1])    # 은닉층과 출력층의 가중치를 초기화합니다.
        # epochs만큼 반복합니다.
        for i in range(epochs):
            loss = 0
            print('.', end='')
            # 제너레이터 함수에서 반환한 미니배치를 순환합니다.
            for img_batch, label_batch in self.gen_batch(img, label):
                a = self.training(img_batch, label_batch)
                # 안전한 로그 계산을 위해 클리핑합니다.
                a = np.clip(a, 1e-10, 1-1e-10)
                # 로그 손실과 규제 손실을 더하여 리스트에 추가합니다.
                loss += np.sum(-label_batch*np.log(a))
            self.losses.append((loss + self.reg_loss()) / len(img))
            # 검증 세트에 대한 손실을 계산합니다.
            self.update_val_loss(img_val, label_val)

    # 미니배치 제너레이터 함수. 
    # img와 label 배열 순서 변경 후 배치 생성
    def gen_batch(self, img, label):
        length = len(img)
        bins = length // self.batch_size # 미니배치 횟수
        if length % self.batch_size:
            bins += 1                    # 나누어 떨어지지 않을 때
        indexes = np.random.permutation(np.arange(len(img))) # 인덱스를 섞습니다.
        img = img[indexes]
        label = label[indexes]
        for i in range(bins):
            start = self.batch_size * i
            end = self.batch_size * (i + 1)
            yield img[start:end], label[start:end]   # batch_size만큼 슬라이싱하여 반환합니다.

    def training(self, img, label):
        m = len(img)                # 샘플 개수를 저장합니다.
        z = self.forpass(img)       # 정방향 계산을 수행합니다.
        a = self.softmax(z)         # 활성화 함수를 적용합니다.
        err = -(label - a)          # 오차를 계산합니다.
        # 오차를 역전파하여 그래디언트를 계산합니다.
        w1_grad, b1_grad, w2_grad, b2_grad = self.backprop(img, err)
        # 그래디언트에서 페널티 항의 미분 값을 뺍니다
        w1_grad += (self.l1 * np.sign(self.w1) + self.l2 * self.w1) / m
        w2_grad += (self.l1 * np.sign(self.w2) + self.l2 * self.w2) / m
        # 은닉층의 가중치와 절편을 업데이트합니다.
        self.w1 -= self.lr * w1_grad
        self.b1 -= self.lr * b1_grad
        # 출력층의 가중치와 절편을 업데이트합니다.
        self.w2 -= self.lr * w2_grad
        self.b2 -= self.lr * b2_grad
        return a
    
    def predict(self, img):
        z = self.forpass(img)          # 정방향 계산을 수행합니다.
        return np.argmax(z, axis=1)  # 가장 큰 값의 인덱스를 반환합니다.
    
    def score(self, img, label):
        # 예측과 타깃 열 벡터를 비교하여 True의 비율을 반환합니다.
        return np.mean(self.predict(img) == np.argmax(label, axis=1))

    def reg_loss(self):
        # 은닉층과 출력층의 가중치에 규제를 적용합니다.
        return self.l1 * (np.sum(np.abs(self.w1)) + np.sum(np.abs(self.w2))) + \
               self.l2 / 2 * (np.sum(self.w1**2) + np.sum(self.w2**2))

    def update_val_loss(self, img_val, label_val):
        z = self.forpass(img_val)            # 정방향 계산을 수행합니다.
        a = self.softmax(z)                # 활성화 함수를 적용합니다.
        a = np.clip(a, 1e-10, 1-1e-10)     # 출력 값을 클리핑합니다.
        # 크로스 엔트로피 손실과 규제 손실을 더하여 리스트에 추가합니다.
        val_loss = np.sum(-label_val*np.log(a))
        self.val_losses.append((val_loss + self.reg_loss()) / len(label_val))


import os
def load_images_from_folder(folder, label, target_size=(28, 28)):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = image.load_img(img_path, color_mode='grayscale', target_size=target_size)
        img_array = image.img_to_array(img)
        img_array = img_array / 255.0  # Normalize to [0, 1]
        images.append(img_array)
        labels.append(label)
    return images, labels

# 경로 설정
train_o_dir = './OX_images/train/O/'
train_x_dir = './OX_images/train/X/'

# O와 X 이미지 로드
o_images, o_labels = load_images_from_folder(train_o_dir, 0)    #O 이미지에는 라벨을 0
x_images, x_labels = load_images_from_folder(train_x_dir, 1)    #X 이미지에는 라벨을 1

# 이미지와 라벨 합치기
all_images = np.array(o_images + x_images)  # O 이미지와 X 이미지 하나로
all_labels = np.array(o_labels + x_labels)  # O 라벨과   X 라벨   하나로

# 데이터셋 섞기
## 데이터를 무작위로 섞은 후 지정된 비율에 따라 나눈다.
### 학습용 : 테스트용 = 8:2 비율로 분할
### img는 이미지, label은 라벨
### label 값은 0은 O, 1은 X 값. 0과 1로 구성되어 있음.
img_train, img_val, label_train, label_val = train_test_split(all_images, all_labels, test_size=0.2, random_state=42)

#plt.imshow(img_train[8], cmap='gray')
#plt.show()

print(label_train[:10])

class_names = ['O', 'X']

print(class_names[label_train[0]])

np.bincount(label_train)
np.bincount(label_val)

img_train = img_train / 255
img_val = img_val / 255

img_train = img_train.reshape(-1, 784)
img_val = img_val.reshape(-1, 784)

#lb = LabelBinarizer()
#lb.fit_transform([0, 1, 3, 1])

tf.keras.utils.to_categorical([0, 1, 3])

# label 인코딩. 0 -> [1, 0]   1 -> [0, 1]
label_train_encoded = tf.keras.utils.to_categorical(label_train)
label_val_encoded = tf.keras.utils.to_categorical(label_val)

# 학습 시작/위까지는 이미지 세팅
fc = MultiClassNetwork(units=100, batch_size=10)    #define된 클래스 이용
fc.fit(img_train, label_train_encoded, 
       img_val=img_val, label_val=label_val_encoded, epochs=120)

plt.plot(fc.losses)
plt.plot(fc.val_losses)
plt.ylabel('loss')
plt.xlabel('iteration')
plt.legend(['train_loss', 'val_loss'])
plt.show()
print(fc.score(img_val, label_val_encoded))
