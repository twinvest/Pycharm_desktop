# -*- coding:utf-8 -*-

from numpy.random import seed
import random
from tensorflow import set_random_seed
import os
from keras.models import Sequential, load_model
from keras.layers import Conv2D, Activation, MaxPooling2D
from keras.layers import Dropout, Dense, Flatten
from keras import optimizers
from sklearn.metrics import confusion_matrix, log_loss
import numpy as np
from util import load_data, draw_result, n2c
import time

seed(0)
random.seed(1)
set_random_seed(2)
os.environ['PYTHONHASHSEED'] = '0'


class ModelMgr():
    def __init__(self, target_class=[3, 5], use_validation=True):
        self.target_class = target_class
        self.use_validation = use_validation
        print('\nload dataset')
        if use_validation:
            (self.x_train, self.y_train), (self.x_val, self.y_val), (self.x_test, self.y_test) = \
                load_data(target_class, use_validation=use_validation)
        else:
            (self.x_train, self.y_train), (self.x_test, self.y_test) = \
                load_data(target_class, use_validation=use_validation)

    def train(self):
        print('\ntrain model')

        model = self.get_model()  # 모델 가져오기
        # model = self.get_model_sample_1()  # 예시 모델1
        # model = self.get_model_sample_2()  # 예시 모델2

        hp = self.get_hyperparameter()  # 파이퍼파라미터 로드

        model.summary()  # 모델 구조 출력
        print('hyperparameters :')
        print('\tbatch size :', hp['batch_size'])
        print('\tepochs :', hp['epochs'])
        print('\toptimizer :', hp['optimizer'].__class__.__name__)
        print('\tlearning rate :', hp['learning_rate'])
        # 모델의 손실 함수 및 최적화 알고리즘 설정
        model.compile(optimizer=hp['optimizer'],
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        if hp['epochs'] > 20:  # epochs은 최대 20로 설정 !!
            hp['epochs'] = 20
        if self.use_validation:
            validation_data = (self.x_val, self.y_val)
        else:
            validation_data = (self.x_test, self.y_test)

        # 모델 학습
        history = model.fit(self.x_train, self.y_train,
                            batch_size=hp['batch_size'],
                            epochs=hp['epochs'],
                            validation_data=validation_data,
                            shuffle=False,
                            verbose=2)

        history.history['hypers'] = hp
        self.model = model
        self.history = history

    def get_hyperparameter(self):
        hyper = dict()
        ############################
        '''
        (1) 파라미터 값들을 수정해주세요
       '''
        hyper['batch_size'] = 16  # 배치 사이즈
        hyper['epochs'] = 10  # epochs은 최대 20 설정 !!
        hyper['learning_rate'] = 0.1  # 학습률
        # 최적화 알고리즘 선택 [sgd, rmsprop, adagrad, adam 등]
        hyper['optimizer'] = optimizers.sgd(lr=hyper['learning_rate'])  # default: SGD
        ############################
        return hyper

    def get_model(self):
        model = Sequential()
        ################
        '''
        (2) 모델 코드를 완성해주세요.
        model.add(...)
        '''
        ################
        '''
        주의사항
        1. 모델의 입력 데이터 크기는 (batch_size, 32, 32, 1) # 고양이 or 강아지 흑백 사진
           출력 데이터 크기는 (batch_size, 2) # 고양이일 확률, 강아지일 확률
        2. 최초 Dense() 사용 시, Flatten()을 먼저 사용해야함
        3. out of memory 오류 시,
            메모리 부족에 의한 오류임.
            batch_size를 줄이거나, 모델 구조의 파라미터(ex. 유닛수)를 줄여야함
        4. BatchNormalization() 사용 금지
            
        기타 문의 : sdh9446@gmail.com (수업조교)
        '''
        return model

    def get_model_sample_1(self):
        # Fully-connected layer만을 이용한 모델
        model = Sequential() # 모델 정의
        # Flatten 함수는 다차원 배열을 벡터로 만듦. ex) (32, 32, 1) -> (1024)
        model.add(Flatten(input_shape=self.x_train.shape[1:])) # 초기 레이어에는 input_shape 명시
        model.add(Dense(64))  # fully-connected layer, 64은 히든유닛 개수
        model.add(Activation('sigmoid')) # 활성화 함수 정의 ['sigmoid', 'tanh', relu' 등]
        model.add(Dropout(0.5))  # 이전 output에서 0.5 비율로 랜덤 선택하여 사용. (없어도 무방)
        model.add(Dense(64))
        model.add(Activation('sigmoid'))
        model.add(Dense(len(self.target_class)))  # output 예측. (이 부분은 필수)
        model.add(Activation('softmax'))  # softmax 함수 적용 (이 부분은 필수)

        return model

    def get_model_sample_2(self):
        # CNN을 이용한 모델
        model = Sequential()
        # Conv2D(필터 개수, 필터 크기(X, X), padding='valid'[no padding]/'same'[zero padding])
        model.add(Conv2D(32, (5, 5), padding='valid',
                         input_shape=self.x_train.shape[1:]))
        model.add(Activation('relu'))
        model.add(Conv2D(32, (7, 7), padding='valid'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))  # feature 크기를 1/2 배로 줄임

        model.add(Flatten())
        model.add(Dense(64))
        model.add(Activation('relu'))
        model.add(Dense(len(self.target_class)))
        model.add(Activation('softmax'))

        return model

    def test(self, model=None):
        print('\ntest model')
        if model is None:
            model = self.model
        start = time.time()
        y_pred = model.predict(self.x_test, batch_size=1, verbose=0)
        end = time.time()

        y_true = np.argmax(self.y_test, -1)
        loss = log_loss(y_true, y_pred)
        y_pred = np.argmax(y_pred, -1)
        y_true = np.argmax(self.y_test, -1)
        cmat = confusion_matrix(y_pred, y_true)
        acc_per_class = cmat.diagonal() / cmat.sum(axis=1)

        print('\n===== TEST RESULTS ====')
        print('Test loss:', str(loss)[:6])
        print('Test Accuracy:')
        print('\tTotal: {}%'.format(str(acc_per_class.mean()*100)[:5]))
        for idx, label in n2c.items():
            print('\t{}: {}%'.format(label, str(acc_per_class[idx] * 100)[:5]))
        print('Test FPS:', str(1/((end-start)/len(self.x_test)))[:6])
        print('=======================')
        if hasattr(self, 'history'):
            self.history.history['test_acc'] = acc_per_class.mean()
        return acc_per_class.mean()

    def save_model(self, model_path='./trained_model.h5'):
        print('\nsave model : \"{}\"'.format(model_path))
        self.model.save(model_path)

    def load_model(self, model_path='./trained_model.h5'):
        print('\nload model : \"{}\"'.format(model_path))
        self.model = load_model(model_path)

    def draw_history(self, file_path='./result.png'):
        print('\nvisualize results : \"{}\"'.format(file_path))
        draw_result(self.history.history, self.use_validation, file_path=file_path)


if __name__ == '__main__':
    trained_model = None
    #trained_model = './trained_model.h5'  # 학습된 모델 테스트 시 사용

    modelMgr = ModelMgr()
    if trained_model is None:
        modelMgr.train()
        modelMgr.save_model('./trained_model.h5')  # 모델 저장 (이름이 같으면 덮어씀)
        modelMgr.test()
        modelMgr.draw_history('./result.png')  # 학습 결과 그래프 저장 (./result.png)
    else:
        modelMgr.load_model(trained_model)
        modelMgr.test()
