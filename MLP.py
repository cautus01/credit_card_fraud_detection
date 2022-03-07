import os,sys
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
# csv 파일 불러오기
validation= pd.read_csv('validation.csv', sep=',')
X_train= pd.read_csv('X_train.csv', sep=',')
y_train= pd.read_csv('y_train.csv', sep=',')
X_test= pd.read_csv('X_test.csv', sep=',')
y_test= pd.read_csv('y_test.csv', sep=',')

X_train = X_train[['Time','V1','V2','V3','V4','V5','V6','V7','V8','V9','V10','V11','V12','V13','V14','V15','V16','V17','V18','V19','V20','V21','V22','V23','V24','V25','V26','V27','V28','Amount']]
X_test = X_test[['Time','V1','V2','V3','V4','V5','V6','V7','V8','V9','V10','V11','V12','V13','V14','V15','V16','V17','V18','V19','V20','V21','V22','V23','V24','V25','V26','V27','V28','Amount']]
y_train = y_train['Class']
y_test = y_test['Class']

# validation set을 class와 나머지 colunm으로 나눈다.
X_val = validation[['Time','V1','V2','V3','V4','V5','V6','V7','V8','V9','V10','V11','V12','V13','V14','V15','V16','V17','V18','V19','V20','V21','V22','V23','V24','V25','V26','V27','V28','Amount']]
y_val=validation['class']

# 데이터 스케일링, fit은 훈련 데이터에만,transformer는 훈련데이터,테스트 데이터,검증데이터에 적용한다.
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
X_val = sc.transform(X_val)

# relu는 은닉층에 사용, 이진 분류 문제이므로 마지막 층은 sigmoid 사용
# Cost-sensitive learning 방법, dropout은 0.25로 설정
# 반복문을 통해 최적의 class_weight를 구해냈다.
class_weight = {0: 0.27,1: 0.73}
model = Sequential(
    [
        Dense(20, input_shape=(30,), activation='relu'),
        Dropout(0.25),
        Dense(20, activation='relu'),
        Dropout(0.25),
        Dense(1, activation='sigmoid')
    ]
)
# 이진 분류 문제이므로 binary_crossentropy를 사용하였다.
# batch size는 15로, epochs는 2로 설정
model.compile(optimizer='rmsprop', loss='binary_crossentropy',metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=15, epochs=2,class_weight=class_weight)

# validation 결과 출력
print("validation 예측")
y_pred = (model.predict(X_val) > 0.5).astype("int32")
print("정확도")
print(accuracy_score(y_val, y_pred))
print(classification_report(y_val, y_pred))
# 테스트 데이터 결과 출력
print("테스트 예측")
y_pred = (model.predict(X_test) > 0.5).astype("int32")
print("정확도")
print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# SMOTE 데이터 처리
smote = SMOTE(random_state=0)
X_train_over,y_train_over = smote.fit_resample(X_train,y_train)

# relu는 은닉층에 사용, 이진 분류 문제이므로 마지막 층은 sigmoid 사용
# SMOTE 방법, dropout은 0.25로 설정
model1 = Sequential(
    [
        Dense(20, input_shape=(30,), activation='relu'),
        Dropout(0.25),
        Dense(20, activation='relu'),
        Dropout(0.25),
        Dense(1, activation='sigmoid')
    ]
)
# 이진 분류 문제이므로 binary_crossentropy를 사용하였다.
model1.compile(optimizer='rmsprop', loss='binary_crossentropy',metrics=['accuracy'])
model1.fit(X_train_over, y_train_over, batch_size=15, epochs=2)

# validation 결과 출력
print("validation 예측")
y_pred = (model1.predict(X_val) > 0.5).astype("int32")
print("정확도")
print(accuracy_score(y_val, y_pred))
print(classification_report(y_val, y_pred))
# 테스트 데이터 결과 출력
print("테스트 예측")
y_pred = (model1.predict(X_test) > 0.5).astype("int32")
print("정확도")
print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))




