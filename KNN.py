from sklearn.neighbors import KNeighborsClassifier
import sys, os
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report # 평가를 위한 테스트 도구


sys.path.append(os.pardir)


X_valid = pd.read_csv("./data/validation.csv", sep=',')
X_train = pd.read_csv("./data/X_train.csv", sep=',')
X_test = pd.read_csv("./data/X_test.csv", sep=',')
y_train = pd.read_csv("./data/y_train.csv", sep=',')
y_test = pd.read_csv("./data/y_test.csv", sep=',')

X_train = X_train[['Time','V1','V2','V3','V4','V5','V6','V7','V8','V9','V10','V11','V12','V13','V14','V15','V16','V17','V18','V19','V20','V21','V22','V23','V24','V25','V26','V27','V28','Amount']]
X_test = X_test[['Time','V1','V2','V3','V4','V5','V6','V7','V8','V9','V10','V11','V12','V13','V14','V15','V16','V17','V18','V19','V20','V21','V22','V23','V24','V25','V26','V27','V28','Amount']]
y_train = y_train['Class']
y_test = y_test['Class']
# validation set을 class와 나머지 colunm으로 나눈다.
y_valid = X_valid['class']
X_valid = X_valid[['Time','V1','V2','V3','V4','V5','V6','V7','V8','V9','V10','V11','V12','V13','V14','V15','V16','V17','V18','V19','V20','V21','V22','V23','V24','V25','V26','V27','V28','Amount']]

# preprocessing작업으로 sklearn의 StandardScaler를 적용한 데이터 셋
st = StandardScaler()
st.fit(X_train)
X_train_scale = st.fit_transform(X_train)
X_test_scale = st.transform(X_test)

smote_model = SMOTE(random_state=0) # 불균형 데이터인것을 처리하기위해 SMOTE 를 사용하여 오버 샘플링
X_smote, y_smote = smote_model.fit_resample(X_train,y_train)
X_train_scale_smote, y_train_scale_smote = smote_model.fit_resample(X_train_scale,y_train)
# RandomUnderSampler를 사용하여 언더 샘플링
X_under, y_under = RandomUnderSampler(random_state =0).fit_resample(X_train,y_train)
X_scale_under, y_scale_under = RandomUnderSampler(random_state =0).fit_resample(X_train_scale,y_train)
print("KNN Classifier")

best_K = 0
scores = []
# k가 1부터 99인경우를 모두 돌려보고, 그중 제일 정확도가 좋은 최적의 k를 찾는다. 너무 오래걸려 실행 후에 주석처리하였다.
#for i in range(1,100):
#    model = KNeighborsClassifier(n_neighbors=i)
#    model.fit(X_train, y_train)
#    score = model.score(X_valid, y_valid)
#    print("k : ",i,"accuracy : ",score)
#   scores.append(score)

#best_K = np.argmax(scores) # 가장 높은 정확도를 가진 index
#print("best k :",best_K + 1) # index에 1을 더하여 k값을 얻는다.
# 최적의 k 를 이용해 test set으로 결과를 얻는다. 최적의 k로 3값을 얻었다.

# 최적의 k =3 일때, smote를 이용하여 결과를 측정.
print("Default KNN - SMOTE")
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_smote, y_smote)
pred = model.predict(X_test)
print(classification_report(y_test, pred,))
# k=3일때, under sampling을 이용하여 결과를 측정.
print("Default KNN - UnderSampling")
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_under, y_under)
pred = model.predict(X_test)
print(classification_report(y_test, pred,))

# StandardScaler로 전처리 작업을한 데이터를 사용하여 측정
# 최적의 k =3 일때, smote를 이용하여 결과를 측정.
print("StandardScaler - SMOTE")
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train_scale_smote, y_train_scale_smote)
pred = model.predict(X_test_scale)
print(classification_report(y_test, pred,))
# k=3일때, under sampling을 이용하여 결과를 측정.
print("StandardScaler - UnderSampling")
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_scale_under, y_scale_under)
pred = model.predict(X_test_scale)
print(classification_report(y_test, pred,))
