# 2021/인공지능/인공지능 프로젝트/B611139/이경준

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
import csv
import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler


X_t = []
X_tr = []
y_t = []
y_tr = []
vali = []


def getdata(inputcsv, a):
    file = open(inputcsv, 'r', encoding='utf-8')  # file : 파일객체

    reader = csv.reader(file)  # csv.reader(): for loop을 돌면서 line by line read
    for line in reader:
        a.append(line)  # 날짜 부분은 제외하고 나머지 부분만 append

    file.close()

    # 맨 첫줄 column name 제외하고 data만 xy 매트릭스에 저장
    return np.array(a[1:])


def getclass(inputcsv, a):
    f = open(inputcsv, 'r', encoding='utf-8')  # file : 파일객체

    reader = csv.reader(f)  # csv.reader(): for loop을 돌면서 line by line read
    for line in reader:
        a.append(line)  # 날짜 부분은 제외하고 나머지 부분만 append

    f.close()

    return np.array(a[1:])

# ---------------------------------test, train, validation data 가져오기 ---------------------

X_test = np.delete(getdata('인공지능 프로젝트 데이터/X_test.csv', X_t), 0, axis=1)
X_train = np.delete(getdata('인공지능 프로젝트 데이터/X_train.csv', X_tr), 0, axis=1)
y_test = np.delete(getclass('인공지능 프로젝트 데이터/y_test.csv', y_t), 0, axis=1)
y_train = np.delete(getclass('인공지능 프로젝트 데이터/y_train.csv', y_tr), 0, axis=1)

val = np.delete(getdata('인공지능 프로젝트 데이터/validation.csv', vali), 0, axis=1)
X_val =np.delete(val, 30, axis=1)
y_val = val[:,30:31]
# -------------------------------- 기본 데이터 보존 ------------------------------
X_tr = X_train
y_tr = y_train
X_va = X_val
y_va = y_val
# -------------------------------- data scaling ---------------------------------
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
X_val = sc.transform(X_val)

# ----------------------------------- smote 처리 ---------------------------
smote = SMOTE(random_state=0)
X_train_over,y_train_over = smote.fit_resample(X_train,np.ravel(y_train))

# ---------------------------------- undersampling ----------------
X_train_under, y_train_under = RandomUnderSampler(random_state=0).fit_resample(X_train,np.ravel(y_train))

# ----------------------------------- 학습 및 예측 --------------
# model 0 = 기본 데이터로 만든 모델
#n_estimators = RF에 사용될 트리들의 수, max_depth = 각 트리별 최대 depth(overfitting을 막기위해 지정)
Rforest = RandomForestClassifier(n_estimators=100, oob_score=True, max_depth=8)

Rforest.fit(X_tr, np.ravel(y_tr))

Rfprediction = Rforest.predict(X_va)
print("----------------normal-------------")
print("validation 예측")
print("precision : ", precision_score(y_va, Rfprediction, average=None))
print("recall : ", recall_score(y_va, Rfprediction, average=None))
print("f1_score : ", f1_score(y_va, Rfprediction, average=None))

print('report of validation')
print(classification_report(y_va, Rfprediction))


#-------------------------------------------------------------------------
# model 1 = scaled 된 데이터로 만든 모델

scRforest = RandomForestClassifier(n_estimators=100, oob_score=True, max_depth=8)

scRforest.fit(X_train, np.ravel(y_train))

scRfprediction = scRforest.predict(X_val)
print("----------------scaled-------------")
print("validation 예측")
print("precision : ", precision_score(y_val, scRfprediction, average=None))
print("recall : ", recall_score(y_val, scRfprediction, average=None))
print("f1_score : ", f1_score(y_val, scRfprediction, average=None))

print('report of validation')
print(classification_report(y_val, scRfprediction))

#-------------------------------------------------------------------------
# model 2 = scaled 된 데이터에 smote해서 만든 모델
print("----------------smote-----------------------")

sRforest = RandomForestClassifier(n_estimators=100, oob_score=True, max_depth=8)

sRforest.fit(X_train_over, np.ravel(y_train_over))

sRfprediction = sRforest.predict(X_val)
print("validation 예측")
print("precision : ", precision_score(y_val, sRfprediction, average=None))
print("recall : ", recall_score(y_val, sRfprediction, average=None))
print("f1_score : ", f1_score(y_val, sRfprediction, average=None))

print('report of validation')
print(classification_report(y_val, sRfprediction))

#-------------------------------------------------------------------------
# model 3 = scaled 된 데이터에 Randomundersample해서 만든 모델
print("----------------Randomundersample-----------------------")

uRforest = RandomForestClassifier(n_estimators=100, oob_score=True, max_depth=8)

uRforest.fit(X_train_under, np.ravel(y_train_under))

uRfprediction = uRforest.predict(X_val)
print("validation 예측")
print("precision : ", precision_score(y_val, uRfprediction, average=None))
print("recall : ", recall_score(y_val, uRfprediction, average=None))
print("f1_score : ", f1_score(y_val, uRfprediction, average=None))

print('report of validation')
print(classification_report(y_val, uRfprediction))
