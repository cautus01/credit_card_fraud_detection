# 신용카드 사기거래 탐지하기 팀 프로젝트

## 프로젝트 개요

**진행기간 : 2021.5.6 ~ 2021.5.31**
 
**주요내용**
 
- **주요내용 : 신용카드 정상거래가 99.83%, 사기거래는 0.172%를 차지하는 불균형 데이터를 이용하여 사기거래를 탐지하는 MLP, KNN, Randomforest 모델을 만든다.**

**본인이 기여한점 : MLP 모델을 설계하였다. KNN모델과 MLP모델에서 Recall, Precision, F1 score가 모두 0으로 출력되는 문제를 해결하였다.**

**사용한 skill : Python, KNN, MLP, Randomforest 등**

**어려웠던점: MLP와 KNN으로 설계한 모델들을 실행했을 때 Recall, Precision, F1 score가 모두 0으로 출력되는 문제가 일어났다. 문제의 핵심은 데이터라는 것을 생각해 문제를 해결하기 위해 여러 책과 자료들에서 데이터에 관련된 부분을 보고 공부하였고, 데이터 특성의 스케일이 다르면 모델의 성능에 영향을 줄 수 있다는 것을 깨달아 StandardScaler()로 스케일을 조정하는 전처리를 한 후 MLP와 KNN으로 설계된 모델을 다시 실행하니 문제가 해결됐다.**

**결과**

- KNN는 정확도 : 0.99 , Recall : 0.78, Precision : 0.48, f1-score : 0.60, 

- MLP는 정확도 : 0.99 , Recall : 0.70, Precision : 0.9, f1-score : 0.79, 

- Randomforest는 정확도 : 0.99 , Recall : 0.74, Precision : 0.99, f1-score : 0.85**
