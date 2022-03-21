# credit_card_fraud_detection

## 프로젝트 개요

**진행기간 : 2021.5.6 ~ 2021.5.31**
 
**주요내용**
 
- **불균형 신용카드 거래 데이터를 이용하여 KNN, MLP, Randomforest 모델을 이용하여 '신용카드 사기거래 탐지'하기**

**사용한 skill : Python, KNN, MLP, Randomforest**

**어려웠던점: MLP와 KNN으로 설계한 모델들을 실행했을 때 Recall, Precision, F1 score가 모두 0으로 출력되는 문제가 일어났다. 문제의 핵심은 데이터라는 것을 생각해 문제를 해결하기 위해 여러 책과 자료들에서 데이터에 관련된 부분을 보고 공부하였고, 데이터 특성의 스케일이 다르면 모델의 성능에 영향을 줄 수 있다는 것을 깨달아 StandardScaler()로 스케일을 조정하는 전처리를 한 후 MLP와 KNN으로 설계된 모델을 다시 실행하니 문제가 해결됐습니다..**

**결과 : KNN는 정확도 : 0.99 , Recall : 0.78, Precision : 0.48, f1-score : 0.60, MLP는 정확도 : 0.99 , Recall : 0.85, Precision : 0.95, f1-score : 0.89, Randomforest는 정확도 : 0.99 , Recall : 0.74, Precision : 0.99, f1-score : 0.85**
