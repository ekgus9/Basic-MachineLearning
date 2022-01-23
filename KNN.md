# K-NN 알고리즘



K-NN 알고리즘은 흔히 '유유상종'이라는 말로 많이 표현한다. 즉, 비슷한 특성을 가진 요소들이 가까이 모여있다는 것이다. K-NN은 K-Nearest Neighbor으로, K개의 가까운 이웃의 속성에 따라 분류하여 레이블링한다. 이는 머신러닝의 알고리즘 중 지도학습에 속한다. 



* 거리기반 분류분석 모델



K-NN 알고리즘은 거리기반 분류분석 모델로, 상대적으로 거리가 가까운 이웃이 더 가까운 이웃으로 분류되는 모델이다. 
따라서 어떤 새로운 데이터로부터 거리가 가까운 K개의 다른 레이블을 참고하여 K개의 데이터 중 가장 빈도수가 높게 나온 데이터의 레이블로 분류한다.
보통 K가 짝수이면 가까운 이웃 종류의 수가 같아지는 동률이 나올 수 있으므로 홀수값을 사용한다. 
거리를 측정에는 다음과 같은 방법들이 있다. 



1. 유클리드 거리 L2 Distance : K-NN 알고리즘에서 가장 일반적인 방법으로 2차원 평면의 두점 사이 거리를 유클리드 거리 계산법에 의해 도출하는 방법이다.

```
d(A,B) = sqrt((x2-x1)^2 + (y2-y1)^2)
```

2. 맨해튼 거리 L1 Distance : 유클리드 공식과는 달리 직선으로 이동할 수 없는 건물들이 많은 체계적인 지역에서 거리를 재기 위해 탄생한 공식이다.

```
d(A,B) = |x1-x2| + |y1-y2|
```

* 변수값 범위 재조정



분류 모델이 상대적으로 큰 단위(범위)를 가지는 경우, 상대적으로 작은 값을 가지는 변수는 무시될 수 있으므로 변수값에 대한 범위 재조정이 필요하다. 다음은 변수값 범위 재조정의 방법들이다.



1. 최소-최대 정규화 : 변수의 범위 0% - 100%

```
Z = (X−min(X)) / (max(X)−min(X))
```

2. z-점수 표준화 : 변수의 범위 정규 분포화 -> 평균 0, 표준편차 1

```
Z = (X−평균) / 표준편차
```

* K-NN 구현

```
import numpy as np
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

iris = load_iris()
knn = KNeighborsClassifier(n_neighbors = 3) # K = 3

X_train, X_test, Y_train, Y_test = train_test_split(iris['data'], iris['target'], test_size=0.25, stratify = iris['target'])
knn.fit(X_train, Y_train)

y_pred = knn.predict(X_test)
print("prediction accuracy: {:.2f}".format(np.mean(y_pred == Y_test)))
```

참고 : <https://velog.io/@jhlee508/%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D-KNNK-Nearest-Neighbor-%EC%95%8C%EA%B3%A0%EB%A6%AC%EC%A6%98>
