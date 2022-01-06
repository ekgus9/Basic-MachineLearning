# 1. 머신러닝의 개념
    # 어떤 데이터를 바탕으로 학습이 가능한 프로그램

    # 지도 학습 vs 비지도학습
        # label 기반 학습 : 지도 학습 Supervised learning
    
    # Supervised learning
        # Training data set 필요
    
        # 학습 데이터
            # regression : 1~100
            # binary classification : pass or non-pass
            # multi-label classification : A,B,C,D,F
        
    # TensorFlow
        # Data Flow Graph
    
        # pip install --upgrade tensorflow
    
    # 그래프 설계 -> sess.run(op, feed_dict={x:x_data}) -> 그래프 실행
    
        # Rank : [] [[]] [[[]]]
        # Shape : [] [1] [1,2]
        # Type : tf.float32 tf.int32
    
import tensorflow as tf

# 버전 확인
tf.__version__ 

# 출력
hello = tf.constant("Hello, TensorFlow!")

print(hello)

# 연산
node1 = tf.constant(3.0)
node2 = tf.constant(4.0)
node3 = tf.add(node1,node2)

print(node1,node2,node3)

@tf.function # 함수 사용 가능
def add(a,b): return a+b

print(add(node1,node2))


# 2. 선형 회귀 Linear regression
    # H(x) = Wx + b
    
    # Cost function
        # 얼마나 training data와 실제 line이 일치하는가
        # (H(x)-y)^2 -> 차이가 클 때 패널티
        # cost 최소화가 목표!
        
import numpy as np

x_train = [1,2,3]
y_train = [1,2,3]

W = tf.Variable(tf.random.normal([1]),name = 'weight')
b = tf.Variable(tf.random.normal([1]),name = 'bias')

hypothesis = x_train * W + b

cost = tf.reduce_mean(tf.square(hypothesis - y_train))
op = tf.keras.optimizers.SGD(learning_rate = 0.01)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(1,input_dim=1))

model.compile(loss='mean_squared_error',optimizer=op)
	
model.fit(x_train,y_train,epochs=1000)
 
print(model.predict(np.array([5])))
    
    
# 3. Linear Regression에서 Cost 최소화 알고리즘
    # U
    
    # Gradient descent algorithm 경사를 따라 내려가는 알고리즘
        # 최소화하는 (W, b)값 찾아줌
        # 미분
        # Convex function 일때 항상 최소값으로 수렴
        

# 4. Multi-variable Linear Regression
    # H(x1,x2,x3) = w1x1 + w2x2 + w3x3 + b
    
    # Matrix multiplication
        # H(X) = XW