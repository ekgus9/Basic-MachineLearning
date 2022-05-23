# GAN (Generative Adversarial Network)



![image](https://user-images.githubusercontent.com/89879599/168525553-d32b1202-1e7c-446b-a3dd-1c534ec994f8.png)



1. z = noise
2. generator이 위조 지폐 만든다. -> G(z)
3. discriminator이 위조 지폐 0 와 실제 지폐 1 구분 
4. 이렇게 한 번 경찰이 해당 지폐 구분한게 epoch
5. 어느 순간 완벽한 위조지폐가 탄생하면 경찰은 결국 찍기 시작하고 확률은 50%가 된다. 



discriminator은 어떠한 input data가 들어왔을때 해당 값이 어떤 것인지 classify (지도)



generator은 latent code를 가지고 이 data가 training data가 되도록 학습 (비지도)



-> generator model은 실제 데이터 분포가 나타내는 확률 분포 그래프와 유사한 모델을 제작하려는 목적



D(G(z)) == 1이 되도록하는 것이 목적



### 1. 수식



![image](https://user-images.githubusercontent.com/89879599/168527130-eccbdcd5-d2ab-4b59-a8ca-786c1e897efa.png)



왼쪽 : G는 V(D,G)가 최소가 되려하고, D는 V(D,G)가 최대가 되려고 한다는 의미이다.



      log(1) = 0
      log(0) = -infinity
      



**D의 입장**



![image](https://user-images.githubusercontent.com/89879599/168527803-5d444c20-8e07-481a-829f-fa34f1a2a1fc.png)



D는 오른쪽 수식 중 D(x) = 1, D(G(z)) = 0이 목표



D(x) = 1이 D가 뽑을 수 있는 가장 큰 값



**G의 입장**



![image](https://user-images.githubusercontent.com/89879599/168528669-11c2525f-336d-4043-980a-9cb53c14641c.png)



앞에서 D가 어떻게 하던지 상관 없이, D(x) = 0, D(G(z) = 1이 되도록 한다.



G가 원하는 최적의 상황은 음의 무한대 방향



### GAN 종류



1. Deep convolutional GAN (DCGAN)

G와 D에서 convolution layer 를 사용하며, poolinglayer 을 사용하지 않는다.



GAN의 안정성 문제 해결



2. Least Squares GAN (LSGAN)




![image](https://user-images.githubusercontent.com/89879599/169772918-efa34d72-ee9e-4473-b4d5-734b8d2729d0.png)


 
 제대로 학습된 GAN의 D의 경우 decision boundary를 아래는 진짜, 위를 가짜로 구분한다고 할 때, 별로 표시된 가짜가 진짜로 구분되었음을 볼 수 있다. 
 
 G의 입장에서는 잘 속였기 때문에 더이상 학습할 의지가 없다. (gradient vanishing)
 
 
 

 하지만 이 가짜의 경우 진짜와 멀리 떨어져있다. 따라서 가짜를 진짜의 가져오자는게 논문의 아이디어이다.
 
 
 
 3. Semi-Supervised GAN (SGAN)

D가 진짜, 가짜를 구분하는 것뿐만 아니라 class를 구분하며 추가로 fake라는 class를 구분한다. 
 
