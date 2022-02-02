# LSTM (Long Short-Term memory)



LSTM은 **RNN (Recurrent Neural Network)** 의 한 종류이다. RNN은 스스로 반복하면서 이전 단계에 얻은 정보가 지속될 수 있도록 한다. 
그러나 RNN은 필요한 정보와의 시간적 격차가 커질수록  그 정보를 이어나가는 일에 어려움을 겪는다. 아래 그림은 긴 기간에 의존하는 RNN이다.



![image](https://user-images.githubusercontent.com/89879599/152079585-536740d5-1574-4c10-9433-db915f1a3921.png)



LSTM은 RNN의 이러한 단점을 보완할 수 있는 모델이다. 즉, 긴 의존 기간을 필요로 하는 학습을 수행할 수 있는 능력을 지니고 있다. 다음 그림은 LSTM의 반복 모듈이다.



![image](https://user-images.githubusercontent.com/89879599/152079789-319479a4-0870-496c-a53b-b523eb89d3c1.png)



LSTM의 첫단계는 sigmoid layer에 의해 어떤 정보를 버릴 지 말지를 결정하는 것이다. 
다음 단계는 새로 들어오는 정보를 저장할 것인지 결정하고, 이를 tahn layer에서 나온 정보와 합쳐 업그레이드 하는 것이다.
그리고 첫단계와 다음 단계에서 나온 정보들 어떻게 출력할 것인가를 결정하게 된다.



설명한 LSTM의 기본형 말고도 LSTM은 수많은 변형 모델을 가지고 있다. 따라서 필요에 따라 더 좋은 결과를 내는 모델을 만들 수도 있다.
