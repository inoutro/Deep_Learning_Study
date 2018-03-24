# Deep_Learning_Study
Deep Learning Theory class Assignment


본 프로젝트는 고려대학교 뇌공학과 딥러닝이론(BRI641) 수업에서 진행한 과제로, functional magnetic resonance imaging(fMRI) 데이터를 활용하여 머신러닝 기법인 Deep Neural Network를 적용한 코드이다.

## 1. Data description
fMRI 데이터는 12명의 참가자가 left-and clenching (LH), right-hand clenching (RH), auditory attention (AD), and visual stimulus (VS) 총 4가지의 실험에 대하여 각각 30번씩 반응하여 1,440개의 관측치가 존재한다. 기존에 fMRI 데이터는 1d-array와 3d-array가 존재하고, 본 코드에서 사용된 데이터는 1d-array다. 본 과제의 목적은 환자의 fMRI 데이터를 가지고 위의 LH, RH, AD, VS 4가지로 분류하는 모델을 만드는 것이다.

## 2. BRI641_DNN_code.py
Deep Neural Network의 기본 Fully-connected neural network를 구축하였고, Hidden layer는 총 2개로 구성하였다. 각 Sample의 input이 74,484개로 들어가고, 첫 번째 Hidden layer node가 300개, 두 번째 Hidden layer가 100개 그리고 마지막으로 output layer의 노드는 본 과제에서 분류하고자 하는 4개의 class에 따라 4개로 하였다.

## 3. BRI641_CNN_code.py
Convolution Neural Network를 구축하였고, 본 과제에서 사용하는 데이터가 1d-array 이므로 1-dimension CNN을 구축하였다. 일반 CNN에서 input이미지의 높이를 1로 설정하였다. 즉, 1*74484의 input이 들어갔다. CNN Architecture를 정리하면 Input --> Convolution layer --> Convolution layer --> Pooling layer --> Convolution layer --> Pooling layer --> Neural Network --> output layer로 구성되어 있다.

## 4. 평가 및 결과
CNN, DNN의 비교를 위하여 동일하게 각각 Dropout과 Max-Norm Regularization을 적용하였고, 각 모델의 평가 및 비교를 위하여 Leave One Out Cross Validation을 적용하였다. 결과적으로 DNN은 93.8%의 정확도를 보였고, CNN은 96.8%의 정확도를 보였다. DNN 보다는 CNN에서 더 정확하게 예측한 것을 볼 수 있었다. fMRI 데이터 자체가 이미지 데이터이므로 CNN에서 더 높은 정확도를 보인 것으로 생각된다.
