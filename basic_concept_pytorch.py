# 필요 패키지 모음

import torch
import torch.nn as nn
import numpy as np

import random
# 텐서 만들기

## 2차원 텐서 - 연산 수행을 위한 기본 객체 == Tensor
x = torch.Tensor([[1,2], [3,4]])
x = torch.from_numpy(np.array([[1,2], [3,4]]))

## 2차원 텐서 - 넘파이를 활용한 행렬 생성 == numpy

x = np.array([[1,2], [3,4]])

### 책의 92p를 참고하면, 파이토치의 텐서 자료형에 대해 알 수 있음.

# AUtograd
## 파이토치는 자동으로 미분 및 역전파 수행하는 autograd 기능이 있음.

x = torch.FloatTensor(2,2)
y = torch.FloatTensor(2,2)

y.requires_grad_(True)

z = (x + y) + torch.FloatTensor(2,2)

## z로부터 역전파를 수행하면, 미분 값을 전달 가능

## 기울기를 구할 필요가 없는 경우에는
## with 문법을 사용하여 연산 수행

x = torch.FloatTensor(2,2)
y = torch.FloatTensor(2,2)
y.requires_grad_(True)

# with 구문을 통해서 역전파 수행을 막아, 메모리 사용 효율
with torch.no_grad():
    z = (x + y) + torch.FloatTensor(2,2)


# 피드 포워드

## y = xW + b (x는 MxN 행렬, W는 NxP 행렬, b는 P열 벡터)

## 순전파 기능만 하도록 구현

def linear(x, W, b):
    y = torch.mm(x, W) + b

    return y

    x = torch.FloatTensor(16, 10)
    W = torch.FloatTensor(10, 5)
    b = torch.FloatTensor(5)

y = linear(x, W, b)

# nn.Module
## MyLinear라는 클래스를 nn.Module을 상속 받아 선언하기

class MyLinear(nn.Module):

    def __init__(self, input_size, output_size):
        super().__init__()

        self.W = torch.FloatTensor(input_size, output_size)
        self.b = torch.FloatTensor(output_size)


    def forward(self, x):
        y = torch.mm(x, self.W) + self.b

        return y


x = torch.FloatTensor(16,10)
linear = MyLinear(10,5)
y = linear(x)  ## 10개의 원소 -> 5개의 원소로 객체 변환

## 그러나 학습 파라미터가 아직 없는 상황

params = [p.size() for p in linear.parameters()]
print(params) ## 결과로 [] 반환


## parameters 클래스를 통해 텐서를 감싸주어야 함.

Class MyLinear(nn.Module):

    def __init__(self, input_size, output_size):
        super(MyLinear, self).__init__()

        self.W = nn.Parameter(torch.FloatTensor(input_size, output_size),
        requires_grad = True)
        self.b = nn.Parameter(torch.FloatTensor(output_size),
        requires_grad = True)

    def forward(self, x):
        y = torch.mm(x, self.W) + self.b

        return y

## 학습 파라미터 파악

params = [p.size() for p in linear.parameters()]
print(params) ## 결과로 torch.Size([10,5]), torch.Size([5]) 반환

## 더 깔끔하게 모델링

Class MyLinear(nn.Module):

    def __init__(self, input_size, output_size):
        super(MyLinear, self).__init__()

        self.linear = nn.Linear(input_size, output_size)

    def forward(linear):
        y = self.linear(x)

        return y

## nn.Linear 클래스를 사용해서 W와 b를 대체

# 역전파 수행
## 오류 (손실) 값을 뒤로 전달하는 역전파 알고리즘 수행
## linear의 결과값 텐서의 합과 목표값과의 거리 (Error 또는 loss를 구하기)
## 그 거리 값에 backward()함수를 적용하여 기울기를 구함.
## 이 때 손실 값은 스칼라 상태로, 백터나 행렬이 아님.

objective = 100

x = torch.FloatTensor(16,10)
linear = MyLinear(10,5)
y = linear(x)

loss = (objective - y.sum())2
loss.backward()

# training 수행

linear.eval() # 훈련 모드
linear.train() # 추론 모드
## dropout 및 batch-normalization과 같은 학습과 추론이 다른 경우에 사용
## 추론이 끝나면 다시 train() 선언을 해서 원래의 훈련 모드로 돌아가게 해야 함.


# 선형회귀분석

# step1. 모듈 생성

class MyModel(nn.Module):

    def __init__(self, input_size, output_size):
        super(MyModel, self).__init__()


        self.linear = Linear(input_size, output_size)

    def forward(self, x):
        y = self.linear(x)

        return y

# 손실함수 최소로 만드는 함수

def ground_truth(x):
    return 3 * x[:,0] + x[:, 1] - 2 * x[:,2]

# 역전파 알고리즘 수행 및 경사하강법 적용

def train(model, x, y, optim):

    optim.zero_grad()


    # feed-forward
    y_hat = model(x)
    # 손실 값 찾기
    loss = ()(y - y_hat)2).sum() / x.size(0)
    # 역전파 적용
    loss.backward()


    # 경사하강법 1회 적용
    optim.step()

    return loss.data


# 하이퍼 파라미터 설정

batch_size = 1
n_epochs = 1000
n_iter = 10000

model = Mymodel(3, 1)
optim = torch.optim.SGD(model.parameters(), lr = 0.001, momentum = 0.1)

print(model)


# 손실값 줄이기 훈련

for epoch in range(n_epochs):
    avg_loss = 0

    for i in range(n_iter):
        x = torch.rand(batch_size, 3)
        y = ground_truth(x.data)

        loss = train(model, x, y, optim)

        avg_loss += loss
        avg_loss = avg_loss / n_iter

    # 간단한 테스트용 텐서
    x_valid = torch.FloatTensor([[.3, .2, .1]])
    y_valid = ground_truth(x_valid.data)


    model.eval()
    y_hat = model(x_valid)
    model.train()

    print(avg_loss, y_valid.data[0], y_hat.data[0,0])

    if avg_loss < 0.001:
        break

        
