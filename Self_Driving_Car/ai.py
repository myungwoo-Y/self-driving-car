# # AI for Self Driving Car
#
# # Importing the libraries
#
# import numpy as np
# import random
# import os
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# import torch.autograd as autograd
# from torch.autograd import Variable
#
# # Creating the architecture of the Neural Network
#
# class Network(nn.Module):
#
#     def __init__(self, input_size, nb_action):
#         super(Network, self).__init__()
#         self.input_size = input_size
#         self.nb_action = nb_action
#         self.fc1 = nn.Linear(input_size, 30)
#         self.fc2 = nn.Linear(30, nb_action)
#
#     def forward(self, state):
#         x = F.relu(self.fc1(state))
#         q_values = self.fc2(x)
#         return q_values
#
# # Implementing Experience Replay
#
# class ReplayMemory(object):
#
#     def __init__(self, capacity):
#         self.capacity = capacity
#         self.memory = []
#
#     def push(self, event):
#         self.memory.append(event)
#         if len(self.memory) > self.capacity:
#             del self.memory[0]
#
#     def sample(self, batch_size):
#         samples = zip(*random.sample(self.memory, batch_size))
#         return map(lambda x: Variable(torch.cat(x, 0)), samples)
#
# # Implementing Deep Q Learning
#
# class Dqn():
#
#     def __init__(self, input_size, nb_action, gamma):
#         self.gamma = gamma
#         self.reward_window = []
#         self.model = Network(input_size, nb_action)
#         self.memory = ReplayMemory(100000)
#         self.optimizer = optim.Adam(self.model.parameters(), lr = 0.001)
#         self.last_state = torch.Tensor(input_size).unsqueeze(0)
#         self.last_action = 0
#         self.last_reward = 0
#
#     def select_action(self, state):
#         probs = F.softmax(self.model(Variable(state, volatile = True))*100) # T=100
#         action = probs.multinomial(1)
#         return action.data[0,0]
#
#     def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
#         outputs = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
#         next_outputs = self.model(batch_next_state).detach().max(1)[0]
#         target = self.gamma*next_outputs + batch_reward
#         td_loss = F.smooth_l1_loss(outputs, target)
#         self.optimizer.zero_grad()
#         td_loss.backward(retain_graph=True)
#         self.optimizer.step()
#
#     def update(self, reward, new_signal):
#         new_state = torch.Tensor(new_signal).float().unsqueeze(0)
#         self.memory.push((self.last_state, new_state, torch.LongTensor([int(self.last_action)]), torch.Tensor([self.last_reward])))
#         action = self.select_action(new_state)
#         if len(self.memory.memory) > 100:
#             batch_state, batch_next_state, batch_action, batch_reward = self.memory.sample(100)
#             self.learn(batch_state, batch_next_state, batch_reward, batch_action)
#         self.last_action = action
#         self.last_state = new_state
#         self.last_reward = reward
#         self.reward_window.append(reward)
#         if len(self.reward_window) > 1000:
#             del self.reward_window[0]
#         return action
#
#     def score(self):
#         return sum(self.reward_window)/(len(self.reward_window)+1.)
#
#     def save(self):
#         torch.save({'state_dict': self.model.state_dict(),
#                     'optimizer' : self.optimizer.state_dict(),
#                    }, 'last_brain.pth')
#
#     def load(self):
#         if os.path.isfile('last_brain.pth'):
#             print("=> loading checkpoint... ")
#             checkpoint = torch.load('last_brain.pth')
#             self.model.load_state_dict(checkpoint['state_dict'])
#             self.optimizer.load_state_dict(checkpoint['optimizer'])
#             print("done !")
#         else:
#             print("no checkpoint found...")

import numpy as np
# random 으로 실행하기 위해 호출
import random
# 운영체제에서 제공하는 기능들을 사용하기 위해 호출
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# 자동 미분 제공한다. 텐서에 관한 대부분의 연산을 지원한다.
import torch.autograd as autograd
from torch.autograd import Variable

# creating the architecture of the Neural network

# nn.Module을 상속받는다.
class Network(nn.Module):

    def __init__(self, input_size, nb_action):
        # nn.Module을 사용하기 위한 트릭
        super(Network, self).__init__()

        # input size
        self.input_size = input_size
        # output size
        self.nb_action = nb_action

        # input layer와 hidden layer를 full connection 하게 해준다.
        # full connection 이란 layer들 끼리 연결 하는 것이다.
        self.fc1 = nn.Linear(input_size, 30)
        # hidden layer와 ouput layer를 full connection 하게 해준다.
        self.fc2 = nn.Linear(30, nb_action)

    # input(state)에 기반하여 q-value들을 반환한다.
    def forward(self, state):
        # state를 input으로 주어 input-hidden 네트워크를 통과시킨다.
        x = F.relu(self.fc1(state))

        # q_value 들을 얻는다.
        q_values = self.fc2(x)
        return q_values

# Implementing Experience Replay
class ReplayMemory(object):
    # capacity : 몇개의 event 를 저장할지 나타냄
    def __init__(self, capacity):
        self.capacity = capacity

        # event 가 저장될 list
        self.memory = []

    # memory 에 event를 집어넣는 함수
    # event : [last state, new state, last action, last reword]
    def push(self, event):
        self.memory.append(event)

        # 항상 size가 capacity 를 유지하게 만든다.
        if len(self.memory) > self.capacity:
            del self.memory[0]

    # memory에서 랜덤으로 event를 고르는 함수
    # batch_size : 한번의 batch 마다 주는 data size
    def sample(self, batch_size):
        # if list = ((1,2,3),(4,5,6)), then zip(*list) = ((1,4),(2,5),(3,6))
        # 즉 같은 인덱스의 원소끼리 묶어 새로운 list를 생성해준다.
        # input에 맞게 list를 형성한다.
        samples = zip(*random.sample(self.memory, batch_size))

        # 배열 내부의 원소들을 이어붙이기 위해 cat 을 사용한다.
        # return : pytorch type의 variable 들
        return map(lambda x: Variable(torch.cat(x, 0)), samples)


# implementing Deep Q Learning

class Dqn():

    def __init__(self, input_size, nb_action, gamma):
        self.gamma = gamma
        self.reward_window = []
        self.model = Network(input_size, nb_action)
        self.memory = ReplayMemory(100000)

        # 최적의 값을 찾는다. ex) 최솟값 or 최댓값
        # 손실 함수의 값을 최소화 하는 파라미터의 값을 찾아준다.
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        # batch 형식에 맞추기 위하여 tensor로 변환 후 fake demesion을 추가한다.
        self.last_state = torch.Tensor(input_size).unsqueeze(0)

        self.last_action = 0
        self.last_reward = 0

    def select_action(self, state):
        # softmax([1,2,3]) = [0.04. 0.11, 0.85) => softmax([1,2,3]*3) = [0, 0.02, 0.98]
        # T 만큼 곱해주면 곱하기 전과 비교하여 값들간의 차이가 많이 나기때문에
        # 더 큰 수를 곱해 줄 수록 값이 낮은 q-value에 대해서 확률값은 더 작아지고
        # 값이 큰 q-value 의 확률값은 커진다.
        # 즉, q-value 간 확률값의 편차가 커진다.
        probs = F.softmax(self.model(Variable(state, volatile=True))*100) # T=7

        # 확률에 맞게 action을 선탣한다.
        action = probs.multinomial(1)

        return action.data[0, 0]

    # def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
    #     outputs = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
    #     next_outputs = self.model(batch_next_state).detach().max(1)[0]
    #     target = self.gamma*next_outputs + batch_reward
    #     td_loss = F.smooth_l1_loss(outputs, target)
    #     self.optimizer.zero_grad()
    #     td_loss.backward(retain_graph=True)
    #     self.optimizer.step()

    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        # state 와 같은 차원을 만들어 주기 위하여 batch_action.unsqueeze(1)
        # squeeze(1)을 통하여 데이터를 간단하게 만들어 준다.
        # prediction 이라고 할 수 있다.
        outputs = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)

        # 1은 action, 0은 state를 의미한다.
        # 다음 state의 최대 q value를 의미한다.
        next_outputs = self.model(batch_next_state).detach().max(1)[0]

        target = self.gamma*next_outputs + batch_reward

        td_loss = F.smooth_l1_loss(outputs, target)

        # reinitialize
        self.optimizer.zero_grad()
        td_loss.backward(retain_graph=True)
        self.optimizer.step()

    def update(self, reward, new_signal):
        new_state = torch.Tensor(new_signal).float().unsqueeze(0)
        self.memory.push((self.last_state, new_state, torch.LongTensor([int(self.last_action)]), torch.Tensor([self.last_reward])))
        action = self.select_action(new_state)
        if len(self.memory.memory) > 100:
            batch_state, batch_next_state, batch_action, batch_reward = self.memory.sample(100)
            self.learn(batch_state, batch_next_state, batch_reward, batch_action)
        self.last_action = action
        self.last_state = new_state
        self.last_reward = reward
        self.reward_window.append(reward)
        if len(self.reward_window) > 1000:
            del self.reward_window[0]
        return action

    def score(self):
        return sum(self.reward_window)/(len(self.reward_window)+1.)

    def save(self):
        torch.save({'state_dict': self.model.state_dict(),
                    'optimizer' : self.optimizer.state_dict(),
                   }, 'last_brain.pth')

    def update(self, reward, new_signal):
        # new_signal 은 state로서
        # tensor를 이용하여 데이터를 간단하게 변화시킨다.
        # 그 후 unsqueeze 를 이용해 새로운 차원을 생성한다.
        new_state = torch.Tensor(new_signal).float().unsqueeze(0)

        # Layer 에 들어가는 모든것들은 Tensor 화 한 뒤 넣어야 한다.
        self.memory.push((self.last_state, new_state, torch.LongTensor([int(self.last_action)]), torch.Tensor([self.last_reward])))

        # 다음 state로 이동하는 action 선택
        action = self.select_action(new_state)

        # 메모리에 쌓인 데이터가 많으면 학습한다.
        if len(self.memory.memory) > 100:
            # 샘플 데이터를 가져온다.
            # 그 후 새로 넣은 데이터 까지 포함하여 학습한다.
            batch_state, batch_next_state, batch_action, batch_reward = self.memory.sample(100)
            self.learn(batch_state, batch_next_state, batch_reward, batch_action)

        # 학습 후 최근 상태 모두 업데이트
        self.last_action = action
        self.last_state = new_state
        self.last_reward = reward
        self.reward_window.append(reward)

        # reward_window 길이 조절
        if len(self.reward_window) > 1000:
            del self.reward_window[0]

        return action

    # reward 의 평균을 알려준다.
    def score(self):
        return sum(self.reward_window)/(len(self.reward_window)+1.)

    # 현재 모델을 저장한다.
    def save(self):
        # state 와 optimizer를 last_brain.pth 에 저장한다.
        torch.save({'state_dict': self.model.state_dict(),
                    'optimizer' : self.optimizer.state_dict(),
                   }, 'last_brain.pth')


    #model 과 optimizer load
    def load(self):
        if os.path.isfile('last_brain.pth'):
            print("=> loading checkpoint...")
            checkpoint = torch.load('last_brain.pth')
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("done!")
        else:
            print("no checkpoint found ....")





















