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
        # nn.Module을 사용하기 위한 트릭릭
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


































