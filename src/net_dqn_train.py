import os
import time
import torch
import random
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
from collections import deque
from models import NeuralNetwork
from breakout_game import Breakout
from utils import device, init_weights, preprocessing


def train(model, start, lr=0.0003, optim_name='adam', loss_name='mse', num_iter_save=1000):
    # define optimizer
    if optim_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optim_name == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr)

    # initialize error loss
    if loss_name == 'mse':
        criterion = nn.MSELoss() 
    elif loss_name == 'crossentropy':
        criterion = nn.CrossEntropyLoss() 

    # instantiate game
    game_state = Breakout()

    # initialize replay memory
    replay_buffer = deque()

    # initial action
    action = torch.zeros([model.number_of_actions], dtype=torch.float32).to(device)
    image_data, reward, terminal = game_state.take_action(action)
    image_data = preprocessing(image_data)
    state = torch.cat((image_data, image_data, image_data, image_data)).unsqueeze(0) 

    # initialize epsilon value
    epsilon = model.initial_epsilon
    iteration = 0

    # main infinite loop
    while iteration < model.number_of_iterations:
        # get output from the neural network
        output = model(state)[0] # Output size = torch.Size([2]) tensor([-0.0278,  1.7244]
        #output = model(state)

        # initialize action
        action = torch.zeros([model.number_of_actions], dtype=torch.float32)

        # epsilon greedy exploration
        random_action = random.random() <= epsilon

        # Pick action --> random or index of maximum q value
        action_index = [torch.randint(model.number_of_actions, torch.Size([]), dtype=torch.int)
                        if random_action
                        else torch.argmax(output)][0].to(device)

        action[action_index] = 1

        if epsilon > model.final_epsilon:
            epsilon -= (model.initial_epsilon - model.final_epsilon) / model.explore

        # get next state and reward
        image_data_1, reward, terminal = game_state.take_action(action)
        image_data_1 = preprocessing(image_data_1)
        
        state_1 = torch.cat((state.squeeze(0)[1:, :, :], image_data_1)).unsqueeze(0)   
        action = action.unsqueeze(0)
        reward = torch.from_numpy(np.array([reward], dtype=np.float32)).unsqueeze(0).to(device)   
        
        # save transition to replay memory
        replay_buffer.append((state, action, reward, state_1, terminal))

        # if replay memory is full, remove the oldest transition
        if len(replay_buffer) > model.replay_memory_size:
            replay_buffer.popleft()

        # sample random mini_batch
        # it picks k unique random elements, a sample, from a sequence: random.sample(population, k)
        mini_batch = random.sample(replay_buffer, min(len(replay_buffer), model.minibatch_size))

        # unpack mini_batch
        state_batch   = torch.cat(tuple(d[0] for d in mini_batch)).to(device)
        action_batch  = torch.cat(tuple(d[1] for d in mini_batch)).to(device)
        reward_batch  = torch.cat(tuple(d[2] for d in mini_batch)).to(device)
        state_1_batch = torch.cat(tuple(d[3] for d in mini_batch)).to(device)
        
        # get output for the next state
        output_1_batch = model(state_1_batch)

        # set y_j to r_j for terminal state, otherwise to r_j + gamma*max(Q) Target Q value Bellman equation.
        y_batch = torch.cat(tuple(reward_batch[i] if mini_batch[i][4]
                                  else reward_batch[i] + model.gamma * torch.max(output_1_batch[i])
                                  for i in range(len(mini_batch))))

        q_value = torch.sum(model(state_batch) * action_batch, dim=1)

        optimizer.zero_grad()
        y_batch = y_batch.detach()

        # calculate loss
        loss = criterion(q_value, y_batch)

        # do backward pass
        loss.backward()
        optimizer.step()

        # set state to be state_1
        state = state_1
        iteration += 1

        if iteration % num_iter_save == 0:
            print('Saving the model')
            torch.save(model, f"model_weights/model_{iteration}.pth")
        
        elapsed_time = time.time() - start
        elapsed_minutes = int(elapsed_time // 60)
        elapsed_seconds = int(elapsed_time % 60)

        print(f"Iteration: {iteration} Elapsed time: {elapsed_minutes}:{elapsed_seconds} epsilon: {epsilon:.5f} action: {action_index.cpu().item()} Reward: {reward.cpu().numpy()[0][0]:.1f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser() 

    parser.add_argument('--lr', type=float, default=0.0003, help="learning rate")
    parser.add_argument('--optim', type=str, default='adam', help="optimzer: adam, or sgd")
    parser.add_argument('--loss', type=str, default='mse', help="loss function: mse, or cross_entropy")
    parser.add_argument('--num_iter_save', type=int, default=1000, help="number of iterations to save the model checkpoint")

    args = parser.parse_args()

    os.makedirs('model_weights/', exist_ok=True)
    model = NeuralNetwork()
    if os.path.exists('model_weights/best.pth'):
        model = torch.load('model_weights/best.pth')
    else:
        model.apply(init_weights)
    
    model = model.train().to(device)
    
    start = time.time()
    train(model, start, args.lr, args.optim, args.loss, args.num_iter_save)
