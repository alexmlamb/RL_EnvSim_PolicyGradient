'''
main code for running
'''

import sys

sys.path.append("/u/lambalex/DeepLearning/rl_hw_5")
sys.path.append("/u/lambalex/DeepLearning/rl_hw_5/lib")

import theano
import theano.tensor as T
from nn_layers import fflayer, param_init_fflayer
from utils import init_tparams, join2, srng, dropout, inverse_sigmoid
import lasagne
import numpy.random as rng
import numpy as np

from collections import OrderedDict
import gym
import logging
import cPickle
import numpy as np
import argparse

action_size = 1
state_size = 3
reward_size = 1
nfp = 512
nfe = 512
num_steps_simulated_chain = 20
mb = 64

#state -> action
def init_params_policy(p):

    p = param_init_fflayer(options={},params=p,prefix='pn_1',nin=state_size,nout=nfp,ortho=False,batch_norm=False)

    p = param_init_fflayer(options={},params=p,prefix='pn_2',nin=nfp,nout=nfp,ortho=False,batch_norm=False)
    p = param_init_fflayer(options={},params=p,prefix='pn_3',nin=nfp,nout=action_size,ortho=False,batch_norm=False)

    return init_tparams(p)

#state, action -> next_state, reward
def init_params_envsim(p):

    p = param_init_fflayer(options={},params=p,prefix='es_1',nin=action_size+state_size,nout=nfe,ortho=False,batch_norm=True)
    p = param_init_fflayer(options={},params=p,prefix='es_2',nin=nfe,nout=nfe,ortho=False,batch_norm=True)
    p = param_init_fflayer(options={},params=p,prefix='es_state',nin=nfe,nout=state_size,ortho=False,batch_norm=False)

    p = param_init_fflayer(options={},params=p,prefix='es_reward',nin=nfe,nout=reward_size,ortho=False,batch_norm=False)

    return init_tparams(p)

def policy_network(p,state):

    inp = state

    h1 = fflayer(tparams=p,state_below=inp,options={},prefix='pn_1',activ='lambda x: tensor.nnet.relu(x,alpha=0.02)',batch_norm=False)

    h2 = fflayer(tparams=p,state_below=h1,options={},prefix='pn_2',activ='lambda x: tensor.nnet.relu(x,alpha=0.02)',batch_norm=False)

    action = fflayer(tparams=p,state_below=h2,options={},prefix='pn_3',activ='lambda x: x',batch_norm=False)

    return action

def envsim_network(p,state,action):

    inp = join2(state,action)

    h1 = fflayer(tparams=p,state_below=inp,options={},prefix='es_1',activ='lambda x: tensor.nnet.relu(x,alpha=0.02)',batch_norm=False)

    h2 = fflayer(tparams=p,state_below=h1,options={},prefix='es_2',activ='lambda x: tensor.nnet.relu(x,alpha=0.02)',batch_norm=False)

    next_state = fflayer(tparams=p,state_below=h2,options={},prefix='es_state',activ='lambda x: x',batch_norm=False)

    reward = fflayer(tparams=p,state_below=h2,options={},prefix='es_reward',activ='lambda x: x',batch_norm=False)

    return next_state, reward


def net_simulated_chain(params_envsim, params_policy, initial_state, num_steps):


    initial_reward = theano.shared(np.zeros(shape=(mb,1)).astype('float32'))

    def one_step(last_state,last_reward):

        next_action = policy_network(params_policy, initial_state)
        next_state, next_reward = envsim_network(params_envsim, last_state, next_action)
        
        return next_state, next_reward

    [state_lst, reward_lst], _ = theano.scan(fn=one_step,outputs_info=[initial_state, initial_reward],n_steps=num_steps)

    return reward_lst

params_policy = init_params_policy({})
params_envsim  = init_params_envsim({})

env = gym.make('Pendulum-v0')
initial_state = env.reset().astype(np.float32)

state = T.matrix()
action = policy_network(params_policy, state)
#action = theano.function([state, tparams_policy],[next_action])
compute_action = theano.function([state], outputs = action)

def real_chain(initial_state, num_steps):

    # sample action from the policy network
    # pass the action to the simulator network

    action_lst = []
    state_lst = [initial_state.reshape((1, state_size))]
    reward_lst = []

    for i in range(num_steps):
        action = compute_action(state_lst[-1])
        # env.render()
        action = action.reshape((action_size))
        state, reward, done, info = env.step(action)
        state = state.reshape((1,state_size)).astype('float32')
        reward = reward.reshape((1,reward_size)).astype('float32')
        action_lst.append(action)
        state_lst.append(state)
        reward_lst.append(reward)

    return action_lst, state_lst, reward_lst


#<<<<<<< HEAD

#tstate = T.matrix()
#taction = T.matrix()


#next_action = policy_network(params_policy, state)

#next_state, reward = envsim_network(params_envsim, state, action)


#=======


#print 'action list ', action_lst
#print 'state_list, ', state_lst
#print 'rewrd_list, ', reward_lst



initial_state_sim = T.matrix()

simulated_reward_lst = net_simulated_chain(params_envsim, params_policy, initial_state_sim,num_steps_simulated_chain)

simulated_total_reward = T.sum(simulated_reward_lst)
simulation_loss = -1.0 * simulated_total_reward


simulation_updates = lasagne.updates.adam(simulation_loss, params_policy.values())

simulation_function = theano.function(inputs=[initial_state_sim],outputs=[simulated_total_reward],updates=simulation_updates)

#next_state_real, rewards_real = envsim_network(params_envism, state, action)

#next_state_truth = T.vector()
#rewards_truth = T.scalar()

#real_loss = (next_state_truth - next_state_real) ** 2 + (rewards_truth - rewards_real) ** 2

#real_updates = lasagne.updates.adam(real_loss, params_envsim.values())

#compute_rewards = theano.function(inputs=[state, action], outputs=[rewards, next_state])

#num_steps = 20
#state = intial_state
#loss_lst = []

# compute loss for envoriment simulator for the real chain
#for i in range(num_steps):
#    reward, state = compute_rewards(state,action_list[0])
#    loss = (state_list[i]- state) ** 2 + (reward_list[i] - reward) ** 2
#    loss_lst.append(loss)
    

state = T.matrix()
action = T.matrix()


########################################################################
#Build method for training the environment simulator
########################################################################
last_state_envtr = T.matrix()
action_envtr = T.matrix()
next_state_envtr = T.matrix()
reward_envtr = T.matrix()

next_state_pred, next_reward_pred = envsim_network(params_envsim, last_state_envtr, action_envtr)

envtr_loss = T.mean((next_state_pred - next_state_envtr)**2) + T.mean((next_reward_pred - reward_envtr)**2)

envtr_updates = lasagne.updates.adam(envtr_loss, params_envsim.values())

train_envsim = theano.function(inputs = [last_state_envtr,action_envtr,next_state_envtr,reward_envtr], outputs = [envtr_loss], updates = envtr_updates)


for iteration in range(0,50000): 
    loss_list = []
    
    action_set = []
    state_set = []
    reward_set = []
    for i in range(mb):
        action_list, state_list, reward_list = real_chain(initial_state, 20)
        action_set.append(action_list)
        state_set.append(state_list)
        reward_set.append(reward_list)

    action_set = np.array(action_set).transpose(1,0,2).reshape((20,64,action_size))
    state_set = np.array(state_set).transpose(1,0,2,3).reshape((21,64,state_size))
    reward_set = np.array(reward_set).transpose(1,0,2,3).reshape((20,64,reward_size))



    #print 'action_list ', action_list
    #print 'state_list ', state_list
    #print 'reward_list ', reward_list
    for i in range(20):
        
        loss = train_envsim(state_set[i], action_set[i], state_set[i+1], reward_set[i])
        loss_list.append(loss)

    if iteration % 100 == 1:
        print 'loss_list ', np.array(loss_list).mean()

    initial_state_sim = rng.uniform(low=-2,high=2,size=(64,state_size)).astype('float32')

    simulated_reward = simulation_function(initial_state_sim)

    if iteration % 500 == 1:
        print iteration, simulated_reward





