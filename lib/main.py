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

from collections import OrderedDict
import gym
import logging
import cPickle
import numpy as np
import argparse

action_size = 1
state_size = 1
reward_size = 1
nfp = 512
nfe = 512

#state -> action
def init_params_policy(p):

    p = param_init_fflayer(options={},params=p,prefix='pn_1',nin=state_size,nout=nfp,ortho=False,batch_norm=True)

    p = param_init_fflayer(options={},params=p,prefix='pn_2',nin=nfp,nout=nfp,ortho=False,batch_norm=True)
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

    h1 = fflayer(tparams=p,state_below=inp,options={},prefix='pn_1',activ='lambda x: tensor.nnet.relu(x,alpha=0.02)',batch_norm=True)


    h2 = fflayer(tparams=p,state_below=h1,options={},prefix='pn_2',activ='lambda x: tensor.nnet.relu(x,alpha=0.02)',batch_norm=True)


    action = fflayer(tparams=p,state_below=h2,options={},prefix='pn_3',activ='lambda x: x',batch_norm=False)

    return action

def envsim_network(p,state,action):

    inp = join2(state,action)

    h1 = fflayer(tparams=p,state_below=inp,options={},prefix='es_1',activ='lambda x: tensor.nnet.relu(x,alpha=0.02)',batch_norm=True)

    h2 = fflayer(tparams=p,state_below=h1,options={},prefix='es_2',activ='lambda x: tensor.nnet.relu(x,alpha=0.02)',batch_norm=True)

    next_state = fflayer(tparams=p,state_below=h2,options={},prefix='es_state',activ='lambda x: x',batch_norm=False)

    reward = fflayer(tparams=p,state_below=h2,options={},prefix='es_reward',activ='lambda x: x',batch_norm=False)

    return next_state, reward

def real_chain(params_policy, initial_state):

    action_lst = []
    state_lst = []
    reward_lst = []

    return action_lst, state_lst, reward_lst

def simulated_chain(params_envsim, params_policy, initial_state):


    reward_lst = []
    return reward_lst

params_policy = OrderedDict()
params_envsim  = OrderedDict()

params_policy = init_params_policy({params_policy})
params_envsim = init_params_envsim({params_envsim})

tstate = T.matrix()
taction = T.matrix()


next_action = policy_network(params_policy, state)

next_state, reward = envsim_network(params_envsim, state, action)

env = gym.make('Pendulum-v0')
state_value = env.reset()

compute_action = theano.function([state, tparams_policy],[next_action])


for i_episode in range(100):
    # sample action from the policy network
    # pass the action to the simulator network
    action_ = compute_action(state_value, params_policy)
