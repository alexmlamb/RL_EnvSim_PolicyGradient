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

action_size = 1
state_size = 1
reward_size = 1
nfp = 512
nfe = 512
num_steps_simulated_chain = 20
mb = 64

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

def net_simulated_chain(params_envsim, params_policy, initial_state, num_steps):


    initial_reward = theano.shared(np.zeros(shape=(mb,1)).astype('float32'))

    def one_step(last_state,last_reward):

        next_action = policy_network(params_policy, initial_state)
        next_state, next_reward = envsim_network(params_envsim, last_state, next_action)
        
        return next_state, next_reward

    [state_lst, reward_lst], _ = theano.scan(fn=one_step,outputs_info=[initial_state, initial_reward],n_steps=num_steps)

    return reward_lst

params_policy = init_params_policy({})
params_envsim = init_params_envsim({})

initial_state_sim = T.matrix()

simulated_reward_lst = net_simulated_chain(params_envsim, params_policy, initial_state_sim,num_steps_simulated_chain)

simulated_total_reward = T.sum(simulated_reward_lst)
simulation_loss = -1.0 * simulated_total_reward

simulation_updates = lasagne.updates.adam(simulation_loss, params_policy.values())

simulation_function = theano.function(inputs=[initial_state_sim],outputs=[simulated_total_reward],updates=simulation_updates)

state = T.matrix()
action = T.matrix()

next_action = policy_network(params_policy, state)

next_state, reward = envsim_network(params_envsim, state, action)

for iteration in range(0,50000): 

    initial_state_sim = rng.normal(size = (64, state_size)).astype('float32')

    simulated_reward = simulation_function(initial_state_sim)

    if iteration % 500 == 1:
        print iteration, simulated_reward


