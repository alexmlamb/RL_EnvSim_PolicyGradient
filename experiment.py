import sys
import os
import argparse
import logging
import cPickle
import numpy as np
import experiment as exp

import gym


env = gym.make('Pendulum-v0')
env.reset()

for i_episode in range(10000):
    observ = env.reset()
    for t in range(100):
        env.render()
        print observ
        import ipdb; ipdb.set_trace()
        action = env.action_space.sample()
        #env.step(env.action_space.sample())
        observ, reward, done, info = env.step(action)
        if done:
            print 'episode finished after {} steps'.format(t+1)
            break



