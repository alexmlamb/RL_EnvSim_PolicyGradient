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
    env.render()
    env.step(env.action_space.sample())



