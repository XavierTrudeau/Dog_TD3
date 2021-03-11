import gym
from environment import env
import numpy as np
from td3_tf2 import Agent
from utils import plot_learning_curve



env = env()

agent = Agent(alpha=0.001, beta=0.001,
        input_dims=env.observation_space.shape, tau=0.005,
        env=env, batch_size=100, layer1_size=256, layer2_size=128, warmup=1000,
        n_actions=env.action_space.shape[0], noise=0.1)

n_games = 5000
show_every = 50
aggregate_stats_every = 10
filename = 'plots/' + 'walker_' + str(n_games) + '_games.png'

best_score = env.reward_range[0]
score_history = []

agent.load_models()

observation = env.reset()

done = False
score = 0
while not done:
    action = agent.run_model(observation)
    observation_, reward, done, info = env.step(action)
    observation = observation_
    score += reward

print("Simulation finished with a score of " + str(score) + "!")






