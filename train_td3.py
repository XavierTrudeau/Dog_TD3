from environment import env
import numpy as np
from td3_tf2 import Agent
from utils import plot_learning_curve
from pynput import keyboard


def on_press(key):
    if key == keyboard.Key.esc:
        env.render(False,False)

    elif key == keyboard.Key.f5:
        env.render(True, True)

    elif key == keyboard.Key.f6:
        print("*** ACTION (Length = " + str(len(action)) + ") ***")
        print(np.array(action))
        print("*****************************************\n")
    elif key == keyboard.Key.f7:
        print("*** OBSERVATION (Length = " + str(len(observation)) + ") ***")
        print(observation)
        print("*****************************************\n")

    try:
        k = key.char  # single-char keys
    except:
        k = key.name  # other keys


#env = gym.make('LunarLanderContinuous-v2')
#env = gym.make('Pendulum-v0')
env = env()
#env = gym.make('BipedalWalker-v3')
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

#agent.load_models()

listener = keyboard.Listener(on_press=on_press)
listener.start()  # start to listen on a separate thread

print("*************** LEARNING PROCESS STARTING ***************")
print("*** CONTROLS :")
print("*** ESC: Disable Rendering")
print("*** F5: Enable Rendering")
print("*** F6: Print Current Action")
print("*** F7: Print Current Observation")
print("*********************************************************\n")


for ep in range(n_games):

    observation = env.reset()

    done = False
    score = 0
    while not done:
        action = agent.choose_action(observation)
        observation_, reward, done, info = env.step(action)
        agent.remember(observation, action, reward, observation_, done)
        agent.learn()
        score += reward
        observation = observation_

    score_history.append(score)
    if ep > 20:
        avg_score = np.mean(score_history[-100:])
    else:
        avg_score = env.reward_range[0]

    if avg_score > best_score:
        best_score = avg_score
        agent.save_models()

    print('episode ', ep, 'score %.1f' % score,
            'average score %.1f' % avg_score)

    if not ep % aggregate_stats_every:
        x = [i+1 for i in range(ep+1)]
        plot_learning_curve(x, score_history, filename)



