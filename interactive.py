import gym
import minerl
from time import sleep, time
from IPython import embed

env = gym.make('MineRLNavigate-v0')

# set the environment to allow interactive connections on port 6666
# and slow the tick speed to 6666.
env.make_interactive(port=6666, realtime=True)

# reset the env
env.seed(2)
env.reset()
ini = time()

while True:
    embed()
    action = env.action_space.sample()
    obs, reward, done, _ = env.step(action)

env.close()
