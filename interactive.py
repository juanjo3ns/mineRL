import gym
import minerl
from time import sleep, time

env = gym.make('MineRLTreechop-v0')

# set the environment to allow interactive connections on port 6666
# and slow the tick speed to 6666.
# env.make_interactive(port=6666, realtime=True)

# reset the env
env.reset()
ini = time()

while True:
    action = env.action_space.sample()
    obs, reward, done, _ = env.step(action)
    print(obs.keys())
    for k,v in obs.items():
        print(k, ' shape: ', v.shape)
    sleep(2)
    if time()-ini > 10:
        break

env.close()
