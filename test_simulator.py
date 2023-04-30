import gym
import numpy as np
import torch
from collections import defaultdict
import pylab as plt
from ai_traineree.loggers.tensorboard_logger import TensorboardLogger
from ai_traineree.multi_agents.maddpg import MADDPGAgent
from ai_traineree.runners.multiagent_env_runner import MultiAgentCycleEnvRunner
from ai_traineree.tasks import PettingZooTask
from intersim.envs.multisim import MultiSim, wrap_env

tmp_path = "/home/methier/school/ece_750/InteractionSimulatorFork/test_outputs/render_"
# SCENARIO_NAME = "DR_USA_Intersection_MA"
# TRACK = 100
# CONTROLLED_AGENTS = [2, 4]
# BLACKBOX_AGENTS = [5, 6]
SCENARIO_NAME = "DR_CHN_Merging_ZS2"
TRACK = 181
CONTROLLED_AGENTS = [3, 4]
BLACKBOX_AGENTS = [5, 6]
RUN_NAME = SCENARIO_NAME + str(TRACK)

# env = gym.make('intersim:multisim-v0',
#     scenario_name=SCENARIO_NAME,
#     track=100,
#     controlled_agents=CONTROLLED_AGENTS,
#     blackbox_agents=BLACKBOX_AGENTS
#     )
#reward_coeffs = [0, 0, -1, 0]
reward_coeffs = [0, 0, -1, 0]
env = MultiSim(
    scenario_name=SCENARIO_NAME,
    track=TRACK,
    controlled_agents=CONTROLLED_AGENTS,
    blackbox_agents=BLACKBOX_AGENTS,
    reward_coeffs=reward_coeffs
)
env = wrap_env(env)
ma_task = PettingZooTask(env)
ma_task.reset()

obs_size = int(ma_task.obs_size[0])
action_size = int(ma_task.action_size.shape[0])
agent_number = ma_task.num_agents
config = {
    "device": "cuda",
    "update_freq": 20,
    "batch_size": 400,
    "agent_names": env.agents,
    "hidden_layers": (256, 256, 256),
    "actor_lr": 0.001,
    "critic_lr": 0.001
}
EXP_NAME = "v2sheepdogs-MADDPG-eps0.999-lr0.001-reward" + '-'.join([str(c) for c in reward_coeffs])
agent_name = env.possible_agents[0]
ma_agent = MADDPGAgent(ma_task.observation_spaces[agent_name], ma_task.action_spaces[agent_name], agent_number, **config)
data_logger = TensorboardLogger(log_dir="runs/"+EXP_NAME)
# data_logger = None
env_runner = MultiAgentCycleEnvRunner(ma_task, ma_agent, max_iterations=10000, data_logger=data_logger)
scores = env_runner.run(reward_goal=1000*ma_task.num_agents, max_episodes=10000, eps_decay=0.999, log_episode_freq=5, checkpoint_every=500, force_new=True, render_path=tmp_path+EXP_NAME, render_every_n=1000)
#ma_task.env.env.env.render_anim(f"{tmp_path}_{0}")
# scores = env_runner.run(reward_goal=1000*ma_task.num_agents, max_episodes=1000, eps_decay=0.9999, log_episode_freq=5, force_new=True, checkpoint_every=500)
# ma_task.env.env.env.render_anim(f"{tmp_path}_{1}")
# env_runner.save_state(RUN_NAME+str(1))
# for i in range(2, 11):
#     scores = env_runner.run(reward_goal=1000*ma_task.num_agents, max_episodes=i*1000, eps_decay=0.9999, log_episode_freq=5, force_new=False, checkpoint_every=500)
#     ma_task.env.env.env.render_anim(f"{tmp_path}_{i}")
#     env_runner.save_state(RUN_NAME+str(i))

######################################

# dog_names = ['Meg', 'Nell', 'Ben', 'Fly', 'Glen', 'Moss', 'Roy', 'Jess', 'Spot', 'CapFlash', 
#             'Jill', 'Bess', 'Sweep', 'Jim', 'Tess', 'Lassie', 'Sam', 'Shep', 'Bob', 'Queen', 
#             'Floss', 'Mirk', 'Lad', 'Mist', 'Bill', 'Don', 'Dot', 'Craig', 'Tweed', 'Nan', 
#             'Lass', 'Gyp', 'Jan', 'Nap', 'Mac', 'Bet', 'Kim', 'Jack', 'Joe', 'Jed', 'Maid']

# TOTAL_EPISODES = 10000
# EPISODES_BETWEEN_SAVES = 500
# INTERVALS = TOTAL_EPISODES / EPISODES_BETWEEN_SAVES

# env = gym.make('intersim:multisim-v0',
#     scenario_name="DR_USA_Intersection_MA",
#     track=100,
#     controlled_agents=[2, 4],
#     blackbox_agents=[5, 6]
#     )
# c_a = [2, 4]
# # Create agents
# dogs = []
# for i in range(len(c_a)):
#     #DDPG critic_model, actor_model, observation_space, action_space, experience='ReplayMemory-1000', exploration='OUNoise', lr_actor=0.01, lr_critic=0.01, gamma=0.95, batch_size=32, target_update_freq=None, name='DDPGAgent
#     #dogs.append(MADDPG("MlpNet", "MlpNet", env.observation_space, env.action_space, gamma=1, target_update_freq=100, name=dog_names[i]))
#     #MADDPG critic_model, actor_model, observation_space, action_space, index=None, experience='ReplayMemory-1000', exploration='OUNoise', lr_actor=0.01, lr_critic=0.01, gamma=0.95, batch_size=32, tau=0.01, use_target_net=100, name='MADDPGAgent'
#     maddpg = MADDPGAgent(
#         ContinuousCritic(1, 1, hidden_size=[64, 64]), 
#         MlpNet(1, 60, hidden_size=[64, 64]), 
#         env.observation_space, 
#         env.action_space, 
#         index=i,
#         gamma=1,
#         name=dog_names[i],
#         exploration='EpsGreedy'
#     )
#     dogs.append(maddpg)


# # Create the trainable multi-agent system
# mas = MARL(agents_list=dogs)

# # Assign MAS to each agent
# for i in range(len(c_a)):
#     dogs[i].set_mas(mas)

# for ep_num in range(TOTAL_EPISODES):
#     # Train the agent
#     mas.learn(env, nb_timesteps=EPISODES_BETWEEN_SAVES)
#     # Test the agents for 10 episodes
#     print(mas.test(env, nb_episodes=10, time_laps=0.1, max_num_step=1, render=False,)['mean_by_episode'])
#     if ep_num%EPISODES_BETWEEN_SAVES == 0:
#         save_policy(self, folder='models', filename='model', timestep=ep_num)