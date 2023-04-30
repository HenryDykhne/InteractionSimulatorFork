import gym
import numpy as np
import torch
import pandas as pd
import os

from intersim.default_policies.idm import IDM, IDM2
from intersim.default_policies.rulepolicy import RulePolicy
from intersim.envs.multisim import MultiSim, wrap_env
import intersim


tmp_path = "/home/methier/school/ece_750/InteractionSimulatorFork/test_outputs/render"
#SCENARIO_NAME = "DR_USA_Intersection_EP0"

SCENARIO_NAME = "DR_CHN_Merging_ZS2"
TRACK = 181
CONTROLLED_AGENTS = [3, 4]
BLACKBOX_AGENTS = [5, 6]

# SCENARIO_NAME = "DR_CHN_Merging_ZS2"
# TRACK = 499
# CONTROLLED_AGENTS = [3, 4, 5]
# BLACKBOX_AGENTS = [6, 7, 8]

# SCENARIO_NAME = "DR_CHN_Merging_ZS2"
# TRACK = 700
# CONTROLLED_AGENTS = list(range(2, 12))
# BLACKBOX_AGENTS = list(range(13, 31))

# SCENARIO_NAME = "DR_CHN_Merging_ZS0"
# TRACK = 200
# CONTROLLED_AGENTS = [2, 4, 5, 6, 7, 8, 9]
# BLACKBOX_AGENTS = [10, 11, 12, 13, 14, 15]

# SCENARIO_NAME = "DR_DEU_Merging_MT"
# TRACK = 200
# CONTROLLED_AGENTS = [3, 4, 5]
# BLACKBOX_AGENTS = [6, 7]
DATASET_BASE = os.path.normpath(os.path.join(os.path.dirname(intersim.__file__), '..'))
# path = os.path.join(DATASET_BASE, 'datasets', 'trackfiles', SCENARIO_NAME, 'vehicle_tracks_%04i.csv'%(TRACK))
# df = pd.read_csv(path)
# tid_max = df.track_id.max()
# valid_tracks = []
# for i in range(1, tid_max+1):
#     df_tid = df[df.track_id == i]
#     if df_tid.frame_id.min() == 1 and df_tid.frame_id.max() == 40:
#         valid_tracks.append(i)
# print(valid_tracks)

# CONTROLLED_AGENTS = valid_tracks[:-20]
# BLACKBOX_AGENTS = valid_tracks[-20:-10]

# Find all valid tracks
# for track in range(2, 1043):
#     path = os.path.join(DATASET_BASE, 'datasets', 'trackfiles', SCENARIO_NAME, 'vehicle_tracks_%04i.csv'%(track))
#     df = pd.read_csv(path)

#     tid_max = df.track_id.max()
    
#     valid_tracks = []
#     for i in range(1, tid_max+1):
#         df_tid = df[df.track_id == i]
#         if df_tid.frame_id.min() == 1 and df_tid.frame_id.max() == 40:
#             valid_tracks.append(i)
#             #print(f"tid: {i}, fidmin: {df_tid.frame_id.min()}, fidmax: {df_tid.frame_id.max()}")
#     if len(valid_tracks) >= 4:
#         print(track)
#         print(valid_tracks)

env = MultiSim(
    scenario_name=SCENARIO_NAME,
    track=TRACK,
    controlled_agents=CONTROLLED_AGENTS,
    blackbox_agents=BLACKBOX_AGENTS,
    blackbox_policy=IDM,
)
_ = env.reset()
print(env.agents)

# Set action to drive in a straight line at constant velocity
#action = torch.zeros((30, 2)) # action: (num_controlled_agents, traj_length, 2)
actions = {agent: torch.zeros((30, 2)) for agent in env.agents}
ob, reward, done, info = env.step(actions)
env.render_anim(filestr=str(tmp_path))
