import gym
import numpy as np
from matplotlib import pyplot as plt
from naf_agent import *
import torch
import tqdm
import time
import hydra
from pathlib import Path
import wandb
import warnings
warnings.filterwarnings("ignore", category=UserWarning) 
warnings.filterwarnings("ignore", category=DeprecationWarning) 

sys.path.insert(0, os.path.abspath("../.."))
from common import helper as h
from common import logger as logger
from common.buffer import ReplayBuffer
from NAF.make_env import *

def to_numpy(tensor):
    return np.float32(tensor.squeeze(0).cpu().numpy())

env_name = "hopper_medium"
@hydra.main(config_path='.', config_name=env_name)
def main(cfg):
    # set random seed
    h.set_seed(cfg.seed)
    cfg.run_id = int(time.time())
    # create folders if needed
    work_dir = Path().cwd()/'results'/cfg.env_name
    if cfg.save_logging:
        logging_path = work_dir / 'logging'
        h.make_dir(logging_path)
        L = logger.Logger() # create a simple logger to record stats
    if cfg.save_model:
        model_path = work_dir / 'model'
        h.make_dir(model_path)
    
    # use wandb to store stats
    if cfg.use_wandb:
        wandb.init(project="rl_aalto",
                    name=f'{cfg.exp_name}-{cfg.agent_name}-{cfg.env_name}-{str(cfg.seed)}-{cfg.run_id}',
                    group=f'{cfg.exp_name}-{cfg.env_name}',
                    config=cfg)
    
    # create env
    env = create_env(config_file_name=env_name, seed=cfg.seed)
    env.seed(cfg.seed)
    # get action and state dimensions
    action_dims = env.action_space.shape[0]
    state_shape = env.observation_space.shape
    # init agent
    agent = NAFAgent(state_shape, action_dims, batch_size=cfg.batch_size, hidden_dims=cfg.hidden_dims,
                         gamma=cfg.gamma, lr=cfg.lr, tau=cfg.tau)
    agent.policy_net.train(False)
    agent.load(model_path)
    
    total_test_reward = 0
    num_episode = 50
    
    for ep in range(num_episode):
        state, done, test_reward, env_step = env.reset(), False, 0, 0
        #eps = max(cfg.glie_b/(cfg.glie_b + ep), 0.05)
        # collecting data and fed into replay buffer
        while not done:
            action = agent.get_action_wo_noise(state)
            # bring to cpu in numpy format             
            if isinstance(action, np.ndarray): action = np.float32(action.item())
            if not isinstance(action, np.ndarray): action = to_numpy(action)
            # simulate
            next_state, reward, done, _ = env.step(action)
            test_reward += reward
            # Move to the next state
            state = next_state

        info = {'episode': ep, 'test_reward': test_reward}
        total_test_reward += test_reward
        if cfg.use_wandb: wandb.log(info)
        if cfg.save_logging: L.log(**info)
        print(info)
    print("Average test reward:", total_test_reward/num_episode)
    

if __name__ == '__main__':
    main()