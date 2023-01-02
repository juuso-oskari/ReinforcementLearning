import gym
import numpy as np
from matplotlib import pyplot as plt
from naf_agent import *
import torch
torch.set_default_dtype(torch.float32)
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
    return np.float32(tensor.squeeze().cpu().numpy())


def test(env, agent):
    total_test_reward = 0
    num_episode = 50
    
    for ep in range(num_episode):
        state, done, test_reward, env_step = env.reset(), False, 0, 0
        #eps = max(cfg.glie_b/(cfg.glie_b + ep), 0.05)
        # collecting data and fed into replay buffer
        while not done:
            action = agent.get_action_wo_noise(state)
            # bring to cpu in numpy format             
            action = to_numpy(action)
            # simulate
            next_state, reward, done, _ = env.step(action)
            test_reward += reward
            # Move to the next state
            state = next_state

        total_test_reward += test_reward
    return total_test_reward/num_episode

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
    #env = gym.make(cfg.env_name, render_mode='rgb_array' if cfg.save_video else None, max_episode_steps=cfg.max_episode_steps)
    env = create_env(config_file_name=env_name, seed=cfg.seed)
    env.seed(cfg.seed)
    if cfg.save_video:
        env = gym.wrappers.RecordVideo(env, work_dir/'video'/'train',
                                        episode_trigger=lambda x: x % 100 == 0,
                                        name_prefix=cfg.exp_name) # save video for every 100 episodes
    # get action and state dimensions
    action_dims = env.action_space.shape[0]
    state_shape = env.observation_space.shape
    # init agent
    agent = NAFAgent(state_shape, action_dims, batch_size=cfg.batch_size, hidden_dims=cfg.hidden_dims,
                         gamma=cfg.gamma, lr=cfg.lr, tau=cfg.tau, use_shallow=True)
    agent.policy_net.train(False)
    #  init buffer
    buffer = ReplayBuffer(state_shape, action_dim=action_dims, max_size=int(cfg.buffer_size))
    for ep in range(cfg.train_episodes):
        state, done, ep_reward, env_step = env.reset(), False, 0, 0
        #eps = max(cfg.glie_b/(cfg.glie_b + ep), 0.05)
        # collecting data and fed into replay buffer
        while not done:
            env_step += 1
            if ep < cfg.random_episodes: # in the first #random_episodes, collect random trajectories
                action = env.action_space.sample()
            else: # Select and perform an action according to (noisy) policy
                action = agent.get_action(state)
                # bring to cpu in numpy.float32 format             
                action = to_numpy(action)
            # simulate
            next_state, reward, done, _ = env.step(action)
            ep_reward += reward
            # Store the transition in replay buffer
            buffer.add(state, action, next_state, reward, done)
            # Move to the next state
            state = next_state
            # with batch input, set the batch normalization layers to learning mode
            agent.policy_net.train(True)
            # Perform one update_per_episode step of the optimization
            if ep >= cfg.random_episodes:
                update_info = agent.update(buffer)
            else: 
                update_info = {}
                
            agent.policy_net.train(False)

        info = {'episode': ep, 'train_reward': ep_reward}
        #if ep >= cfg.random_episodes:
        #    info.update(update_info)

        if cfg.use_wandb: wandb.log(info)
        if cfg.save_logging: L.log(**info)
        if (not cfg.silent) and (ep % 100 == 0):
            print(info)
            
    # save model and logging    
    if cfg.save_model:
        agent.save(model_path)
    if cfg.save_logging:
        L.save(logging_path/'logging.pkl')
    
    print('------ Training Finished ------')


if __name__ == '__main__':
    main()