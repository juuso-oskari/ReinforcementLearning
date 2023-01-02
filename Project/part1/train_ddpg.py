import sys, os
sys.path.insert(0, os.path.abspath("../.."))
os.environ["MUJOCO_GL"] = "egl" # for mujoco rendering
#os.environ["MUJOCO_GL"]="glfw"
import time
from pathlib import Path



import torch
import gym
import hydra
import wandb
import warnings
warnings.filterwarnings("ignore", category=UserWarning) 
warnings.filterwarnings("ignore", category=DeprecationWarning) 
torch.autograd.set_detect_anomaly(True)

from make_env import *
from ddpg_self_made import DDPG
from common import helper as h
from common import logger as logger

def to_numpy(tensor):
    return tensor.cpu().numpy().flatten()

# Policy training function
def train(agent, env, max_episode_steps=1000):
    # Run actual training        
    reward_sum, timesteps, done, episode_timesteps = 0, 0, False, 0
    # Reset the environment and observe the initial state
    obs = env.reset()
    while not done:
        episode_timesteps += 1
        
        # Sample action from policy
        action, act_logprob = agent.get_action(obs)

        # Perform the action on the environment, get new state and reward
        next_obs, reward, done, _ = env.step(to_numpy(action))

        # Store action's outcome (so that the agent can improve its policy)
        if isinstance(agent, DDPG):
            # ignore the time truncated terminal signal
            done_bool = float(done) if episode_timesteps < max_episode_steps else 0 
            agent.record(obs, action, next_obs, reward, done_bool)
        else: raise ValueError

        # Store total episode reward
        reward_sum += reward
        timesteps += 1

        # update observation
        obs = next_obs.copy()

    # update the policy after one episode
    info = agent.update()

    # Return stats of training
    info.update({'timesteps': timesteps,
                'ep_reward': reward_sum,})
    return info


# Function to test a trained policy
@torch.no_grad()
def test(agent, env, num_episode=10):
    total_test_reward = 0
    for ep in range(num_episode):
        obs, done= env.reset(), False
        test_reward = 0

        while not done:
            # Similar to the training loop above -
            # get the action, act on the environment, save total reward
            # (evaluation=True makes the agent always return what it thinks to be
            # the best action - there is no exploration at this point)
            action, _ = agent.get_action(obs, evaluation=True)
            obs, reward, done, info = env.step(to_numpy(action))
            
            test_reward += reward

        total_test_reward += test_reward
        #print("Test ep_reward:", test_reward)

    print("Average test reward:", total_test_reward/num_episode)
    return total_test_reward/num_episode

env_name = "lunarlander_continuous_medium"
# The main function
@hydra.main(config_path='.', config_name=env_name)
def main(cfg):
    # sed seed
    h.set_seed(cfg.seed)
    cfg.run_id = int(time.time())

    # create folders if needed
    work_dir = Path().cwd()/'results'/f'{cfg.env_name}'
    if cfg.save_model: h.make_dir(work_dir/"model")
    if cfg.save_logging: 
        h.make_dir(work_dir/"logging")
        L = logger.Logger() # create a simple logger to record stats

    # Model filename
    if cfg.model_path == 'default':
        cfg.model_path = work_dir/'model'

    # use wandb to store stats; we aren't currently logging anything into wandb during testing
    if cfg.use_wandb and not cfg.testing:
        wandb.init(project="rl_aalto",
                    name=f'{cfg.exp_name}-{cfg.env_name}-{str(cfg.seed)}-{str(cfg.run_id)}',
                    group=f'{cfg.exp_name}-{cfg.env_name}',
                    config=cfg)

    # create a env
    env = create_env(config_file_name=env_name, seed=cfg.seed)
    if cfg.save_video:
        # During testing, save every episode
        if cfg.testing:
            ep_trigger = 1
            video_path = work_dir/'video'/'test'
        # During training, save every 50th episode
        else:
            ep_trigger = 50
            video_path = work_dir/'video'/'train'
        env = gym.wrappers.RecordVideo(env, video_path,
                                        episode_trigger=lambda x: x % ep_trigger == 0,
                                        name_prefix=cfg.exp_name) # save video every 50 episode

    state_shape = env.observation_space.shape
    action_dim = env.action_space.shape[0]
    max_action = env.action_space.high[0]
    
    ## Automated HYPER PARAMETER OPTIMIZATION with grid search
    do_hyperparameter_optimization = False
    if do_hyperparameter_optimization:
        network_architectures = [(400,300), (100,100)]
        ac_lrs = [1e-4*0.8, 1e-4*0.9, 1e-4, 1e-4*1.1, 1e-4*1.2]
        cr_lrs = [1e-3*0.8, 1e-3*0.9, 1e-3, 1e-3*1.1, 1e-3*1.2]
        buffer_sizes = [1e6, 2e6]
        batch_sizes = [256]
        
        best_mean_reward = 0
        
        best_settings = []
        
        for na in network_architectures:
            for ac_lr in ac_lrs:
                for cr_lr in cr_lrs:
                    for buffer_sz in buffer_sizes:
                        for batch_size in batch_sizes:
                            print("Testing with settings", [na, ac_lr, cr_lr, buffer_sz, batch_size])
                            # create agent with currently investigated settings
                            agent = DDPG(state_shape, action_dim, max_action,
                                                ac_lr, cr_lr, cfg.gamma, cfg.tau,
                                                batch_size=int(batch_size), buffer_size=int(buffer_sz), network_architecture=na, use_ou=True)
                            iter_count = 0
                            mean_ep_reward = 0
                            for ep in range(1000+1):
                                train_info = train(agent, env)
                                #print(train_info['ep_reward'])
                                if (not cfg.silent) and (ep > 995):
                                    print({"ep": ep, **train_info})
                                    mean_ep_reward += train_info['ep_reward']
                                    iter_count += 1
                                
                            mean_ep_reward /= iter_count
                            print("mean 5 episode reward for, **train settings: na: ", na, " ac_lr: ", ac_lr, " cr_lr: ", cr_lr, " buffer_sz: ", buffer_sz, " batch_size: ", batch_size)
                            print(mean_ep_reward)
                            if best_mean_reward < mean_ep_reward:
                                best_mean_reward = mean_ep_reward
                                best_settings = [na, ac_lr, cr_lr, buffer_sz, batch_size]
                            
                            print("Best settings thus far: ", best_settings)
    
    
    # Manual hyper parameter optimization, take inspiration from TD3
    
    # suggestion based on TD3 tricks
    # Trick Two: “Delayed” Policy Updates. TD3 updates the policy (and target networks) less frequently than the Q-function.
    # The paper recommends one policy update for every two Q-function updates -> critic lr = 2 * actor lr
    
    # Trick Three: Target Policy Smoothing. TD3 adds noise to the target action,
    # to make it harder for the policy to exploit Q-function errors by smoothing out Q along changes in action. -> use_ou=True, adds noise (0 mean, 0.1 std) to action output
    manual_hp_optimization = False        
    if(manual_hp_optimization):
    
        best_settings = {"network_architecture" : (400, 300), "actor_learning_rate" : 0.5 * 0.0003, "critic_learning_rate" : 0.0003, "buffer_size" : 2e6, "batch_size" : 256,
                        "gamma" : 0.99, "tau" : 0.005, "use_output_action_noise" : True}
        
        print(best_settings)
        agent = DDPG(state_shape, action_dim, max_action,
                    actor_lr=best_settings["actor_learning_rate"], critic_lr=best_settings["critic_learning_rate"], gamma=best_settings["gamma"], tau=best_settings["tau"],
                    batch_size=best_settings["batch_size"], buffer_size=best_settings["buffer_size"], network_architecture=best_settings["network_architecture"], use_ou=best_settings["use_output_action_noise"], ou_std = 0.2)
        
    use_default_from_ex_6 = True
    
    if(use_default_from_ex_6):
        #agent = DDPG(state_shape, action_dim, max_action, cfg.lr, cfg.gamma, cfg.tau, batch_size=int(cfg.batch_size), buffer_size=int(cfg.buffer_size))
        agent = DDPG(state_shape, action_dim, max_action, cfg.lr, cfg.gamma, cfg.tau, batch_size=int(cfg.batch_size), buffer_size=2*int(cfg.buffer_size), action_noise=0.20)
    

    running_mean = 0
    mean_iter = 0

    if not cfg.testing: # training
        for ep in range(cfg.train_episodes + 1):
            # collect data and update the policy
            train_info = train(agent, env)
            if cfg.use_wandb:
                wandb.log(train_info)
            if cfg.save_logging:
                L.log(**train_info)
            if ( (ep+1) % 100 == 0):
                print("Checkpoint save: ")
                agent.save(cfg.model_path)
                print({"ep": ep, **train_info})
                mean_50 = test(agent, env, num_episode=50)
                print("Mean test episode reward for 50 episodes: ", mean_50)
                if(mean_50 > 100):
                    print("Early stopping condition fulfilled")
                    break
                # zero counters
                running_mean = 0
                mean_iter = 0
                
            running_mean += train_info["ep_reward"]
            mean_iter += 1

        
        if cfg.save_model:
            agent.save(cfg.model_path)

    else: # testing
        model_path = work_dir/'model'

        # load model
        agent.load(model_path)
        
        print('Testing ...')
        test(agent, env, num_episode=50)


# Entry point of the script
if __name__ == "__main__":
    main()


