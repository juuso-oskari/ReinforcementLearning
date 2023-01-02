import sys, os
sys.path.insert(0, os.path.abspath(".."))

import gym
import numpy as np

from sklearn.linear_model import SGDRegressor
from sklearn.kernel_approximation import RBFSampler
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import FeatureUnion

from common.buffer import Batch
from common import helper as h

class RBFAgent(object):
    def __init__(self, num_actions, gamma=0.98, batch_size=32):
        self.scaler = None
        self.featurizer = None
        self.q_functions = None
        self.gamma = gamma
        self.batch_size = batch_size
        self.num_actions = num_actions
        self._initialize_model()

    def _initialize_model(self):
        # Draw some samples from the observation range and initialize the scaler (used to normalize data)
        obs_limit = np.array([4.8, 5, 0.5, 5])
        samples = np.random.uniform(-obs_limit, obs_limit, (1000, obs_limit.shape[0]))
        
        # calculate the mean and var of samples, used later to normalize training data
        self.scaler = StandardScaler().fit(samples) 

        # Initialize the RBF featurizer
        self.featurizer = FeatureUnion([
            ("rbf1", RBFSampler(gamma=5.0, n_components=100)),
            ("rbf2", RBFSampler(gamma=2.0, n_components=80)),
            ("rbf3", RBFSampler(gamma=1.0, n_components=50)),
        ])
        self.featurizer.fit(self.scaler.transform(samples))

        # Create a value approximator for each action dimension
        self.q_functions = [SGDRegressor(learning_rate="constant", max_iter=500, tol=1e-3)
                       for _ in range(self.num_actions)]

        # Initialize it to whatever values; implementation detail
        for q_a in self.q_functions:
            q_a.partial_fit(self.featurize(samples), np.zeros((samples.shape[0],)))

    def featurize(self, state):
        """ Map state to a higher dimension for better representation."""
        if len(state.shape) == 1:
            state = state.reshape(1, -1)
        
        # TODO: comment this and comment out the RBF features in Task 1b
        # Manual features, Task 1a 
        
        state = np.concatenate((state, np.abs(state)), axis=1)
        return state

        # RBF features, Task 1b 
        # map a state to a higher dimension (100+80+50)
        # Hint: use self.featurizer.transform and self.scaler.transform
        #return None
        return self.featurizer.transform(self.scaler.transform(state))

    def get_action(self, state, epsilon=0.0):
        # TODO: Task 1: Implement epsilon-greedy policy
        ########## Your code starts here ##########
        # Hints:
        # 1. self.q_functions is a list which defines a q function for each action dimension
        # 2. for each q function, use predict(feature) to obtain the q value
        if len(state.shape) == 1:
            state = state.reshape(1, -1)
        #print("shape at get action")
        #print(np.shape(state))
        u = np.random.random()
        if u < epsilon:
            a = np.random.randint(low=0, high=self.num_actions)
            return a
        else:
            q_values = np.array([q.predict(self.featurize(state))[0] for q in self.q_functions])
            #print(q_values)
            return np.argmax(q_values)

        

        ########## Your code ends here #########


    def _to_squeezed_np(self, batch:Batch) -> Batch:
        """ A simple helper function squeeze data dimension and cover data format from tensor to np.ndarray."""
        _ret = {}
        for name, value in batch._asdict().items():
            if isinstance(value, dict): # this is for extra, which is a dict
                for k, v in value.items():
                    value[k] = v.squeeze().numpy()
                _ret[name] = value
            else:
                _ret[name] = value.squeeze().numpy()
        return Batch(**_ret)
        
    def update(self, buffer):
        # batch is a namedtuple, which has state, action, next_state, not_done, reward
        # you can access the value be batch.<name>, e.g, batch.state
        batch = buffer.sample(self.batch_size) 
        
        # the returned batch is a namedtuple, where the data is torch.Tensor
        # we first squeeze dim and then covert it to Numpy array.
        batch = self._to_squeezed_np(batch)

        # TODO: Task 1, update q_functions
        ########## You code starts here #########
        # Hints: 
        # 1. feature the state and next_state
        # 2. calculate q_target via TD(0),
        # 3. self.q_functions is a list which defines a q function for each action dimension
        #    for each q function, use predict(feature) to obtain the q value 
        # 4. to fit the q function, check the _initialize_model() might be helpful
        # featurize the state and next_state
        f_state = self.featurize(batch.state)
        f_next_state = self.featurize(batch.next_state)
        
        q_tar = batch.reward
        for i, status in enumerate(batch.not_done):
            if status==1:
                #temp = f_next_state[i,:]
                #temp = temp.reshape(1,-1)
                temp = batch.next_state[i,:]
                #print("Prööt")
                max_a = self.get_action(temp, epsilon=0.0)
                next_state = f_next_state[i,:]
                next_state = next_state.reshape(1,-1)
                q_tar[i] = q_tar[i] + self.gamma * self.q_functions[max_a].predict(next_state)

        


        # Get new weights for each action separately
        for a in range(self.num_actions):
            # Find states where `a` was taken
            idx = batch.action == a

            # If a not present in the batch, skip and move to the next action
            if np.any(idx):
                act_states = f_state[idx]
                act_targets = q_tar[idx]

                # TODO: Perform a single SGD step on the Q-function params to update the q_function corresponding to action a
                #loss = (q_tar - self.q_functions[idx].predict(act_states))**2
                (self.q_functions[a]).partial_fit(act_states, act_targets)

        ########## You code ends here #########
        # if you want to log something in wandb, you can put it inside the {}, otherwise, just leave it empty.
        return {}

    def save(self, fp):
        path = fp/'rbf.pkl'
        h.save_object(
            {'q': self.q_functions, 'featurizer': self.featurizer},
            path
        )

    def load(self, fp):
        path = fp/'rbf.pkl'
        d = h.load_object(path)

        self.q_functions = d['q']
        self.featurizer = d['featurizer']
