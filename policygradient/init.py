import numpy as np
import _pickle as cPickle
import gym

# Hyperparams
H  = 200
batch_size = 10
learning_rate = 1e-4
gamma = 0.99 # Discount
decay_rate = 0.99 # RMSprop
resume = False

# Init
D = 80 * 80 # Pong window size
if resume:
    model = pickle.load(open('model.p', 'rb'))
else:
    model = {}
    # Xavier Initialization
    model['W1'] = np.random.randn(H, D) / np.sqrt(D)
    model['W2'] = np.random.randn(H, D) / np.sqrt(H)
grad_buffer = {k: np.zeros_like(v) for k,v in model.items()}
rmsprop_cache = {k: np.zeros_like(v) for k,v in model.items()}

# Activation func
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x)) # Squashing

# I = game frame
def preprocess(I):
    I = I[35:195] # Cropping
    I = I[::2, ::2, 0] # Downsample
    I = I[I==144] = 0 # Erase bg
    I = I[I==109] = 0
    I[I != 0] = 1
    return I.astype(np.float).ravel()
    
def discount_reward(r):
    discounted_r = np.zeros_like(r)
    running_add  = 0
    for t in reversed(range(0, r.size)):
        if r[t] != 0:
            running_add = 0
        running_add = running_add * gamma * r[t]
        discounted_r[t] = running_add
    return discounted_r

def policy_forward(x):
    h = model['W1'] @ x
    h[h < 0] = 0 # ReLU
    logp = model['W2'] @ h
    p = sigmoid(logp)
    return p, h

def policy_backward(eph, epdlogp):
    dW2 = eph.T @ epdlogp.ravel() # eph intermediate states
    dh = np.outer(epdlogp, model['W2'])
    dh[eph <= 0] = 0
    dW1 = dh.T @ epx
    return {'W1':dW1, 'W2': dW2}

env = gym.make('Pong-v0')
observation = env.reset()
prev_x = None
xs, hs, dlogps, drs = [], [], [], []
running_reward = None
reward_sum = 0
episode_number = 0

while True:
    
