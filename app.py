from windowhook import hook
from directkeys import PressKey, ReleaseKey
from memoryread import memoryReset, memoryValues
from inputcodes import RUN, JUMP, SPIN, LEFT, RIGHT, RESET

import time
import random
from skimage import transform
import numpy as np
import matplotlib.pyplot as plt
# import tensorflow as tf      
from collections import deque
import warnings
warnings.filterwarnings('ignore')

def stack_frames(stacked_frames, state, is_new_episode):
    # Preprocess frame
    frame = preprocessing(state)
    if is_new_episode:
        # Clear our stacked_frames
        stacked_frames = deque([np.zeros((84,84), dtype=np.int) for i in range(stack_size)], maxlen=4)
        
        # Because we're in a new episode, copy the same frame 4x
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        
        # Stack the frames
        stacked_state = np.stack(stacked_frames, axis=2)
        
    else:
        # Append frame to deque, automatically removes the oldest frame
        stacked_frames.append(frame)

        # Build the stacked state (first dimension specifies different frames)
        stacked_state = np.stack(stacked_frames, axis=2) 
    
    return stacked_state, stacked_frames

def keyPress(key, action_time):
    if(key == [1, 0, 0, 0, 0]):
        key = JUMP
    if(key == [0, 1, 0, 0, 0]):
        key = SPIN
    if(key == [0, 0, 1, 0, 0]):
        key = RUN
    if(key == [0, 0, 0, 1, 0]):
        key = LEFT
    if(key == [0, 0, 0, 0, 1]):
        key = RIGHT
    PressKey(key)
    time.sleep(action_time)
    ReleaseKey(key)

def fitnessFunction(score, distance, deltat):
    fitness = score/3 + ((distance*5)/deltat)
    return fitness
    # Acessar endereços de memória do emulador
    # Encontrar endereço de memória relacionado ao score e distancia
    # Levar em consideração score, distância percorrida e tempo para calcular o reward

    # Main Memory Address - 1408D8C40
    # Score address - offset F34 (Score/10)
    # X Position address - 000094 (X-axis)

def randomActions():
    JUMP = [1, 0, 0, 0, 0]
    SPIN = [0, 1, 0, 0, 0]
    RUN = [0, 0, 1, 0, 0]
    LEFT = [0, 0, 0, 1, 0]
    RIGHT = [0, 0, 0, 0, 1]
    possible_actions = [JUMP, SPIN, RUN, LEFT, RIGHT]
    return possible_actions
    # Desempenhar comandos aleatórios
    # Calcular e retornar fit de cada individuo
    # Utilizar conceitos evolucionais

def preprocessing(frame):
    preprocessed_frame = transform.resize(normalized_frame, [84,84])
    return preprocessed_frame
    # Scaledown + Greyscale
    # 4 image stacking

class DQNetwork:
    def __init__(self, state_size, action_size, learning_rate, name='DQNetwork'):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        
        with tf.variable_scope(name):
            # We create the placeholders
            # *state_size means that we take each elements of state_size in tuple hence is like if we wrote
            # [None, 84, 84, 4]
            self.inputs_ = tf.placeholder(tf.float32, [None, *state_size], name="inputs")
            self.actions_ = tf.placeholder(tf.float32, [None, 3], name="actions_")
            
            # Remember that target_Q is the R(s,a) + ymax Qhat(s', a')
            self.target_Q = tf.placeholder(tf.float32, [None], name="target")
            
            """
            First convnet:
            CNN
            BatchNormalization
            ELU
            """
            # Input is 84x84x4
            self.conv1 = tf.layers.conv2d(inputs = self.inputs_,
                                         filters = 32,
                                         kernel_size = [8,8],
                                         strides = [4,4],
                                         padding = "VALID",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                         name = "conv1")
            
            self.conv1_batchnorm = tf.layers.batch_normalization(self.conv1,
                                                   training = True,
                                                   epsilon = 1e-5,
                                                     name = 'batch_norm1')
            
            self.conv1_out = tf.nn.elu(self.conv1_batchnorm, name="conv1_out")
            ## --> [20, 20, 32]
            
            
            """
            Second convnet:
            CNN
            BatchNormalization
            ELU
            """
            self.conv2 = tf.layers.conv2d(inputs = self.conv1_out,
                                 filters = 64,
                                 kernel_size = [4,4],
                                 strides = [2,2],
                                 padding = "VALID",
                                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                 name = "conv2")
        
            self.conv2_batchnorm = tf.layers.batch_normalization(self.conv2,
                                                   training = True,
                                                   epsilon = 1e-5,
                                                     name = 'batch_norm2')

            self.conv2_out = tf.nn.elu(self.conv2_batchnorm, name="conv2_out")
            ## --> [9, 9, 64]
            
            
            """
            Third convnet:
            CNN
            BatchNormalization
            ELU
            """
            self.conv3 = tf.layers.conv2d(inputs = self.conv2_out,
                                 filters = 128,
                                 kernel_size = [4,4],
                                 strides = [2,2],
                                 padding = "VALID",
                                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                 name = "conv3")
        
            self.conv3_batchnorm = tf.layers.batch_normalization(self.conv3,
                                                   training = True,
                                                   epsilon = 1e-5,
                                                     name = 'batch_norm3')

            self.conv3_out = tf.nn.elu(self.conv3_batchnorm, name="conv3_out")
            ## --> [3, 3, 128]
            
            
            self.flatten = tf.layers.flatten(self.conv3_out)
            ## --> [1152]
            
            
            self.fc = tf.layers.dense(inputs = self.flatten,
                                  units = 512,
                                  activation = tf.nn.elu,
                                       kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                name="fc1")
            
            
            self.output = tf.layers.dense(inputs = self.fc, 
                                           kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                          units = 3, 
                                        activation=None)

  
            # Q is our predicted Q value.
            self.Q = tf.reduce_sum(tf.multiply(self.output, self.actions_), axis=1)
            
            
            # The loss is the difference between our predicted Q_values and the Q_target
            # Sum(Qtarget - Q)^2
            self.loss = tf.reduce_mean(tf.square(self.target_Q - self.Q))
            
            self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)

class Memory():
    def __init__(self, max_size):
        self.buffer = deque(maxlen = max_size)
    
    def add(self, experience):
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        buffer_size = len(self.buffer)
        index = np.random.choice(np.arange(buffer_size),
                                size = batch_size,
                                replace = False)
        
        return [self.buffer[i] for i in index]

def enviroment():
    random.seed(time)
    deltat = 0
    reward = []
    while(1):
        values = memoryValues()
        if(values[0] == 9 or values[0] == 9225): # Death Flags Memory Address - 0071
            reward.append(fitnessFunction(values[1], values[2], deltat))
            keyPress(RESET, action_time)
            memoryReset() # Reset Score
            deltat = 0
            break

        action = random.choice(randomActions())
        action_time = random.randint(1, 50)/100
        keyPress(action, action_time)
        time.sleep(0.041) # Framerate 1/24 = 0.041seg/frame = 24fps
        deltat += (0.041 + action_time)
    i += 1
    return reward

def main():
    stack_size = 4
    stacked_frames  =  deque([np.zeros((84,84), dtype=np.int) for i in range(stack_size)], maxlen=4) 

    ### MODEL HYPERPARAMETERS
    state_size = [84,84,4]      # Our input is a stack of 4 frames hence 84x84x4 (Width, height, channels) 
    action_size = game.get_available_buttons_size()              # 3 possible actions: left, right, shoot
    learning_rate =  0.0002      # Alpha (aka learning rate)

    ### TRAINING HYPERPARAMETERS
    total_episodes = 500        # Total episodes for training
    max_steps = 100              # Max possible steps in an episode
    batch_size = 64             

    # Exploration parameters for epsilon greedy strategy
    explore_start = 1.0            # exploration probability at start
    explore_stop = 0.01            # minimum exploration probability 
    decay_rate = 0.0001            # exponential decay rate for exploration prob

    # Q learning hyperparameters
    gamma = 0.95               # Discounting rate

    ### MEMORY HYPERPARAMETERS
    pretrain_length = batch_size   # Number of experiences stored in the Memory when initialized for the first time
    memory_size = 1000000          # Number of experiences the Memory can keep

    ### MODIFY THIS TO FALSE IF YOU JUST WANT TO SEE THE TRAINED AGENT
    training = True

    ## TURN THIS TO TRUE IF YOU WANT TO RENDER THE ENVIRONMENT
    episode_render = False

    tf.reset_default_graph()

    DQNetwork = DQNetwork(state_size, action_size, learning_rate)

        # Instantiate memory
    memory = Memory(max_size = memory_size)

    # Render the environment

    random.seed(time)
    deltat = 0
    reward = []

    for i in range(pretrain_length):
        values = memoryValues()
        if i == 0:
            # First we need a state
            state = hook()
            state, stacked_frames = stack_frames(stacked_frames, state, True)

        if(values[0] == 9 or values[0] == 9225): # Death Flags Memory Address - 0071
            reward.append(fitnessFunction(values[1], values[2], deltat))
            keyPress(RESET, action_time)
            memoryReset() # Reset Score
            deltat = 0
            i += 1
        else:
            action = random.choice(randomActions())
            action_time = random.randint(1, 50)/100
            keyPress(action, action_time)
            time.sleep(0.041) # Framerate 1/24 = 0.041seg/frame = 24fps
            deltat += (0.041 + action_time)
    i += 1
        if i == 0:
            # First we need a state
            state = game.get_state().screen_buffer
            state, stacked_frames = stack_frames(stacked_frames, state, True)
        
        # Random action
        action = random.choice(possible_actions)
        
        # Get the rewards
        reward = game.make_action(action)
        
        # Look if the episode is finished
        done = game.is_episode_finished()
        
        # If we're dead
        if done:
            # We finished the episode
            next_state = np.zeros(state.shape)
            
            # Add experience to memory
            memory.add((state, action, reward, next_state, done))
            
            # Start a new episode
            game.new_episode()
            
            # First we need a state
            state = game.get_state().screen_buffer
            
            # Stack the frames
            state, stacked_frames = stack_frames(stacked_frames, state, True)
            
        else:
            # Get the next state
            next_state = game.get_state().screen_buffer
            next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
            
            # Add experience to memory
            memory.add((state, action, reward, next_state, done))
            
            # Our state is now the next_state
            state = next_state
