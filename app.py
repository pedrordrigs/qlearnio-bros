from getframe import getFrame
from directkeys import PressKey, ReleaseKey
from memoryread import memoryReset, memoryValues
from inputcodes import RUN, JUMP, SPIN, LEFT, RIGHT, RESET
from dqnetwork import dqNetwork
from memory import Memory

import time
import random
from skimage import transform
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf      
from collections import deque
import warnings
warnings.filterwarnings('ignore')

stack_size = 4

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

def keyPress(key, past_action):
    # if(key == [0, 1, 0, 0, 0]):
    #     key = SPIN
    # if(key == [0, 0, 1, 0, 0]):
    #     key = RUN

    if(key == [1, 0, 0]):
        key = JUMP
    elif(key == [0, 1, 0]):
        key = LEFT
    elif(key == [0, 0, 1]):
        key = RIGHT
    if(past_action == [1, 0, 0]):
        past_action = JUMP
    elif(past_action == [0, 1, 0]):
        past_action = LEFT
    elif(past_action == [0, 0, 1]):
        past_action = RIGHT

    if(key == RESET):
        PressKey(RESET)
        ReleaseKey(RESET)
    else:
        ReleaseKey(past_action)
        PressKey(key)

def fitnessFunction(score, distance, deltat, death):
    fitness = (score*3 + ((distance*6)/deltat))
    if(death == 9 or death == 9225):
        fitness = fitness-500
    return fitness
    # Acessar endereços de memória do emulador
    # Encontrar endereço de memória relacionado ao score e distancia
    # Levar em consideração score, distância percorrida e tempo para calcular o reward

    # Main Memory Address - 1408D8C40
    # Score address - offset F34 (Score/10)
    # X Position address - 000094 (X-axis)

def randomActions():
    JUMP = [1, 0, 0]
    LEFT = [0, 1, 0]
    RIGHT = [0, 0, 1]
    possible_actions = [JUMP, LEFT, RIGHT]
    return possible_actions
    # Desempenhar comandos aleatórios
    # Calcular e retornar fit de cada individuo
    # Utilizar conceitos evolucionais

def preprocessing(frame):
    preprocessed_frame = transform.resize(frame, [84,84])
    return preprocessed_frame
    # 4 image stacking

def enviroment():
    random.seed(time)
    deltat = 0
    reward = []
    while(1):
        values = memoryValues()
        if(values[0] == 9 or values[0] == 9225): # Death Flags Memory Address - 0071
            reward.append(fitnessFunction(values[1], values[2], deltat, values[0]))
            # keyPress(RESET, action_time)
            memoryReset() # Reset Score
            deltat = 0
            break

        action = random.choice(randomActions())
        action_time = random.randint(1, 50)/100
        # keyPress(action, action_time)
        # time.sleep(0.041) # Framerate 1/24 = 0.041seg/frame = 24fps
        deltat += (0.041 + action_time)
    i += 1
    return reward

def predict_action(explore_start, explore_stop, decay_rate, decay_step, state, action, sess, DQNetwork):
    ## EPSILON GREEDY STRATEGY
    # Choose action a from state s using epsilon greedy.
    ## First we randomize a number
    exp_exp_tradeoff = np.random.rand()

    # Here we'll use an improved version of our epsilon greedy strategy used in Q-learning notebook
    explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)
    
    if (explore_probability > exp_exp_tradeoff):
        # Make a random action (exploration)
        action = random.choice(randomActions())
        
    else:
        # Get action from Q-network (exploitation)
        # Estimate the Qs values state
        Qs = sess.run(DQNetwork.output, feed_dict = {DQNetwork.inputs_: state.reshape((1, *state.shape))})
        
        # Take the biggest Q value (= the best action)
        choice = np.argmax(Qs)
        action = randomActions()[int(choice)]
                
    return action, explore_probability

def main():
    stacked_frames  =  deque([np.zeros((84,84), dtype=np.int) for i in range(stack_size)], maxlen=4) 

    ### MODEL HYPERPARAMETERS
    state_size = [84,84,4]      # Our input is a stack of 4 frames hence 84x84x4 (Width, height, channels) 
    action_size = len(randomActions())             # 3 possible actions: left, right, shoot
    learning_rate =  0.0005      # Alpha (aka learning rate)

    ### TRAINING HYPERPARAMETERS
    total_episodes = 100        # Total episodes for training
    max_steps = 200              # Max possible steps in an episode
    batch_size = 64             

    # Exploration parameters for epsilon greedy strategy
    explore_start = 0.9           # exploration probability at start
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

    tf.compat.v1.reset_default_graph()
    
    DQNetwork = dqNetwork(state_size, action_size, learning_rate)

    # Instantiate memory
    memory = Memory(max_size = memory_size)

    # Render the environment

    random.seed(time)
    deltat = 0
    reward = 0
    done = False
    action = JUMP

    for i in range(pretrain_length):
        values = memoryValues()
        death = False
        if i == 0:
            # First we need a state
            state = getFrame()
            print(state.shape)
            state, stacked_frames = stack_frames(stacked_frames, state, True)


        past_action = action

        # Random Action
        action = random.choice(randomActions())
        keyPress(action, past_action)

        # Reward for action
        deltat += (0.05)
        reward = fitnessFunction(values[1], values[2], deltat, values[0])

        if(values[0] == 9 or values[0] == 9225): # Death Flags Memory Address - 0071
            done = True

            next_state = np.zeros(state.shape)
            memory.add((state, action, reward, next_state, done))

            keyPress(action, past_action)
            memoryReset() # Reset Score
            deltat = 0

            state = getFrame()
            state, stacked_frames = stack_frames(stacked_frames, state, True)
        else:
            next_state = getFrame()
            next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
            memory.add((state, action, reward, next_state, done))
            state = next_state

    # Setup TensorBoard Writer
    writer = tf.compat.v1.summary.FileWriter("/tensorboard/dqn/1")
     ## Losses
    tf.compat.v1.summary.scalar("Loss", DQNetwork.loss)

    write_op = tf.compat.v1.summary.merge_all()

    saver = tf.compat.v1.train.Saver()
# --------------------------------------------------------------------------------------------
    if training == True:
        with tf.compat.v1.Session() as sess:
            saver.restore(sess, "./models/model.ckpt")
            # Initialize the variables
            sess.run(tf.compat.v1.global_variables_initializer())
            
            # Initialize the decay rate (that will use to reduce epsilon) 
            decay_step = 0

            # Init the game

            for episode in range(total_episodes):
                # Set step to 0
                step = 0
                keyPress(RESET, past_action)
                memoryReset() # Reset Score
                values = memoryValues()
                deltat = 0
                # Initialize the rewards of the episode
                episode_rewards = []
                # Make a new episode and observe the first state
                state = getFrame()
                # Remember that stack frame function also call our preprocess function.
                state, stacked_frames = stack_frames(stacked_frames, state, True)

                while step < max_steps:
                    values = memoryValues()
                    step += 1
                    # Increase decay_step
                    decay_step +=1
                    past_action = action
                    # Predict the action to take and take it
                    action, explore_probability = predict_action(explore_start, explore_stop, decay_rate, decay_step, state, randomActions(), sess, DQNetwork)
                    keyPress(action, past_action)
                    # Do the action
                    deltat += (0.04)
                    reward = fitnessFunction(values[1], values[2], deltat, values[0])
                    # time.sleep(0.041) # Framerate 1/24 = 0.041seg/frame = 24fps
                    # Look if the episode is finished
                    
                    # Add the reward to total reward
                    episode_rewards.append(reward)

                    # If the game is finished
                    if(values[0] == 9 or values[0] == 9225): # Death Flags Memory Address - 0071
                        # the episode ends so no next state
                        next_state = np.zeros((84,84), dtype=np.int)
                        next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)

                        # Set step = max_steps to end the episode
                        step = max_steps

                        # Get the total reward of the episode
                        total_reward = np.sum(episode_rewards)

                        print('Episode: {}'.format(episode),
                                'Total reward: {}'.format(total_reward),
                                'Training loss: {:.4f}'.format(loss),
                                'Explore P: {:.4f}'.format(explore_probability))

                        memory.add((state, action, reward, next_state, done))

                        keyPress(RESET, past_action)
                        memoryReset() # Reset Score
                        deltat = 0

                    else:
                        # Get the next state
                        next_state = getFrame()
                        # Stack the frame of the next_state
                        next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
                        

                        # Add experience to memory
                        memory.add((state, action, reward, next_state, done))
                        
                        # st+1 is now our current state
                        state = next_state

                    ### LEARNING PART            
                    # Obtain random mini-batch from memory
                    batch = memory.sample(batch_size)
                    states_mb = np.array([each[0] for each in batch], ndmin=3)
                    actions_mb = np.array([each[1] for each in batch])
                    rewards_mb = np.array([each[2] for each in batch]) 
                    next_states_mb = np.array([each[3] for each in batch], ndmin=3)
                    dones_mb = np.array([each[4] for each in batch])

                    target_Qs_batch = []

                    # Get Q values for next_state 
                    Qs_next_state = sess.run(DQNetwork.output, feed_dict = {DQNetwork.inputs_: next_states_mb})
                    
                    # Set Q_target = r if the episode ends at s+1, otherwise set Q_target = r + gamma*maxQ(s', a')
                    for i in range(0, len(batch)):
                        terminal = dones_mb[i]

                        # If we are in a terminal state, only equals reward
                        if terminal:
                            target_Qs_batch.append(rewards_mb[i])
                            
                        else:
                            target = rewards_mb[i] + gamma * np.max(Qs_next_state[i])
                            target_Qs_batch.append(target)
                            

                    targets_mb = np.array([each for each in target_Qs_batch])

                    loss, _ = sess.run([DQNetwork.loss, DQNetwork.optimizer],
                                        feed_dict={DQNetwork.inputs_: states_mb,
                                                DQNetwork.target_Q: targets_mb,
                                                DQNetwork.actions_: actions_mb})

                    summary = sess.run(write_op, feed_dict={DQNetwork.inputs_: states_mb,
                                                    DQNetwork.target_Q: targets_mb,
                                                    DQNetwork.actions_: actions_mb})
                    writer.add_summary(summary, episode)
                    writer.flush()
                    # Write TF Summaries

                # Save model every 5 episodes
                if episode % 5 == 0:
                    save_path = ""
                    save_path = saver.save(sess, "./models/model.ckpt")
                    print("Model Saved")

main()