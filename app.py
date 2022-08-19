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
# from collections import deque
import warnings
warnings.filterwarnings('ignore')

def keyPress(key, action_time):
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
    possible_actions = [JUMP, SPIN, RUN, LEFT, RIGHT]
    return possible_actions
    # Desempenhar comandos aleatórios
    # Calcular e retornar fit de cada individuo
    # Utilizar conceitos evolucionais

def preprocessing():
    return
    # Scaledown + Greyscale
    # 4 image stacking

def model():
    return
    # Definição de hyperparametros

def main():
    random.seed(time)
    deltat = 0
    episodes = 10
    reward = []
    for i in range(episodes):
        time.sleep(3)
        while(1):
            values = memoryValues()
            img = hook()

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
main()