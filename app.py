from game_hook import hook
from directkeys import PressKey, ReleaseKey
from memoryread import memoryValues
import time
import random
from skimage import transform
import numpy as np
import matplotlib.pyplot as plt
# import tensorflow as tf      
# from collections import deque
import warnings
warnings.filterwarnings('ignore')

def keyPress(key, time):
    PressKey(key)
    time.sleep(time)
    ReleaseKey(key)

def fitnessFunction():
    score = memoryValues()
    print(score)
    return(score)
    # Acessar endereços de memória do emulador
    # Encontrar endereço de memória relacionado ao score
    # Levar em consideração distância percorrida e tempo no score fitness

    # (Sf/Δt) + score

    # Main Memory Address - 1408D8C40
    # Score address - offset F34 (Score/10)
    # X Position address - 000094 (X-axis)
    # Time - 000F33 (seconds)

def randomActions():
    return
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
    while(1):
        img = hook()
        # Framerate 1/24 = 0.041seg/frame = 24fps
        time.sleep(0.5)
main()