from game_hook import hook
from directkeys import PressKey, ReleaseKey

import time
import random
from skimage import transform
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf      
from collections import deque
import warnings
warnings.filterwarnings('ignore')

def keyPress(key, time):
    PressKey(key)
    time.sleep(time)
    ReleaseKey(key)

def fitnessFunction():
    # Acessar endereços de memória do emulador
    # Encontrar endereço de memória relacionado ao score
    # Levar em consideração distância percorrida e tempo no score fitness

    # (Δt * Sf) - score

def randomActions():
    # Desempenhar comandos aleatórios
    # Calcular e retornar fit de cada individuo
    # Utilizar conceitos evolucionais

def preprocessing():
    # Scaledown + Greyscale
    # 4 image stacking

def model():
    # Definição de hyperparametros

def main():
    while(1):
        img = hook()
        # Framerate 1/24 = 0.041seg/frame = 24fps
        time.sleep(0.041)

main()