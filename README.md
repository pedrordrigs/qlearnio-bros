# qlearnio-bros
Super Mario World - Deep-Q-Learning Project

# Introduction
That's an implementation of a Deep Q-Network without using OpenAI Gym as you commonly see around other projects of this kind.

Implementing without using OpenAI Gym it's a way harder task, but also give us some insight on how to implement this kind of technology on projects that couldn't count on support libraries to make things easier.

# First Steps

The first step on your first Deep Q-Learning project is choosing which problem (games in this specific case) you could solve with Markov decision process (MDP).
Also preferably a problem that you could easily spot which metrics you could extract to determine the model performance.

I chose Super Mario World (SNES) to develop this project.

Why? First of all, as we all know SMW is a pretty old game that we could easily emulate, speed up and wouldn't need to feed our neural network with high resolution images for it to perform well.

Also, SMW has a linear game design, we could easily determine the fitness metrics based on those five things: 

**X-position** - our main goal is to reach the rightmost of the screen (the end pole) so we can take how much Mario advanced right on the phase as the main positive metric.

**Score** - we want our model to learn that kill the enemies on the screen is a positive thing, so he will not just avoid enemies (that would be boring if you ask me).

**Coins** - we also want our model to take all the coins in the path and hit the mysterious boxes.

**Time** - the time that our model took to reach the current position.

**Death** - last but not least, death, we want our model to avoid it so everytime our model take an action that results on death it must receive a negative feedback.

# Environment
This project was made and tested only in Windows 10 and probably won't work on other OS.

Those external libraries were used: **Tensorflow 2.9.2**, **scikit-learn**, **matplotlib**, **numpy**, **Pillow**, **Win32gui**, **Win32ui**.

Necessary drivers for running it on GPU:

Nvidia CUDA 11.7 | compatible Nvidia Cuda Toolkit | cudNN 8.5
