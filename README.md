# qlearnio-bros
Super Mario World - Deep-Q-Learning Project

# Introduction
That's an implementation of a Deep Q-Network without using OpenAI Gym as you commonly see around other projects of this kind.

Implementing without using OpenAI Gym it's a way harder task, but also give us some insight on how to implement this kind of technology on projects that couldn't count on support libraries to make things easier.

# Environment
This project was made and tested only in Windows 10 and probably won't work on other OS.

Those external libraries were used: **Tensorflow 2.9.1**, **scikit-learn**, **matplotlib**, **numpy**, **Pillow**, **Win32gui**, **Win32ui**.

Necessary drivers for running it on GPU:

Nvidia CUDA 11.7 | compatible Nvidia Cuda Toolkit | cudNN 8.5

# First Steps

The first step on your first Deep Q-Learning project is choosing which problem (games in this specific case) you could solve with Markov decision process (MDP).
Also preferably a problem that you could easily spot which metrics you could extract to determine the model performance.

I chose Super Mario World (SNES) to develop this project.

Why? First of all, as we all know SMW is a pretty old game that we could easily emulate, speed up and theres no need to feed our neural network with high resolution images for it to perform well.

Also, considering that SMW has a linear game design, so we could easily determine the fitness metrics based on those five things: 

**X-position** - our main goal is to reach the rightmost of the screen (the end pole) so we can take how much Mario advanced right on the phase as the main positive metric.

**Score** - we want our model to learn that kill the enemies on the screen is a positive thing, so he will not just avoid enemies (that would be boring if you ask me).

**Coins** - we also want our model to take all the coins in the path and hit the mysterious boxes.

**Time** - the time that our model took to reach the current position.

**Death** - last but not least, death, we want our model to avoid it so everytime our model take an action that results on death it must receive a negative feedback.

# Extracting Frames

By using **Pillow** we can extract a frame everytime we call a specific function that you can find on **getframe.py**.

It's a really good idea to crop useless portions of the frame so the Q-Network can rely on mmore relevant image data.

I strongly recommend stacking atleast 4 frames to feed the Q-Network an image with some kind of motion that may be taken in consderation to evaluate the next action based on the current trajectory, momentum and so on.

That's how the image look like before the staking:

<img src="https://github.com/pedrordrigs/qlearnio-bros/blob/main/images/targetwindow.jpg?raw=true" alt="Frame" style="height: 400px; width:450px;"/>

# Fitness Metrics Extraction

As what was discussed on the introduction, OpenAI Gym makes the implementation way easier, we can get every single metric with a simple call.

The source of our previously discussed metrics are the memory addresses of the emulator that stores each one of the metrics.

You can check out the data extraction function on **memoryread.py**.

I've used Cheat Engine to reverse engineer the game so I could find the static address of each metric I needed.

# Actions

One of the first things we got to have in mind are which actions that our model can take as an output.

In the first iteration of this project I've decided to let our model run right, left and jump. That's enough for it to finish the level and also explore.

So we should make an action matrix that may look like this:

RIGHT = [1,0,0]
LEFT = [0,1,0]
JUMP = [0,0,1]

You can check the keypress function on app.py to learn how the neural network output turn into a input to our game.

# Game Loop

That's where we replicate the game behaviour such as the actions inputs, game resets, control how many generations and episodes we are in and so on.

You can check the loops on app.py which slight differences.
