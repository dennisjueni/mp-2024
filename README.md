# Human Motion Prediction

## Task Description
You are given sequences of pre-recorded human motion data and your task is to predict how the motion continues for several frames in the future.
You are expected to train and evaluate several models that solve this task.
Along with the data, you will receive a skeleton code written in pytorch that can load the data, train a dummy neural network and visualize predictions.
The code is only meant as a guideline; you are free to modify it to whatever extent you deem necessary.

## Approach
Our approach was to mostly leverage the architecture explained in this [paper](https://arxiv.org/pdf/1910.09070), and then improve the model by adding some additional attention connections motivated by human joing connections.
