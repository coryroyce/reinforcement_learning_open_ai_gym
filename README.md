# Reinforcement Learning with OpenAI Gym

Apply Reinforcement Learning (RL) to a simulated environment and then improve it with deep learning networks. A sub goal was to work within Gym entirely in the browser with Colab Notebooks.

Check out the complete Colab Notebook [here](https://github.com/coryroyce/reinforcement_learning_open_ai_gym/blob/main/notebook/220306_CMPE_252_HW_4_Reinforcement_Learning_Cory_Randolph.ipynb)



| ![Untrained Model](https://github.com/coryroyce/reinforcement_learning_open_ai_gym/blob/main/reference/Mountain_Car_Untrained.gif) | ![Basic Q-Learning (5,000 steps)](https://github.com/coryroyce/reinforcement_learning_open_ai_gym/blob/main/reference/Mountain_Car_Basic_Q_Learning_Trained_5k_steps.gif) | ![Deep Q-Network (200,000 steps)](https://github.com/coryroyce/reinforcement_learning_open_ai_gym/blob/main/reference/Mountain_Car_DQN_Trained_200k_steps.gif) |
| :---:   |    :---: |  :---: |
| Untrained Model   | Basic Q-Learning    | Deep Q-Network    |


# Project Overview
1. Setup Gym Environment (Mountain Car environment)
1. Apply Basic Q-learning
1. Improve Score with Deep Learning


# Setup Gym Environment
There are many reference tutorials on what OpenAi Gym is and how to use them, but one area I found lacking was how to render the video environments from within Google Colab.Check out my [Development Notebook]() for full details, but the highlight is that after installing a few additional visualization packages, you can wrap the rendering environment so that it save video file locally in Colab and then can be downloaded if needed.


# Apply Basic Q-learning
I applied a basic Q-learning based on Genevieve Hayes' [article](https://towardsdatascience.com/getting-started-with-reinforcement-learning-and-open-ai-gym-c289aca874f) and while it provide results that were able to complete the task, I had a feeling it could be done more optimally. For example, the score is for the Mountain Car environment is based on the number of frames it takes to get the car past the flag with a maximum of 200 frames cut off, so a failing score is -200. The Q-learning was able to get scores in the -160 to -190 (see image below) after 5,000 to 10,000 training iterations.

![Q-learning Scores (Iterations vs Score)](https://github.com/coryroyce/reinforcement_learning_open_ai_gym/blob/main/reference/Basic_Q_Learning_Score_Results.jpg)


# Improve Score with Deep Learning
Since I wanted to apply a Deep Q-Network (DQN) that I could control the architecture of I used Keras RL to build out a model and fit it. While this took longer to train, it was able to keep learning and past the plateau from basic Q-learning. The DQN network was able to get scores around -80 depending on the initial starting position of the car. The optimization can easily bee seen when comparing the 3 video (random Actions, Basic Q-Learning, And Deep Q-Network)


# Potential Future Work
Apply Stable Baselines models to the task to see if performance can be further optimized.


# Reference

Reviewed Q-policy RL from [Genevieve Hayes](https://towardsdatascience.com/getting-started-with-reinforcement-learning-and-open-ai-gym-c289aca874f)

Got the Colab install dependencies and video saving from  [cwkx's video](https://www.youtube.com/watch?v=BNSwFURmaCA&ab_channel=cwkx)

RL Overview picture and comments from [sadiakhaf](https://github.com/sadiakhaf/IEEE-Hands-On-RL-using-Python)

Sample code for using Keras RL with Mountain Car [aslamplr](https://github.com/aslamplr/mountaincar_gym)
