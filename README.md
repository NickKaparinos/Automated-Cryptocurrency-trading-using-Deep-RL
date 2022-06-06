# Automated Cryptocurrency trading using Deep RL
Automated trading is a method of participating in financial markets by using a computer program that makes automaticly the trading decisions and then executes them. Usually, the trading algorithm executes pre-set rules for entering and exiting trades. In contrast, in this project, the computer learns a trading policy using Machine Learning. Specifically, the trading algorithm is a Deep Reinforcement Learning (Deep RL) Agent. The agent is trained using trial and error in order to maximize the expected Reward. The Reward Function chosen is the trading Profit and Loss, also know as [PnL](https://en.wikipedia.org/wiki/PnL_Explained). In this project, the Agent is able to trade bewteen 4 Cryptocurrencies (ADA, BTC, ETH and LTC) and US Dollars. The agents starts with $1000.

# Reinforcement Learning Algorithm
The Reinforcement Learning algorithm chosen is Double Dueling Deep Q Learning (DQN) using Prioritized Experience Replay. In Deep DQN, a neural network is trained in order to approximate the Q and V functions.

# Actions
The RL algorithm can take 5 actions, one corresponding to each cryptocurrency and one corresponding to US Dollars. If the agent does not have in its portfolio the currency chosen, then, in the same timestep, it sells its whole portfolio and then buys as many of the selected currency as possible. If the agent choses a currency it already owns, then the action acts as a Hold. The agent takes an action once every hour.

# State
The state consists of the High, Low and Close statistics for each cryptocurrency in the last 24 Hours. Also, it contains a One-Hot vector of length 5 that informs the Agent of which currency it currently has in its portfolio.



# Neural Network
<p align="center"><img src="https://github.com/NickKaparinos/Automated-Cryptocurrency-trading-using-Deep-RL/blob/master/Results/crypto_arch2.png" alt="drawing"/>
  
First, the Timeseries are encoded using separate encoders into Timeseries embeddings. Encoder Architectures experimented with include Multi Layer Perceptrons (MLPs), 1D Convolutional Neural Networks (CNNs), Recurrent Neural Networks (RNNs) as well as Transformers. LSTM networks provided the optimal performance.
  
Then, the Timeseries embeddings are concatenated together, as well as with the portfolio embedding into the final state embedding.
  
Finally, MLP Head networks map the state embedding to the Q and V values.
  
# Evaluation
After a specified number of training steps (Epoch) the Agent is evaluated. The evaluation episode is 1000 Timesteps long. The agent was trained and evaluated without trading fees. However, the final evaluation was conducted using the standard 0.1% trading fee. Surprisingly, this training regime resulted in the optimal performance. The best way to evaluate trading agents is against the naive method called Buy And Hold (B&H). This comparison provides a more accuracy estimation of the agent`s performance, since it is mostly aleviated of the effect of the market trend throughout the evaluation episode. The following plots depict the Agent`s performance after each training Epoch.
  

<p align="center"><img src="https://github.com/NickKaparinos/Automated-Cryptocurrency-trading-using-Deep-RL/blob/master/Results/epoch_bah_boxplot_distribution-final.png" alt="drawing" width="500"/>

<p align="center"><img src="https://github.com/NickKaparinos/Automated-Cryptocurrency-trading-using-Deep-RL/blob/master/Results/epoch_boxplot_distribution-final.png" alt="drawing" width="500"/>
  
# Final Evaluation
After training, the final agent was evaluated using the standard 0.1% trading fee. In 50 Evaluation episodes, the average Agent`s performance was on average **30%** better than the Buy And Hold strategy.
  
T-Test was conducted to determine the statistical significance of the results. The p-value against the null hypothesis that the Buy&Hold strategy is better than the RL Agent is **0.000006**.
  
 <p align="center"><img src="https://github.com/NickKaparinos/Automated-Cryptocurrency-trading-using-Deep-RL/blob/master/Results/boxplot-final-vs-bah-ratio.png" alt="drawing" width="500"/>

<p align="center"><img src="https://github.com/NickKaparinos/Automated-Cryptocurrency-trading-using-Deep-RL/blob/master/Results/boxplot-final-ratio.png" alt="drawing" width="500"/>
  
## Final Evaluation Episode 1
  <p align="center"><img src="https://github.com/NickKaparinos/Automated-Cryptocurrency-trading-using-Deep-RL/blob/master/Results/test_episode-0-actions.png" alt="drawing"/>
    
 <p float="left">
  <img src="https://github.com/NickKaparinos/Automated-Cryptocurrency-trading-using-Deep-RL/blob/master/Results/test_episode-0-portfolio.png" width="32.9%" /> 
  
  <img src="https://github.com/NickKaparinos/Automated-Cryptocurrency-trading-using-Deep-RL/blob/master/Results/test_episode-0-reward.png" width="32.9%" />
   <img src="https://github.com/NickKaparinos/Automated-Cryptocurrency-trading-using-Deep-RL/blob/master/Results/test_episode-0-reward-per-series.png" width="32.9%" />
   
## Final Evaluation Episode 2
   
   <p align="center"><img src="https://github.com/NickKaparinos/Automated-Cryptocurrency-trading-using-Deep-RL/blob/master/Results/test_episode-1-actions.png" alt="drawing"/>
    
 <p float="left">
  <img src="https://github.com/NickKaparinos/Automated-Cryptocurrency-trading-using-Deep-RL/blob/master/Results/test_episode-1-portfolio.png" width="32.9%" /> 
  
  <img src="https://github.com/NickKaparinos/Automated-Cryptocurrency-trading-using-Deep-RL/blob/master/Results/test_episode-1-reward.png" width="32.9%" />
   <img src="https://github.com/NickKaparinos/Automated-Cryptocurrency-trading-using-Deep-RL/blob/master/Results/test_episode-1-reward-per-series.png" width="32.9%" />
