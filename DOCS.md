# QAM:

This is a trading bot deployment package which includes data collection, data preparation, training and inference part. The model (Trading Bot) used here trained is based on reinforcement learning paradigm.

## Details:

### Data Collection:

- Supports data download sources like YFinance(from Yahoo), and Alpaca market data server.
- Collects and processes data by applying normalisation && adding extra stock informations like (blah, blah, blah)....
- Splitted storing of data for different time intervals (like 1minute, 1hour etc.)

### Model Training:

- Since market prediction is a Markov Decision Process and since our model needs some state and a sequential decision making environment, we chose to go with RL based approach.
- Provided, starting from random weights for an RL model could end up in divergence with a very high probability. So we developed a two stage training pipeline.
    - The 1st stage of training is a Self-Supervised learning paradigm, where the model naively learns the trend patterns of different stocks.
        - This way our model is aware of the market dynamics before stepping into the RL paradigm.
        - We'll feed the model with enough samples of history timepoints and ask the model to predict the direction of trend in the future, somewhat like a classification task. The choice of going to a classification task rather than a similarity reduction of future timepoint is that, CrossEntropyLoss constraint is more stable and more intuitive for our follow-up downstream task than any other learning idea.
    - The 2nd stage of training is a Reinforcement Learning paradigm, where the model learns to make sequential decisions which helps in getting higher returns.
        - For this we chose a policy gradient based method, PPO due to it's simplicity and promising convergence over a lot of experiments that've published in many other domains (particularly LLM space).

#### Model Architecture:

- 1st stage training
    - For this, we've chosen FastConformer based architecture, having a solid foundation in speech domain, where we could structurally match high similarities between spectrogram and stock timepoints, and also due to the enormous success of FastConformer architecture, it's impressive learning ability both spacially and temporally.
        - We got 18 layers of FastConformer Encoder with 8x subsampling factor and a classification head, with each class point the trend prediction, like UP, DOWN, VERY UP, VERY DOWN etc...

- 2nd stage training
    - Since for this stage, we need both the input timepoints as well as the holding state (i.e) whether we've bought and holding or waiting to buy, we need one more input to be processed by model.
    - So we chose to process the timepoint state by 1st 6 layers of FastConformer, and the timepoint of entry by an MLP, and we'll add up those two and pass the latent vector to the rest of the 12 layers of FastConformer.
    - For policy and value networks, we share the same FastConformer encoder weights, and have seperate projection heads for each.

### Inference:

- Since trading requires really low-latency processing, we've planed to export the model to ONNX via torch (for model training we use PyTorch), and write the inference logic in C++.
