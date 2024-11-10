**Beta v0**
- Train three models, one on 15mins chart and another on 60mins chart and another on 5mins chart
    - to capture both primary and short-term trend
- For pretraining, use masked 4 class prediction conformer(fast-conformer) encoder model, (the 4 are more high, high, low, more low)
- Use Kalman filter to preprocess (smooth) input samples. (check filterpy)
- add tianshou to pyproject.toml

**Beta v1**
- Add asyncio for alpaca data collection.
- Include preprocessing scripts for alpaca raw data format.
- Since the Data Format is O, C, H, L, V, Nt, VAvg; we have to have a decision boundary from our model, like, if the Model predicts VH for a stock, we buy immediately with more quantity; like, for VL, if we were holding any then we try to sell, if not then we do nothing; NOTE: this should be more concrete, as this decides the profit/loss. OR, we can use RL to fine tune our pretrained model on stock trend prediction, like, VH, H, L, VL; and try to predict even the buy/sell action, its volume etc...

**Beta v2**
- During training, try to add a mask randomly, like, 2 or 3 consecutive timesteps input, likewise add 2 or 3 or more masks per sample, to add complexity into input...
