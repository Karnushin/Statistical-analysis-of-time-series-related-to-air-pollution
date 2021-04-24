# Statistical-analysis-of-time-series-related-to-air-pollution

This repo includes analysis of air pollutation based on measurements from 1983 to 2017 such elements as TSP, SO2, SO4, HG, PB, CD. 

**First part** includes some preparation of data using division by seasons on need, fitting distributions for all series and its seasonal components, checking different hypotheses such as goodness of fit and like homogenity with some size of window and other similar things which was needed. Everything was made after deleting missed values. It's made in 2019

**Second part** is more for prediction series. It includes choosing the best series to analyze based on minimum missed values. Here's almost no checking hypotheses but different models to forecast used: Prophet (Facebook), Holt-Winters(exponential smoothing), SARIMA, LSTM, BiLSTM. It's made in 2020

**Third part** is almost union previous ones where some extra things were added. Mainly it contains:

  - statistics of series;
  - detecting anomalies using Box Plot, Z-score, Isolation Forest and Local Outlier Factor;
  - Classification of anomalies using Logistic Regression, Random Forest, Lightgbm
  - Forecasting series using Random Forest, Lightgbm, Ridge Regression, (Bi)LSTM
