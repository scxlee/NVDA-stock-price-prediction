# NVDA-stock-price-prediction
Capstone project for an AI/ML course

#### Contents
1. Introduction
2. Project files
3. Data collection
4. Exploratory data analysis (EDA)
5. Feature processing
6. Model selection
7. Model evaluation
8. Other considerations

#### 1. Introduction

This capstone project seeks to predict the stock price of Nvidia (NVDA) using three machine learning models. Many factors influence the price of a given stock. This project is focused on quantitative factors where data that is publicly available. The omission of qualitative factors is due to challenges accessing the data over the requisite time period. 

#### 2. Project files

* Jupyter notebooks and .csv files
	* qtdata.ipynb - to compile data from APIs and downloaded csv files and export to compiled.csv
	* eda.ipynb - to perform EDA on compiled.csv
	* preproc.ipynb - to perform data preprocessing on compiled.csv and export to processed.csv
	* models_reg.ipynb - to train and evaluate models on processed.csv
	* newdata.ipynb -  to compile new / forward-looking data from APIs and export to newdata.csv
	* preproc_newdata.ipynb - to perform data preprocessing on newdata.csv and export to processed_new.csv
	* pastdata.ipynb - to compile past / backward-looking data from APIs and export to pastdata.csv
	* preproc_pastdata.ipynb - to perform data preprocessing on pastdata.csv and export to processed_past.csv

#### 3. Data collection

* Qualitative factors
	* The original scope of this project was to include qualitative factors such as news media, social media, and company reported news.
	* Sentiment analysis would have been performed on this data to determine daily sentiment scores that could be used to predict stock price. 
	* Efforts were made to gather news data from news APIs (Benzinga, News API, and Alpha Vantage) but the APIs imposed limits and it was only possible to collect news data for only recent months. This was not satisfactory as the time period required was 5 years (to achieve a dataset of at least 1000 rows based on daily price).
	* Efforts were made to scrape social media data (Nvidia's twitter and Wallstreetbets Reddit) but the available tools were either out of date or too technical / time consuming. 
	* Effort was made to scrape Nvidia's press releases (on Nvidia's news page) using beautiful soup but due to the time consuming nature, it was abandoned. For the year 2024 alone (and now only May), there were 24 web pages so going back over 5 years would involve too much time. 
	* There were considerations to manually copy and paste text from Nvidia's quarterly presentation but this would be time consuming and the idea was abandoned. 
	* There were also considerations to incorporate news on specific keywords like 'AI' and 'tech' but given the above experience, the idea was abandoned.
	* Based on the above efforts, the project scope was narrowed to focus on quantitative factors / data.
* Quantitative factors
	* An initial search was performed to locate existing datasets (Kaggle and Hugging Face) but what was available related to limited price data. 
	* The decision was made to create a dataset from publicly available sources.
	* Efforts were made to source data for a variety of factors but there were a number of challenges:
		* Data hidden behind paywalls
		* Data limits for free APIs / access (download to csv)
	* The ideal sources were Nasdaq and Benzinga given their financial focus and breadth of data but they required subscriptions.
	* Data was sourced from:
		* Yahoo Finance
		* Alpha Vantage
		* Others: Barchart, Macrotrends, etc.
	* For some factors such as earnings surprise and earnings surprise, they were available on a quarterly basis, presenting a mismatch with stock price data.
	* There were considerations to incorporate technical indicators but an excess of time had been spent sourcing data so the decision was made to focus on what had been collected.
* Selected factors
	* These were driven by data availability and factors that are commonly understood to influence a given stock's price. 
	* For price factors, the price of Nvidia's key competitors, suppliers, and customers were included, as well as the broader market and sector:
		* Competitors: AMD, Intel, Samsung
		* Suppliers: Micron, SK Hynix, TSMC  
		* Customers: Amazon, Google, Microsoft
		* Market/sector: Nasdaq index, S&P500 index, PHLX Semiconductor index
	* In reality Nvidia has many competitors, suppliers and customers but the above were a select few chosen to avoid model complexity.
	* Price change factors were derived from price factors after forward filling nulls in price factors. 
		* The decision was taken to forward fill nulls as they were 3-5 percent of the data.
		* Forward filling nulls was deemed most appropriate compared to other filling methods as the price from one day to the next does not typically change significantly. 

#### 4. Exploratory data analysis (EDA)

* The dataset consists of 40 columns, including a Date column.
* As explained earlier, the dataset contains only quantitative data / numerical variables.
* Price-related variables
	* Current price of NVDA and its competitors, suppliers, and customers ([ticker])
	* Change in price of the same ([ticker]_chg)
 * NVDA-focused variables
 	* Analyst Action (upgrades / downgrades expressed as +1 or -1 and aggregated for each date)
	* Insider sales (expressed in volume)
	* Surprise (the difference between consensus EPS vs. actual EPS)
	* Surprise Perc (the difference expressed as a percentage of consensus EPS)
	* Short (expressed in volume)
	* Short Perc (expressed as a percentage of total Volume)
	* PE Ratio (price/earnings ratio)
	* Volume
	* 50D SMA (50-day simple moving average)
	* 200D SMA (200-day simple moving average)
* Other variables
	* Interest Rate (Fed Funds Rate)
* Findings
	* Nvidia's price trend has broadly followed those of its stakeholders/market/sector but it diverged at the start of 2023 where the price rose significantly, leaving its stakeholders/market/sector behind
	* Significant proportion of nulls for Analyst Action, Insider Sales, Surprise, Surprise Perc but these can be filled with 0 as nulls represent absence of activity on those dates
	* Significant proportion of nulls for Short, Short Perc, PE Ratio simply due to limited data so these can be dropped
	* Binomial distributions: price variables and PE Ratio
	* Normal distributions: price change variables, Surprise, Short Perc
	* Skewed distributions: Nvidia price, Insider Sales, Surprise Perc, Short, Volume, Interest Rate, 50D SMA, 200D SMA
	* Bernoulli distribution: Analyst Action
	* Vastly different ranges for variables so data needs to be scaled in preprocessing
	* Outliers for several variables: SK Hynix, AMD, Micron, Nvidia, price change variables, Insider Sales, Surprise, Surprise Perc, Short, Short Perc, Volume, 50D SMA, 200D SMA
	* Outliers need to be Winsorised in preprocessing
	* Correlations
		* Strong correlation between Nvidia's price and the price (but not price change) of its stakeholders/market/sector so price change variables can be dropped
		* Low correlation between Nvidia's price with variables one would expect to influence price (Analyst Action, Insider Sales, Surprise, Surprise Perc, Volume) so these can be dropped
		* Strong correlation between Nvidia's price and Interest Rate, 50D SMA, 200D SMA
		* Correlations hold when using previous day values rather than current day values
		* Price variables are highly correlated to each other

#### 5. Feature processing

* Feature processing was led by EDA findings and include:
	* Dropping features with significant nulls and weak relationships with Nvidia's price
	* Creating new features of previous day values
	* Dropping features of current day values
	* Forward filling nulls for 50D SMA and 200D SMA - applying the same method as before for price variables
	* Performing Winsorisation on features with outliers
		* This is preferred to dropping outliers and losing data points
		* As outliers fall beyond the maximum value, the lower percentile is set to 0 and the upper percentile to 97
		* 97th percentile is chosen as the upper limit to losing too much information from price data
	* Scaling all features to address differing ranges 
		* MinMax Scaling is preferred as Standard Scaling and Standardisation are more appropriate for Normal distributions (unlike the variables retained for model training)
	* Dropping residual nulls from the creation of previous day features
	* Setting the Date column as the index
	
#### 6. Model selection

* As this is a time series problem with only numerical features, regression models are appropriate.
	* Multiple Linear Regression (MLR)was chosen as the baseline model for its simplicity and easy interpretability.
	* Random Forest Regression (RF) was chosen for its robustness to outliers and its ability to capture non-linear relationships. 
	* XGBoost Regression (XGB) was chosen for its potential to achieve higher accuracy than RF. 
* Long Short Term Memory (LSTM) was considered since it is a deep learning model rather than machine learning model and its common usage in projects predicting stock price. 
	* Attempts were made with LSTM but errors were encountered.
	* In the interest of time, the decision was made to focus on regression models.
	* If time permitted, LSTM would be explored. 
* As RF and XGB involve multiple hyperparameters, GridSearch was used to identify the optimum model.
	* GridSearch was chosen over RandomSearch as it is guaranteed to find the best configuration within the defined grid and due to the small search space involved.
	* Mean Squared Error was chosen as the objective for GridSearch due to its suitability for regression problems.
	
#### 7. Model evaluation

* MLR took the shortest time to train whereas RF and XGB took longer times due to GridSearch trying to identify the optimum model. 
* To evaluate the models
	* Each model predicted on the entire dataset to arrive at predicted values that could be plotted against actual values in price charts.
	* Scatter plots were created for predicted values of only the test set (not the entire dataset). 
	* Evaluation metrics were calculated for predicted values of only the test set (not the entire dataset). 
	* The metrics chosen were:
		* Mean squared error (MSE) - chosen due to its common usage for regression problems
		* Root mean squared error (RMSE) - same reason as above
		* Mean absolute error (MAE) - chosen as it is less sensitive to outliers (present in the dataset)
		* R-squared (R2) - chosen as it illuminates the relationship between features and the target
* Evaluation overview
	* All models have very low MSE values (~0.0001), indicating a small average difference between predicted and actual values.
	* Similar to MSE, RMSE values are very low (~0.01), suggesting small average errors in terms of the actual scale of the values.
	* The errors in predicted values are very small, with MAE values ~0.006.
	* All models achieve very high R2 values (~0.998), indicating that they explain a high proportion of the variance in stock prices based on the chosen features.
	* Based on the scatter plots, RF and XGB model predictions are less accurate towards the high end of Nvidia's price.  
* Evaluation comparison
	* MLR has the lowest MSE, RMSE, MAE compared to RF and XGB, offering marginally higher average accuracy.
	* MLR has the highest R2 compared to RF and XGB, but again the difference is marginal.
* Further evaluation
	* All models performed well with MLR having a slight edge (in terms of predicting on the test set).
	* To ensure generalisability and avoid overfitting concerns, there was further evaluation of the models' predictions on unseen data.
	* Looking forwards, there were six days / data points (after the end date of the original dataset) that could be used for predictions.
	* Looking backwards, the same five-year time period was chosen for predictions.
	* MLR clearly outperforms RF and XGB from just the price charts and scatter plots.
	* Using the same metrics as before, MLR outperforms again but by a big leap.
		* The R2 of RF and XGB drops to <0.15, with the latter achieving a negative value.
		* This negative R2 value means that XGB is worse than using the mean of Nvidia's price.
* Conclusion
	* For this problem, MLR excels.
		* It is the top performer across evaluation scores.
		* It has decent generalisability unlike RF and XGB.
		* It is simple and easy to interpret.
		* It takes little time / compute to train.
	
#### 8. Other considerations

* It was surprising to achieve such high performance with the models, especially MLR.
	* MLR is only a machine learning model, not even a deep learning model.
	* Relatively simple features were used - they were previous day prices of Nvidia's competitors, suppliers, and customers, with the exception of interest rate, 50-day simple moving average and 200-day simple moving average.
	* Features did not include common factors considered to contribute to a stock's price.
	* Features did not include sentiment / qualitative factors.
* As data was Winsorised and scaled before training the models, new data presented to the models for prediction also need to be Winsorised and scaled accordingly.
* Limitations
	* Models are time dependent. If patterns in the data change, they will no longer produce reliable predictions, as observed in the predictions on the previous 5 years. 
	* Price features being highly correlated, presenting the problem of multicollinearity that can affect model performance, particularly for MLR.
	* Regression models are dependent on correlation to make predictions so a change in the underlying causal relationships in new data can lead to unreliable predictions.
* Further development
	* For this stock price prediction problem, there could be exploration with the following that time did not permit for this project:
		* Address multicollinearity through feature selection and other solutions like Ridge Regression and Principal Component Analysis
		* Incorporate technical indicators as features e.g. Relative Strength Index and On Balance Volume *these are readily available via Alpha Vantage's API
		* Incorporate sentiment scores of news data and social media data as features
		* Incorporate features that were dropped / omitted due to lack of access rather than availability e.g. Short, Short Perc, Analyst Price Targets
		* Use price change rather than price as the target
		* Use deep learning models such as LSTM, Neural Basis Additive and Multiplicative Time Series (NBEATS) and Neighbourhood Hitting Transformer (NHITS)
		* Experiment training with different time periods e.g. pre 2023 and 2023 onwards
