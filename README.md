# Improved-detection-of-fraud-cases-for-e-commerce-and-bank-transactions
For this project, we followed the following workflow to reach our data analysis goal.  

# 1.	Define the Problem
We are working with Adey Innovations to improve the detection of fraud cases for e-commerce and bank transactions. We will try to create strong fraud detection models than are capable of handling both e-commerce and banking transactions. It also uses geo-location analysis and transaction pattern recognition to improve detection so we can pinpoint fraudulent activities accurately to make the transaction more secure. We have two datasets, one for each transaction. To reach our objectives, we start cleaning the data and preparing it for analysis and prediction. After that we build different machine learning models so that we can understand and predict when and where fraudulent activities occur. Finally, we will leverage model interpretability to understand how each model made the decision it made.  

# 2.	Data Cleaning and Preparation
As mentioned above, we have two data sets that we can use to build our models. We start with cleaning and preparing it for further analysis. This includes:
	Checking for outliers and removing outliers that might affect data accuracy.
	Checking for missing values and removing the most detrimental ones or imputing those that can be imputing using mean or median.
	Encoding categorical values so that they can be used in machine learning models.
	Scaling the data.
	Converting the data into the proper data format like date.

# 3.	Exploratory Data Analysis
After initial preparation of the data, we conduct exploratory data analysis to gain deeper insights to the data. It will help us form initial analysis. This includes:
•	Summarizing key statistics: We can see values such as mean, median, standard deviation to gain insight to the data.
•	Relationship distribution: plot histograms and bar charts to understand the data distribution better. 
•	Merging the ecommerce datasets with the IpAddress_to_Country dataset for  Geolocation Analysis
•	Feature engineering to include important features that are useful our analysis. Create new features such as transaction frequency and velocity (this one is specifically for the ecommerce data) so as to have a more accurate model. 
•	Performing univariate and bivariate analysis to understand how each variable behaves and how it is related to the other variables in the dataset.
 
Figure: Distribution of different types of browsers in the ecommerce dataset.

**Handling class imbalance** 
One specific behavior of the fraudulent data set is how imbalanced it is. It is not uncommon to have only a very small amount of the transaction data be fraudulent causing our data to have an imbalance. We can see this from the distribution of legitimate/fraudulent cases in our dataset. 
Dataset	Percentage of legitimate transactions	Percentage of fraudulent transactions
E-commerce transaction	90.49%	9.5%
Banking transactions	99.8%	0.16%

To address this imbalance, we can use several types of statistical techniques. We can use resampling techniques such as oversampling or undersampling. In undersampling, we reduce the number of majority class samples to balance it with the minority class. This makes us lose data which makes it difficult for the model to generalize well to new data. In oversampling, we increase the number of the minority class using random oversampling – where we duplicate minority class samples, or Synthetic Minority Over-sampling Techniques (SMOTE) – generate minority samples by interpolating between the existing ones.  SMOTE works by taking samples of feature space for each minority class instance and its nearest neighbors and creates new examples by interpolating between a minority class sample and one of its neighbors by adding a random proportion of the difference between feature vectors of 2 samples. It is applied only to the training set. 
For our ecommerce dataset, we used SMOTE to address the imbalance. We used SMOTE library from imblearn. After using this technique, our data had a 50-50 distribution. Our banking dataset, however, had only 0.16% fraudulent cases which meant we couldn’t use SMOTE alone as we only have a few examples to interpolate on. In this case, we combine SMOTE with Edited Nearest Neighbors (SMOTEEN). This technique generates synthetic minority class samples using interpolation then cleans the dataset by removing noisy samples to sharpen class boundaries. 

# 4.	Model Building 
The goal is creating a model that can detect fraud in transactions. So after cleaning and preparing the data, we proceed to build machine learning models. We start with logistic regression for both datasets. We scale the data and feed it to the model. We then build two other models using Random Forest and XGBoost and compare the outcomes. 
To evaluate the models we are using AUC-PR together with F1 Score and confusion matrices.  We can see the outcomes for AUC-PR, which quantifies how well a model distinguishes between positive and negative classes (find as many positives while minimizing false positives), in the table below.  
 
We can see that the ensemble models have a superior performance when compared to the logistic regression. The random forest model does slightly better than the XGBoost model. We also see that while the model does well on the training set, they don’t do as well on the test data this indicates over fitting. Tuning the various hyper parameters in the model and even more targeted feature engineering can give us better models.  

# 5.	Model Interpretability
After building the machine learning models, we proceed to use tools to understand how the models made their decisions. Model interpretability is to the ability to understand and explain how a machine learning model makes its predictions or decisions. It provides insight into why a model produced a certain output based on its input features. It is essential to build trust, detect bias, debug models, and meet regulatory requirements, especially in high-stakes domains like healthcare, finance, and law. A popular method is SHapley Additive exPlanations (SHAP). It is a method based on cooperative game theory that explains the output of any machine learning model by assigning each feature an importance value for a particular prediction. It computes Shapley values that fairly distribute the contribution of each feature to the model's output, considering all possible combinations of features. It provides local explanations (why a specific prediction was made) and global explanations (overall feature importance).   
For the ecommerce transactions, we can see which features have more importance in deciding if a transaction is fraudulent or not. We see that time since sign up is a very important factor as usually fraudulent activities happen immediately after signup. We can also see that the source through which the user came to the site and the browser used to make the transaction are important factors in finding fraudulent transactions. These inputs will be invaluable in deciding which transactions are fraudulent in the future. 
 

We also built the SHAP plot for the banking data. While we can’t say which features are the most important because the features are anonymized, we can see V1 is the most important feature.
 

# Summary 
We are working with Adey Innovations to enhance fraud detection for both banking and e-commerce transactions. Our objective is to create strong models that can effectively handle transactions from these two domains. We possess two distinct datasets: one representing banking transactions and the other representing e-commerce transactions. Our initial step is to prepare and clean this data to make it ready for analysis and prediction. To accurately identify fraudulent activities and increase transaction security, we use transaction pattern recognition alongside geo-location analysis.
Following data preparation, we develop several machine learning models aimed at predicting and understanding the timing and location of fraudulent actions. To conclude, we utilize model interpretability methods to clarify the reasoning behind each model’s decisions.


