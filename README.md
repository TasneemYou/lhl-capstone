# Transformers Faults Reasoner

Transformers are used in the distribution of electricity. It is basically a stationary device that moves electric energy from one circuit to another. So, Transformers are critical because they boost voltages of electricity, so it can travel long distances efficiently. As electricity nears users, transformers reduce voltages so it is suitable for consumption.

![image](https://user-images.githubusercontent.com/34404363/155343788-044dc0f2-3d4b-413b-9fb0-22ca1cb2841b.png)


## Problem Statement

When a fault occurs in a transformer, diagnostic testing is performed to determine what caused the failure.

The transformer is removed, sent for repair, and its fault is determined manually by studying the pictures taken immediately after the load and other condition parameters, also sometimes there is no obvious external damage and diagnostic testing is needed to determine the weaknesses.

## Project Goal

The goal of this project is to build a model that reduces the manual labor of going through all the parameters and predict the outcome with good accuracy, thus saving time, cost, and efforts that go in diagnostic testing. 

## Input/Output Parameters

Transformers faults can be generalized into these categories as shown in the diagram below, this is my target variable. 
![image](https://user-images.githubusercontent.com/34404363/155344356-50300108-882d-4d66-958b-7b891155dacc.png)

There are various categorical, numerical, and ordinal features present in the dataset, such as:
1. Categorical: Cluster, Region, Observations, Make, Work Details, Repair Status, etc.
2. Numerical: Maximum allowed loading per phase, Capacity, Rated Power, Loading on each of the RED, YELLOW, and BLUE phases, etc.
3. Ordinal: Oil quality, Silica gel condition, Tap status, etc.

## Tasks

# 1. Data Extraction: (Pandas)
My project goes through the stages of gathering the data for different years using pandas after which I preprocessed it for analysis.

# 2. Data Cleaning: (Pandas, Numpy, Regex)
Since data was from real world, the data cleaning included a lot of work which was the major challenge I faced during this project. Here, I overcame this hurdle by the extensive use of regex for pattern matching to generalize all features.

# 3. Modeling: (Pipelines, Sklearn, Grid Search)
I applied different multi-class learners to the transformed data in the modeling step using various parameters.

# 4. Cloud Deployment: (AWS EC2, Flask)
Finally, the project was deployed on cloud using the Amazon Web Services EC2 console and Flask.

These steps were done in an iterative fashion where I went back and forth between data cleaning, transformation, and modeling.

## Results

Out of the several models I tried I am comparing my accuracy results with a baseline model which predicts the most frequent class of Loading Issue regardless of the input features. My analysis showed that the best performance was achieved by XGBoost with an accuracy percentage of 76.67%.

With current accuracy, the XGBoost model for predicting a transformer’s fault can be used as the first step to find the possible cause of failure, which can then be confirmed (or rejected) by further diagnostics, for example, analysing the picture of transformer taken when its link blown out.

![image](https://user-images.githubusercontent.com/34404363/155346105-b82efccb-be9a-4c7c-926f-620209d5a5a6.png)

## Future Scope

This kind of supervised learning can be applied to different electrical components deployed to supply, transfer, and use electric power.
In my opinion, the accuracy achieved was very good compared to the baseline however in future neural network can be developed using the available features and images of faulty transformers and other components taken at the time of removal to achieve higher prediction accuracy, thus taking away the need for additional tests.

![image](https://user-images.githubusercontent.com/34404363/155346533-2e4a528e-246c-44ae-974f-cef6e7b9e795.png)



