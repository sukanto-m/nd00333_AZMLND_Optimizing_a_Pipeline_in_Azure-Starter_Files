# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary
In this project, the UCI Bank Marketing dataset is used to predict which of the bank's clients will open a term deposit with them. So, this is a binary classification task which requires predicting 'yes' or 'no'.

For the classification, two approaches have been used:
 * Scikit-learn Logistic Regression using a custom coded model where the hyperparameters are fine tuned using AzureML's Hyperdrive, which is able to achieve an accuracy of 90.7%
 * Azure AutoML (the 'no code/low code' solution), which uses a variety of machine learning models to arrive at the best performing acccuracy of 91.8%, tad better than the Hyperdrive method. 

## Scikit-learn Pipeline
The Scikit-learn pipeline is designed in the training script 'train.py' where the dataset is downloaded in tabular format using the TabularDatasetFacory class and then goes through the preprocessing steps, which include a 'clean_data' function to clean it and featurisation with one-hot encoding. The data is split into training and test sets (80:20 ratio).

Next, the hyperparameters of the Logistic Regression model are fine tuned, viz:
  * 'C' - inverse of regularisation strength, with positive floats in a range of 0.001 to 1000. Smaller values specify stronger regularisation, as per the [Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) documentation in SKLearn.
  * 'max-iter' - the maximum number of iterations taken by the solver to converge, which I chose in a range of 50-300.  

**What are the benefits of the parameter sampler you chose?**

I went with RandomParameterSampling, since it allows for early stopping early stopping like the bandit policy chosen here. After each training run, the model is tested with the test set, its accuracy logged and the model is saved. The model with the best accuracy metric is selected. 

**What are the benefits of the early stopping policy you chose?**

The Bandit policy is used which is based on a slack factor. If the accuracy of the current run isnt within the slack amount of the best run, the current run is terminated, allowing for some saving on comptational time/compute usage.   

## AutoML

The AutoML uses a variety of models including classification, regressiona and others for training, with a timeout criterion to save on usage cost. Among these models, the Voting Ensemble performs the best.  As its name suggests, this ensemble is an ML model that combines the results of a collection of algorithms. The Voting Ensemble here consists of 7 individual models, of which 4 are XGBoost classifiers, and 1 each of a LightGBM, SGD and Logistic Regression.

![image](https://github.com/sukanto-m/nd00333_AZMLND_Optimizing_a_Pipeline_in_Azure-Starter_Files/blob/master/Screenshot%202021-08-10%20at%2011.59.28%20AM.png)
![image](https://github.com/sukanto-m/nd00333_AZMLND_Optimizing_a_Pipeline_in_Azure-Starter_Files/blob/master/Screenshot%202021-08-10%20at%2011.06.24%20AM.png)



## Pipeline comparison

The AutoML performs better than the Hyperdrive, even if the improvement in accuracy is only about 1%-2%. This is simply because AutoML has a number of models at its disposal as compared to Hyperdrive. It also reveals an imbalance in the dataset (data guardrails as shown in the notebook) which offer scope for improvement as detailed in the next section.

![image](https://github.com/sukanto-m/nd00333_AZMLND_Optimizing_a_Pipeline_in_Azure-Starter_Files/blob/master/Screenshot%202021-08-10%20at%2011.06.24%20AM.png)
![image](https://github.com/sukanto-m/nd00333_AZMLND_Optimizing_a_Pipeline_in_Azure-Starter_Files/blob/master/Screenshot%202021-08-10%20at%2011.59.28%20AM.png)

## Future work

Some possible areas of improvement:

* As the AutoML comes up with a best performing ensemble, an XGBoost could be reverse engineered to fit the Hyperdrive - instead of using a Logistic Regression if one has to use only one model to make a prediction.
* The AutoML reveals an class imbalance in the data in terms of bias whichmay or may not result in overfitting. For almost all ML models, their performance is only as good as the quality of the data, while there is no free lunch  either (high accuracy usually comes at a cost). So this should be addressed in terms of better quality data collection/input.
* Accuracy is not always the best primary metric. A model may still fail to make good predictions despite having a high accuracy, so using a different metric may not be a bad idea.

## Proof of cluster clean up
![image](https://github.com/sukanto-m/nd00333_AZMLND_Optimizing_a_Pipeline_in_Azure-Starter_Files/blob/master/Screenshot%202021-08-10%20at%2012.02.01%20PM.png)

