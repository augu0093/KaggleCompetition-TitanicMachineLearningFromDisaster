# SciKit-Learn classification and Bayesian Optimization
By August Semrau Andersen.

This project is an entry in to the Kaggle competition 'Titanic: Machine Learning from Disaster'.  
The goal of the competition is to predict which passengers of the Titanic survived the infamous disaster.

The intent with the project is to display proficiency in using the SciKit-Learn package for a classification task.  
Further ability in tuning of models using Bayesian Optimization is also displayed.


### Scripts
The following scripts are used for completing the competition.
 
1. dataLoader.py which loads .csv data and uses sklearn for preprocessing. 
2. models.py contains some non-tuned sklearn classification models, some tuned models and an XGBoost classifier.
3. predictions.py is used for printing predictions to .csv format for entry in Kaggle-competition.
4. bayesianOptimization.py has been used for optimizing some of the below mentioned classification-models.


### Models and their Accuracy
Below is a short description of each model used and which accuracy they yielded.

- Logistic Regression, no tuning: 0.737 accuracy.

- Naive Bayes, no tuning: 0.718 accuracy.

- Stochastic Gradient Descent, squared_loss: 0.373 accuracy.
This is obviously faulty, worst performance is 0.5 due to the nature of binary classification.
The explanation in this case is that SDG guesses a lot of 1'es (alive) while there are an overload of 0'es (diseased).

- K-Nearest-Neighbors, n=10: 0.634 accuracy

- Decision Tree, no tuning: 0.722 accuracy.

- Random Forest, no tuning: 0.766 accuracy.  
Random Forest, BO tuned: 0.778 accuracy. <- Best.

- Support Vector Machine, no tuning: 0.773 accuracy.

- XGBoost, no tuning: 0.775 accuracy.


Final rank on Kaggle: 7770, top 30 %.
