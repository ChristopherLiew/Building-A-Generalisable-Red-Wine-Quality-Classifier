# Building A Generalisable Red Wine Quality Classifier
![RedWine Image](https://assets.bonappetit.com/photos/5c8940dc92041125f06c3b63/16:9/w_2560%2Cc_limit/Basically-Red-Wine-02.jpg)

## In the Pipeline
1. Deploying our models on a SMOTE-ed *red wine quality* dataset

## Objective & Approach
This project aims to identify, build and tune generalisable classification models to classify quality red wine samples based on their characteristics. In order to achieve this, we will use a combination of red wine domain knowledge, such as their chemical properties and taste profiles, and statistical learning techniques, namely classification algorithms (Linear, Non-Linear, Tree-Based Algorithms etc). 

We also attempted to use statisitcal hypotheses testing as a means of determining and validating key characteristics of good quality red wines. With regards to this, we decided on **permutation importance** as our key test statistic due to its generalisability and usefulness in testing Tree Based Models.

We will also look towards tapping into deep learning methods, such as *neural network classifiers*, to further improve on the efficacy of our classification endeavours.

## Feature Selection & Feature Scaling
Due to the highly numerical nature of our features and our binary classification outcome, we decided on ANOVA as our feature selection method. This is as ANOVA's F-Score will allow us to determine the '*separability*' of the feature's data when grouped by the target class (i.e we select features with *High Between Group Variability* vs. *Within Group Variability* or distinctly different means)

We applied feature scaling using the *Robust Scaler*, due to a significant number of observed outliers in the dataset. Of which, we have decided to keep in order to minimise variance and boost generalisability. However, feature scaling was not necessary and thus not applied to the training sets of models which did not depend on the computation of *euclidean distances* for optimisation (i.e. Probabilistic Naive Bayes, CART algorithms, etc).

## Classification Methods Used
1. Logistic Regression (Baseline)
2. SVM - Linear & Gaussian RBF kernels
3. Gaussian Naive Bayes
4. Decision Trees, Random Forests & Gradient Boosted Trees
5. MLP Binary Classifier

## Model Evaluation Approach 
1. Macro Averaged F1 Score (*Due to underlying Target Class imbalance*)
2. Precision-Recall Curve
3. AUROC (*Biased to dominant class*)

## Tools Used
We largely used Jupyter (CoLab) notebooks due to their versatility and for presentational purposes. However, hyperparameter optimisation was done locally on our computers to speed things up. The code for the project was written in Python 3.7.

### Please refer to our report for our analyses, insights & conclusion 
