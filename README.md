# Classifying-Red-Wine-Quality
![RedWine Image](https://assets.bonappetit.com/photos/5c8940dc92041125f06c3b63/16:9/w_2560%2Cc_limit/Basically-Red-Wine-02.jpg)

## Objective & Approach
This project aims to identify, build and tune a classification models to classify quality red wine samples based on their characteristics.
In order to achieve this, we will use a combination of red wine domain knowledge, such as their chemical properties and taste profiles, and statistical learning techniques, namely classification algorithms (Linear, Non-Linear, Tree-Based Algorithms etc). We also attempted to use statisitcal hypotheses testing as a means of determining and validating key characteristics of good quality red wines. With regards to this, we decided on permutation importance as our key test statistic due to its generalisability and usefulness in testing Tree Based Models.

We will also look towards tapping into deep learning methods, such as *neural network classifiers*, to further improve on the efficacy of our classification endeavours.

## Feature Selection
Due to the highly numerical nature of our features and our binary classification outcome, we decided on ANOVA as our feature selection method. This is as ANOVA's F-Score will allow us to determine the '*separability*' of the feature's data when grouped by the target class (i.e we select features with *High Between Group Variability* vs. *Within Group Variability* or distinctly different means)

## Classification Methods Used (to be deployed)
1. Logistic Regression (Baseline)
2. SVM (Linear & Gaussian RBF kernels)
3. Decision Trees, Randomf Forests & Gradient Boosted Trees
4. L1 & L2 regularisation 
5. NN Binary Classifier (*TBD*)

## Model Evaluation Approach 
1. Macro Averaged F1 Score (*Due to underlying Target Class imbalance*)
2. Precision-Recall Curve
3. ROCAUC (*Biased to dominant class*)
