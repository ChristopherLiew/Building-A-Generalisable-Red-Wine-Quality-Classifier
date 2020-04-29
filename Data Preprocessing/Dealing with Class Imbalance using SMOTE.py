# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# Load and Prepare the Red Wine dataset
wine_df = pd.read_csv("red_wine_quality.csv")
wine_df["good_quality"] = wine_df["quality"] >= 7
# Create the Target Variable (Dependent Var.)
target = wine_df.loc[:, 'good_quality']
target = target.astype(int)  # 1 = good quality & 0 = other quality
# Features only Dataframe (Independent Var.)
feat_df = wine_df.drop(columns=['quality', 'good_quality'])

# Current imbalance
compare_df = wine_df.copy()
compare_df.drop(columns=['quality'])
compare_df.loc[:, 'good_quality'].replace({True: "Good Quality", False: "Other Quality"}, inplace=True)
print("Other Quality Count: " + str(sum(compare_df.good_quality == "Other Quality")))
print("Good Quality Count: " + str(sum(compare_df.good_quality == "Good Quality")))

# Histogram of Imbalanced Classes
fig, ax = plt.subplots(1)
plt.hist(compare_df.good_quality, align='mid', rwidth=5, bins=3, alpha=0.7, color='red')
plt.grid(axis='y', alpha=0.75)
plt.ylabel('No. of Red Wine Samples')
plt.xlabel('Red Wine Quality')
plt.title('Red Wine Sample Count by Target Class')
ax.set_xticklabels([])
plt.annotate("Other Quality: \n      86.4%", (0.06, 800))
plt.annotate("Good Quality: \n      13.6%", (0.73, 60))

# Create Training & Testing Datasets
X_train, X_test, y_train, y_test = train_test_split(feat_df, target, test_size=0.25, random_state=0)

# Apply SMOTE (Oversampling of minority class) onto Training Data
sm = SMOTE(sampling_strategy="minority")
X_train_sm, y_train_sm = sm.fit_sample(X_train, y_train)

# Balanced Dataset
a = pd.DataFrame(X_train_sm, columns=feat_df.columns)
b = pd.DataFrame(y_train_sm)
b.columns = ["good_quality"]
new_df = pd.concat([a, b], axis=1)
new_df

# Current Balance
new_df.loc[:, 'good_quality'].replace({1: "Good Quality", 0: "Other Quality"}, inplace =True)
print("Other Quality Count: " + str(sum(new_df.good_quality == "Other Quality")))
print("Good Quality Count: " + str(sum(new_df.good_quality == "Good Quality")))

# Histogram of Balanced Classes
fig, ax = plt.subplots(1)
plt.hist(new_df.good_quality, align='mid', rwidth=5, bins=3, alpha=0.7, color='yellow')
plt.grid(axis='y', alpha=0.75)
plt.ylabel('No. of Red Wine Samples')
plt.xlabel('Red Wine Quality')
plt.title('Red Wine Sample Count by Target Class')
ax.set_xticklabels([])
plt.annotate("Other Quality: \n      50.0%", (0.06, 500))
plt.annotate("Good Quality: \n      50.0%", (0.73, 500))
