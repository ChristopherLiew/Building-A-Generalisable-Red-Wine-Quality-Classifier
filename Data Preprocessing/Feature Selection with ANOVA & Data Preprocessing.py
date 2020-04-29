import pandas as pd
from sklearn.feature_selection import f_classif
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler

# Load the dataset
wine_df = pd.read_csv("red_wine_quality.csv")

# Create Binary Target Variable
wine_df['good_quality'] = wine_df['quality'] >= 7
target = wine_df['good_quality'].astype(int)

# Create interaction terms
features_df = wine_df.iloc[:, :-2]
scaler = MinMaxScaler()
features_df_scaled = pd.DataFrame(scaler.fit_transform(features_df), columns=features_df.columns)
features_df_scaled['total_acidity'] = features_df_scaled['volatile acidity'] + features_df_scaled['fixed acidity']
features_df_scaled['pH_sulphate'] = features_df_scaled['pH']*features_df_scaled['sulphates']

# Feature Selection with ANOVA
anova_results = pd.DataFrame(f_classif(features_df_scaled, target), columns=features_df_scaled.columns)
p_vals = anova_results.iloc[1,:]
selected_features = (p_vals <= 0.05)
selected_features = selected_features[selected_features == True]
feat = p_vals[p_vals <= 0.05]
feat.columns = ["Selected Features", "P-Value"]

# Final unscaled dataset
final_df = wine_df.iloc[:, :-2].copy()
final_df['total_acidity'] = final_df['volatile acidity'] + final_df['fixed acidity']
final_df['pH_sulphate'] = final_df['pH']*final_df['sulphates']
final_df = final_df.loc[:, selected_features.index]
final_df.head()

# Train-Test data
X_test, X_train, y_test, y_train = train_test_split(final_df, target, test_size=0.25, random_state=0)

# Scale the training and testing sets independently using Robust Sclaer to mitigate the effect of Outliers
rob_scaler = RobustScaler()
X_test_sc = pd.DataFrame(rob_scaler.fit_transform(X_test), columns=X_test.columns)
X_train_sc = pd.DataFrame(rob_scaler.fit_transform(X_train), columns=X_train.columns)

# Row-wise Standard Scaler to improve class separation in feature space
std_scaler = StandardScaler()
X_test_sc = pd.DataFrame(std_scaler.fit_transform(X_test_sc), columns=X_test_sc.columns)
X_train_sc = pd.DataFrame(std_scaler.fit_transform(X_train_sc), columns=X_train_sc.columns)