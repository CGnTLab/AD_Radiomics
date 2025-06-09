
import pandas as pd 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, matthews_corrcoef
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from imblearn.under_sampling import RandomUnderSampler
import joblib
from scipy.stats import randint
import matplotlib.pyplot as plt



##########################################################################
### CN VS AD - Classification Model 1
#########################################################################

# Load your dataset (update the file path)
df = pd.read_csv("/home/sushree/ish_project/data/AD_classification_without_age.csv")
#df.dropna(inplace=True)

# Update mapping for binary classification: 'CN' (1) vs 'AD' (0)
df['Group'] = df['Group'].map({'CN': 1, 'AD': 0})

# Print size of each class
print("Size of CN", (df['Group'] == 1).sum())
print("Size of AD:", (df['Group'] == 0).sum())

# Drop rows related to 'MCI'
df = df[df['Group'].isin([0, 1])]

# Separate features and target variable
X = df.drop(['Image_ID','Group','Subject_ID','VISCODE'], axis=1)
y = df['Group']

# Split data into training and test sets with 80-20 split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
# Print size of training and test sets
print("Size of Training Set:", x_train.shape[0])
print("Size of Test Set:", x_test.shape[0])

# Preprocessing: Centering, Scaling, and KNN-Imputation
scaler = StandardScaler()
imputer = KNNImputer(n_neighbors=5)

x_train_scaled = scaler.fit_transform(imputer.fit_transform(x_train))
x_test_scaled = scaler.transform(imputer.transform(x_test))

# Print number of instances in each class before undersampling
print("Number of instances in each class before Random Undersampling:")
print(y_train.value_counts())

# Random Undersampling for handling class imbalance
undersampler = RandomUnderSampler(random_state=42)
x_train_balanced, y_train_balanced = undersampler.fit_resample(x_train_scaled, y_train)

# Print number of instances in each class after Random Undersampling
print("Number of instances in each class after Random Undersampling:")
print(pd.Series(y_train_balanced).value_counts())

# Hyperparameter Tuning using RandomizedSearchCV
param_dist = {
    'n_estimators': randint(100, 1000),
    'max_features': ['auto', 'sqrt'],
    'max_depth': [10, 20, 30, 40, 50, None],
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 20),
    'bootstrap': [True, False]
}

rf_classifier = RandomForestClassifier(random_state=42)
random_search = RandomizedSearchCV(estimator=rf_classifier, param_distributions=param_dist, n_iter=100, cv=5, verbose=2, random_state=42, n_jobs=-1)
random_search.fit(x_train_balanced, y_train_balanced)

best_rf_classifier = random_search.best_estimator_

# Save the best trained model
joblib.dump(best_rf_classifier, 'best_random_forest_model.pkl')

# Load the saved model
saved_model = joblib.load('best_random_forest_model.pkl')

# Apply the saved model on the test set
y_test_pred = saved_model.predict(x_test_scaled)
y_test_pred_proba = saved_model.predict_proba(x_test_scaled)[:, 1]  # Probabilities for class 1

# Calculate evaluation metrics for the training set
accuracy_train = accuracy_score(y_train, best_rf_classifier.predict(x_train_scaled))
precision_train = precision_score(y_train, best_rf_classifier.predict(x_train_scaled))
recall_train = recall_score(y_train, best_rf_classifier.predict(x_train_scaled))
f1_train = f1_score(y_train, best_rf_classifier.predict(x_train_scaled))
roc_auc_train = roc_auc_score(y_train, best_rf_classifier.predict(x_train_scaled))
mcc_train = matthews_corrcoef(y_train, best_rf_classifier.predict(x_train_scaled))# Calculate evaluation metrics for the test set
accuracy_test = accuracy_score(y_test, y_test_pred)
precision_test = precision_score(y_test, y_test_pred)
recall_test = recall_score(y_test, y_test_pred)
f1_test = f1_score(y_test, y_test_pred)
roc_auc_test = roc_auc_score(y_test, y_test_pred_proba)  # Use y_test_pred_proba for ROC AUC
mcc_test = matthews_corrcoef(y_test, y_test_pred)

# Print metrics for the training set
print("Metrics for Trained Model (Training Set):")
print("Accuracy: {:.2f}%".format(accuracy_train * 100))
print("Precision: {:.2f}".format(precision_train))
print("Recall: {:.2f}".format(recall_train))
print("F1-score: {:.2f}".format(f1_train))
print("ROC AUC: {:.2f}".format(roc_auc_train))
print("MCC: {:.2f}".format(mcc_train))

# Print metrics for the test set
print("\nMetrics for Trained Model (Test Set):")
print("Accuracy: {:.2f}%".format(accuracy_test * 100))
print("Precision: {:.2f}".format(precision_test))
print("Recall: {:.2f}".format(recall_test))
print("F1-score: {:.2f}".format(f1_test))
print("ROC AUC: {:.2f}".format(roc_auc_test))
print("MCC: {:.2f}".format(mcc_test))



#########################################################################
### CN VS AD With AGE - Classification Model 2
#########################################################################


# Load your dataset (update the file path)
df = pd.read_csv("/home/sushree/ish_project/data/AD_classification_with_age.csv")
#df.dropna(inplace=True)

# Update mapping for binary classification: 'CN' (1) vs 'AD' (0)
df['Group'] = df['Group'].map({'CN': 1, 'AD': 0})

# Print size of each class
print("Size of CN", (df['Group'] == 1).sum())
print("Size of AD:", (df['Group'] == 0).sum())

# Drop rows related to 'MCI'
df = df[df['Group'].isin([0, 1])]

# Separate features and target variable
X = df.drop(['Image_ID','Group','Subject_ID','VISCODE'], axis=1)
y = df['Group']

# Split data into training and test sets with 80-20 split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
# Print size of training and test sets
print("Size of Training Set:", x_train.shape[0])
print("Size of Test Set:", x_test.shape[0])

# Preprocessing: Centering, Scaling, and KNN-Imputation
scaler = StandardScaler()
imputer = KNNImputer(n_neighbors=5)

x_train_scaled = scaler.fit_transform(imputer.fit_transform(x_train))
x_test_scaled = scaler.transform(imputer.transform(x_test))

# Print number of instances in each class before undersampling
print("Number of instances in each class before Random Undersampling:")
print(y_train.value_counts())

# Random Undersampling for handling class imbalance
undersampler = RandomUnderSampler(random_state=42)
x_train_balanced, y_train_balanced = undersampler.fit_resample(x_train_scaled, y_train)

# Print number of instances in each class after Random Undersampling
print("Number of instances in each class after Random Undersampling:")
print(pd.Series(y_train_balanced).value_counts())

# Hyperparameter Tuning using RandomizedSearchCV
param_dist = {
    'n_estimators': randint(100, 1000),
    'max_features': ['auto', 'sqrt'],
    'max_depth': [10, 20, 30, 40, 50, None],
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 20),
    'bootstrap': [True, False]
}

rf_classifier = RandomForestClassifier(random_state=42)
random_search = RandomizedSearchCV(estimator=rf_classifier, param_distributions=param_dist, n_iter=100, cv=5, verbose=2, random_state=42, n_jobs=-1)
random_search.fit(x_train_balanced, y_train_balanced)

best_rf_classifier = random_search.best_estimator_

# Save the best trained model
joblib.dump(best_rf_classifier, 'best_random_forest_model.pkl')

# Load the saved model
saved_model = joblib.load('best_random_forest_model.pkl')

# Apply the saved model on the test set
y_test_pred = saved_model.predict(x_test_scaled)
y_test_pred_proba = saved_model.predict_proba(x_test_scaled)[:, 1]  # Probabilities for class 1

# Calculate evaluation metrics for the training set
accuracy_train = accuracy_score(y_train, best_rf_classifier.predict(x_train_scaled))
precision_train = precision_score(y_train, best_rf_classifier.predict(x_train_scaled))
recall_train = recall_score(y_train, best_rf_classifier.predict(x_train_scaled))
f1_train = f1_score(y_train, best_rf_classifier.predict(x_train_scaled))
roc_auc_train = roc_auc_score(y_train, best_rf_classifier.predict(x_train_scaled))
mcc_train = matthews_corrcoef(y_train, best_rf_classifier.predict(x_train_scaled))# Calculate evaluation metrics for the test set
accuracy_test = accuracy_score(y_test, y_test_pred)
precision_test = precision_score(y_test, y_test_pred)
recall_test = recall_score(y_test, y_test_pred)
f1_test = f1_score(y_test, y_test_pred)
roc_auc_test = roc_auc_score(y_test, y_test_pred_proba)  # Use y_test_pred_proba for ROC AUC
mcc_test = matthews_corrcoef(y_test, y_test_pred)

# Print metrics for the training set
print("Metrics for Trained Model (Training Set):")
print("Accuracy: {:.2f}%".format(accuracy_train * 100))
print("Precision: {:.2f}".format(precision_train))
print("Recall: {:.2f}".format(recall_train))
print("F1-score: {:.2f}".format(f1_train))
print("ROC AUC: {:.2f}".format(roc_auc_train))
print("MCC: {:.2f}".format(mcc_train))

# Print metrics for the test set
print("\nMetrics for Trained Model (Test Set):")
print("Accuracy: {:.2f}%".format(accuracy_test * 100))
print("Precision: {:.2f}".format(precision_test))
print("Recall: {:.2f}".format(recall_test))
print("F1-score: {:.2f}".format(f1_test))
print("ROC AUC: {:.2f}".format(roc_auc_test))
print("MCC: {:.2f}".format(mcc_test))



#########################################################################
### MCI vs AD - Classification Model 3
#########################################################################

# Load your dataset (update the file path)
df = pd.read_csv("/home/sushree/ish_project/data/AD_classification_without_age.csv")
#df.dropna(inplace=True)

# Update mapping for binary classification: 'MCI' (1) vs 'AD' (0)
df['Group'] = df['Group'].map({'MCI': 1, 'AD': 0})

# Print size of each class
print("Size of MCI", (df['Group'] == 1).sum())
print("Size of AD:", (df['Group'] == 0).sum())

# Drop rows related to 'CN'
df = df[df['Group'].isin([0, 1])]

# Separate features and target variable
X = df.drop(['Image_ID','Group','Subject_ID','VISCODE'], axis=1)
y = df['Group']

# Split data into training and test sets with 80-20 split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
# Print size of training and test sets
print("Size of Training Set:", x_train.shape[0])
print("Size of Test Set:", x_test.shape[0])

# Preprocessing: Centering, Scaling, and KNN-Imputation
scaler = StandardScaler()
imputer = KNNImputer(n_neighbors=5)

x_train_scaled = scaler.fit_transform(imputer.fit_transform(x_train))
x_test_scaled = scaler.transform(imputer.transform(x_test))

# Print number of instances in each class before undersampling
print("Number of instances in each class before Random Undersampling:")
print(y_train.value_counts())

# Random Undersampling for handling class imbalance
undersampler = RandomUnderSampler(random_state=42)
x_train_balanced, y_train_balanced = undersampler.fit_resample(x_train_scaled, y_train)

# Print number of instances in each class after Random Undersampling
print("Number of instances in each class after Random Undersampling:")
print(pd.Series(y_train_balanced).value_counts())

# Hyperparameter Tuning using RandomizedSearchCV
param_dist = {
    'n_estimators': randint(100, 1000),
    'max_features': ['auto', 'sqrt'],
    'max_depth': [10, 20, 30, 40, 50, None],
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 20),
    'bootstrap': [True, False]
}

rf_classifier = RandomForestClassifier(random_state=42)
random_search = RandomizedSearchCV(estimator=rf_classifier, param_distributions=param_dist, n_iter=100, cv=5, verbose=2, random_state=42, n_jobs=-1)
random_search.fit(x_train_balanced, y_train_balanced)

best_rf_classifier = random_search.best_estimator_

# Save the best trained model
joblib.dump(best_rf_classifier, 'best_random_forest_model.pkl')

# Load the saved model
saved_model = joblib.load('best_random_forest_model.pkl')

# Apply the saved model on the test set
y_test_pred = saved_model.predict(x_test_scaled)
y_test_pred_proba = saved_model.predict_proba(x_test_scaled)[:, 1]  # Probabilities for class 1

# Calculate evaluation metrics for the training set
accuracy_train = accuracy_score(y_train, best_rf_classifier.predict(x_train_scaled))
precision_train = precision_score(y_train, best_rf_classifier.predict(x_train_scaled))
recall_train = recall_score(y_train, best_rf_classifier.predict(x_train_scaled))
f1_train = f1_score(y_train, best_rf_classifier.predict(x_train_scaled))
roc_auc_train = roc_auc_score(y_train, best_rf_classifier.predict(x_train_scaled))
mcc_train = matthews_corrcoef(y_train, best_rf_classifier.predict(x_train_scaled))# Calculate evaluation metrics for the test set
accuracy_test = accuracy_score(y_test, y_test_pred)
precision_test = precision_score(y_test, y_test_pred)
recall_test = recall_score(y_test, y_test_pred)
f1_test = f1_score(y_test, y_test_pred)
roc_auc_test = roc_auc_score(y_test, y_test_pred_proba)  # Use y_test_pred_proba for ROC AUC
mcc_test = matthews_corrcoef(y_test, y_test_pred)

# Print metrics for the training set
print("Metrics for Trained Model (Training Set):")
print("Accuracy: {:.2f}%".format(accuracy_train * 100))
print("Precision: {:.2f}".format(precision_train))
print("Recall: {:.2f}".format(recall_train))
print("F1-score: {:.2f}".format(f1_train))
print("ROC AUC: {:.2f}".format(roc_auc_train))
print("MCC: {:.2f}".format(mcc_train))

# Print metrics for the test set
print("\nMetrics for Trained Model (Test Set):")
print("Accuracy: {:.2f}%".format(accuracy_test * 100))
print("Precision: {:.2f}".format(precision_test))
print("Recall: {:.2f}".format(recall_test))
print("F1-score: {:.2f}".format(f1_test))
print("ROC AUC: {:.2f}".format(roc_auc_test))
print("MCC: {:.2f}".format(mcc_test))



#########################################################################
### MCI VS AD With AGE - Classification Model 4
#########################################################################

# Load your dataset (update the file path)
df = pd.read_csv("/home/sushree/ish_project/data/AD_classification_with_age.csv")
#df.dropna(inplace=True)

# Update mapping for binary classification: 'MCI' (1) vs 'AD' (0)
df['Group'] = df['Group'].map({'MCI': 1, 'AD': 0})

# Print size of each class
print("Size of MCI", (df['Group'] == 1).sum())
print("Size of AD:", (df['Group'] == 0).sum())

# Drop rows related to 'CN'
df = df[df['Group'].isin([0, 1])]

# Separate features and target variable
X = df.drop(['Image_ID','Group','Subject_ID','VISCODE'], axis=1)
y = df['Group']

# Split data into training and test sets with 80-20 split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
# Print size of training and test sets
print("Size of Training Set:", x_train.shape[0])
print("Size of Test Set:", x_test.shape[0])

# Preprocessing: Centering, Scaling, and KNN-Imputation
scaler = StandardScaler()
imputer = KNNImputer(n_neighbors=5)

x_train_scaled = scaler.fit_transform(imputer.fit_transform(x_train))
x_test_scaled = scaler.transform(imputer.transform(x_test))

# Print number of instances in each class before undersampling
print("Number of instances in each class before Random Undersampling:")
print(y_train.value_counts())

# Random Undersampling for handling class imbalance
undersampler = RandomUnderSampler(random_state=42)
x_train_balanced, y_train_balanced = undersampler.fit_resample(x_train_scaled, y_train)

# Print number of instances in each class after Random Undersampling
print("Number of instances in each class after Random Undersampling:")
print(pd.Series(y_train_balanced).value_counts())

# Hyperparameter Tuning using RandomizedSearchCV
param_dist = {
    'n_estimators': randint(100, 1000),
    'max_features': ['auto', 'sqrt'],
    'max_depth': [10, 20, 30, 40, 50, None],
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 20),
    'bootstrap': [True, False]
}

rf_classifier = RandomForestClassifier(random_state=42)
random_search = RandomizedSearchCV(estimator=rf_classifier, param_distributions=param_dist, n_iter=100, cv=5, verbose=2, random_state=42, n_jobs=-1)
random_search.fit(x_train_balanced, y_train_balanced)

best_rf_classifier = random_search.best_estimator_

# Save the best trained model
joblib.dump(best_rf_classifier, 'best_random_forest_model.pkl')

# Load the saved model
saved_model = joblib.load('best_random_forest_model.pkl')

# Apply the saved model on the test set
y_test_pred = saved_model.predict(x_test_scaled)
y_test_pred_proba = saved_model.predict_proba(x_test_scaled)[:, 1]  # Probabilities for class 1

# Calculate evaluation metrics for the training set
accuracy_train = accuracy_score(y_train, best_rf_classifier.predict(x_train_scaled))
precision_train = precision_score(y_train, best_rf_classifier.predict(x_train_scaled))
recall_train = recall_score(y_train, best_rf_classifier.predict(x_train_scaled))
f1_train = f1_score(y_train, best_rf_classifier.predict(x_train_scaled))
roc_auc_train = roc_auc_score(y_train, best_rf_classifier.predict(x_train_scaled))
mcc_train = matthews_corrcoef(y_train, best_rf_classifier.predict(x_train_scaled))# Calculate evaluation metrics for the test set
accuracy_test = accuracy_score(y_test, y_test_pred)
precision_test = precision_score(y_test, y_test_pred)
recall_test = recall_score(y_test, y_test_pred)
f1_test = f1_score(y_test, y_test_pred)
roc_auc_test = roc_auc_score(y_test, y_test_pred_proba)  # Use y_test_pred_proba for ROC AUC
mcc_test = matthews_corrcoef(y_test, y_test_pred)

# Print metrics for the training set
print("Metrics for Trained Model (Training Set):")
print("Accuracy: {:.2f}%".format(accuracy_train * 100))
print("Precision: {:.2f}".format(precision_train))
print("Recall: {:.2f}".format(recall_train))
print("F1-score: {:.2f}".format(f1_train))
print("ROC AUC: {:.2f}".format(roc_auc_train))
print("MCC: {:.2f}".format(mcc_train))

# Print metrics for the test set
print("\nMetrics for Trained Model (Test Set):")
print("Accuracy: {:.2f}%".format(accuracy_test * 100))
print("Precision: {:.2f}".format(precision_test))
print("Recall: {:.2f}".format(recall_test))
print("F1-score: {:.2f}".format(f1_test))
print("ROC AUC: {:.2f}".format(roc_auc_test))
print("MCC: {:.2f}".format(mcc_test))


#########################################################################
### MCI vs CN - Classification Model 5
#########################################################################

# Load your dataset (update the file path)
df = pd.read_csv("/home/sushree/ish_project/data/AD_classification_without_age.csv")
#df.dropna(inplace=True)

# Update mapping for binary classification: 'MCI' (1) vs 'CN' (0)
df['Group'] = df['Group'].map({'MCI': 1, 'CN': 0})

# Print size of each class
print("Size of CN", (df['Group'] == 1).sum())
print("Size of MCI:", (df['Group'] == 0).sum())

# Drop rows related to 'AD'
df = df[df['Group'].isin([0, 1])]

# Separate features and target variable
X = df.drop(['Image_ID','Group','Subject_ID','VISCODE'], axis=1)
y = df['Group']

# Split data into training and test sets with 80-20 split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
# Print size of training and test sets
print("Size of Training Set:", x_train.shape[0])
print("Size of Test Set:", x_test.shape[0])

# Preprocessing: Centering, Scaling, and KNN-Imputation
scaler = StandardScaler()
imputer = KNNImputer(n_neighbors=5)

x_train_scaled = scaler.fit_transform(imputer.fit_transform(x_train))
x_test_scaled = scaler.transform(imputer.transform(x_test))

# Print number of instances in each class before undersampling
print("Number of instances in each class before Random Undersampling:")
print(y_train.value_counts())

# Random Undersampling for handling class imbalance
undersampler = RandomUnderSampler(random_state=42)
x_train_balanced, y_train_balanced = undersampler.fit_resample(x_train_scaled, y_train)

# Print number of instances in each class after Random Undersampling
print("Number of instances in each class after Random Undersampling:")
print(pd.Series(y_train_balanced).value_counts())

# Hyperparameter Tuning using RandomizedSearchCV
param_dist = {
    'n_estimators': randint(100, 1000),
    'max_features': ['auto', 'sqrt'],
    'max_depth': [10, 20, 30, 40, 50, None],
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 20),
    'bootstrap': [True, False]
}

rf_classifier = RandomForestClassifier(random_state=42)
random_search = RandomizedSearchCV(estimator=rf_classifier, param_distributions=param_dist, n_iter=100, cv=5, verbose=2, random_state=42, n_jobs=-1)
random_search.fit(x_train_balanced, y_train_balanced)

best_rf_classifier = random_search.best_estimator_

# Save the best trained model
joblib.dump(best_rf_classifier, 'best_random_forest_model.pkl')

# Load the saved model
saved_model = joblib.load('best_random_forest_model.pkl')

# Apply the saved model on the test set
y_test_pred = saved_model.predict(x_test_scaled)
y_test_pred_proba = saved_model.predict_proba(x_test_scaled)[:, 1]  # Probabilities for class 1

# Calculate evaluation metrics for the training set
accuracy_train = accuracy_score(y_train, best_rf_classifier.predict(x_train_scaled))
precision_train = precision_score(y_train, best_rf_classifier.predict(x_train_scaled))
recall_train = recall_score(y_train, best_rf_classifier.predict(x_train_scaled))
f1_train = f1_score(y_train, best_rf_classifier.predict(x_train_scaled))
roc_auc_train = roc_auc_score(y_train, best_rf_classifier.predict(x_train_scaled))
mcc_train = matthews_corrcoef(y_train, best_rf_classifier.predict(x_train_scaled))# Calculate evaluation metrics for the test set
accuracy_test = accuracy_score(y_test, y_test_pred)
precision_test = precision_score(y_test, y_test_pred)
recall_test = recall_score(y_test, y_test_pred)
f1_test = f1_score(y_test, y_test_pred)
roc_auc_test = roc_auc_score(y_test, y_test_pred_proba)  # Use y_test_pred_proba for ROC AUC
mcc_test = matthews_corrcoef(y_test, y_test_pred)

# Print metrics for the training set
print("Metrics for Trained Model (Training Set):")
print("Accuracy: {:.2f}%".format(accuracy_train * 100))
print("Precision: {:.2f}".format(precision_train))
print("Recall: {:.2f}".format(recall_train))
print("F1-score: {:.2f}".format(f1_train))
print("ROC AUC: {:.2f}".format(roc_auc_train))
print("MCC: {:.2f}".format(mcc_train))

# Print metrics for the test set
print("\nMetrics for Trained Model (Test Set):")
print("Accuracy: {:.2f}%".format(accuracy_test * 100))
print("Precision: {:.2f}".format(precision_test))
print("Recall: {:.2f}".format(recall_test))
print("F1-score: {:.2f}".format(f1_test))
print("ROC AUC: {:.2f}".format(roc_auc_test))
print("MCC: {:.2f}".format(mcc_test))




#########################################################################
### MCI VS CN With AGE - Classification Model 6
#########################################################################

# Load your dataset (update the file path)
df = pd.read_csv("/home/sushree/ish_project/data/AD_classification_with_age.csv")
#df.dropna(inplace=True)

# Update mapping for binary classification: 'MCI' (1) vs 'CN' (0)
df['Group'] = df['Group'].map({'MCI': 1, 'CN': 0})

# Print size of each class
print("Size of MCI", (df['Group'] == 1).sum())
print("Size of CN:", (df['Group'] == 0).sum())

# Drop rows related to 'AD'
df = df[df['Group'].isin([0, 1])]

# Separate features and target variable
X = df.drop(['Image_ID','Group','Subject_ID','VISCODE'], axis=1)
y = df['Group']

# Split data into training and test sets with 80-20 split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
# Print size of training and test sets
print("Size of Training Set:", x_train.shape[0])
print("Size of Test Set:", x_test.shape[0])

# Preprocessing: Centering, Scaling, and KNN-Imputation
scaler = StandardScaler()
imputer = KNNImputer(n_neighbors=5)

x_train_scaled = scaler.fit_transform(imputer.fit_transform(x_train))
x_test_scaled = scaler.transform(imputer.transform(x_test))

# Print number of instances in each class before undersampling
print("Number of instances in each class before Random Undersampling:")
print(y_train.value_counts())

# Random Undersampling for handling class imbalance
undersampler = RandomUnderSampler(random_state=42)
x_train_balanced, y_train_balanced = undersampler.fit_resample(x_train_scaled, y_train)

# Print number of instances in each class after Random Undersampling
print("Number of instances in each class after Random Undersampling:")
print(pd.Series(y_train_balanced).value_counts())

# Hyperparameter Tuning using RandomizedSearchCV
param_dist = {
    'n_estimators': randint(100, 1000),
    'max_features': ['auto', 'sqrt'],
    'max_depth': [10, 20, 30, 40, 50, None],
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 20),
    'bootstrap': [True, False]
}

rf_classifier = RandomForestClassifier(random_state=42)
random_search = RandomizedSearchCV(estimator=rf_classifier, param_distributions=param_dist, n_iter=100, cv=5, verbose=2, random_state=42, n_jobs=-1)
random_search.fit(x_train_balanced, y_train_balanced)

best_rf_classifier = random_search.best_estimator_

# Save the best trained model
joblib.dump(best_rf_classifier, 'best_random_forest_model.pkl')

# Load the saved model
saved_model = joblib.load('best_random_forest_model.pkl')

# Apply the saved model on the test set
y_test_pred = saved_model.predict(x_test_scaled)
y_test_pred_proba = saved_model.predict_proba(x_test_scaled)[:, 1]  # Probabilities for class 1

# Calculate evaluation metrics for the training set
accuracy_train = accuracy_score(y_train, best_rf_classifier.predict(x_train_scaled))
precision_train = precision_score(y_train, best_rf_classifier.predict(x_train_scaled))
recall_train = recall_score(y_train, best_rf_classifier.predict(x_train_scaled))
f1_train = f1_score(y_train, best_rf_classifier.predict(x_train_scaled))
roc_auc_train = roc_auc_score(y_train, best_rf_classifier.predict(x_train_scaled))
mcc_train = matthews_corrcoef(y_train, best_rf_classifier.predict(x_train_scaled))# Calculate evaluation metrics for the test set
accuracy_test = accuracy_score(y_test, y_test_pred)
precision_test = precision_score(y_test, y_test_pred)
recall_test = recall_score(y_test, y_test_pred)
f1_test = f1_score(y_test, y_test_pred)
roc_auc_test = roc_auc_score(y_test, y_test_pred_proba)  # Use y_test_pred_proba for ROC AUC
mcc_test = matthews_corrcoef(y_test, y_test_pred)

# Print metrics for the training set
print("Metrics for Trained Model (Training Set):")
print("Accuracy: {:.2f}%".format(accuracy_train * 100))
print("Precision: {:.2f}".format(precision_train))
print("Recall: {:.2f}".format(recall_train))
print("F1-score: {:.2f}".format(f1_train))
print("ROC AUC: {:.2f}".format(roc_auc_train))
print("MCC: {:.2f}".format(mcc_train))

# Print metrics for the test set
print("\nMetrics for Trained Model (Test Set):")
print("Accuracy: {:.2f}%".format(accuracy_test * 100))
print("Precision: {:.2f}".format(precision_test))
print("Recall: {:.2f}".format(recall_test))
print("F1-score: {:.2f}".format(f1_test))
print("ROC AUC: {:.2f}".format(roc_auc_test))
print("MCC: {:.2f}".format(mcc_test))




















