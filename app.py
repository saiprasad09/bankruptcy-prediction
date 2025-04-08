import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix , classification_report, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import streamlit as st


import warnings
warnings.filterwarnings('ignore')


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

st.title("Bankruptcy Prediction App")

uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx"])

if uploaded_file is not None:
    xlsx = pd.ExcelFile(uploaded_file)
    sheet_names = xlsx.sheet_names
    st.write("Available sheets:", sheet_names)

    sheet_to_read = st.selectbox("Select a sheet", sheet_names)
    df = pd.read_excel(xlsx, sheet_name=sheet_to_read)

    st.write("Data Preview:")
    st.dataframe(df)


    
    df = data.copy()







# Display basic information and first few rows


"""
The dataset is not properly parsed in separate columns.
The issue is that all values are stored in a single column
instead of being split into multiple columns.
"""

# To Split the single column into multiple columns using semicolon as the delimiter
df = df.iloc[:, 0].str.split(";", expand=True)

df.head()

df.columns

# Nameing the columns of the dataset df
df.columns = ["industrial_risk", "management_risk", "financial_flexibility",
    "credibility", "competitiveness", "operating_risk", "class"]

df.head()

df.info()# Asummary of the Dataset

#data types of the columns
df.dtypes

#Data Structure
df.shape

#data type convertion - converting to float from object type
df['industrial_risk'] = df['industrial_risk'].astype(float)

df['management_risk'] = df['management_risk'].astype(float)

df['financial_flexibility'] = df['financial_flexibility'].astype(float)

df['credibility'] = df['credibility'].astype(float)

df['competitiveness'] = df['competitiveness'].astype(float)

df['operating_risk'] = df['operating_risk'].astype(float)

df.dtypes

unique_values = {col: df[col].unique() for col in df.columns}
print(unique_values)


#check duplicaties
df.duplicated().sum()

# dropping the duplicated rows
df = df.drop_duplicates()

# 103 Non-Duplicated rows with 7 columns each
df.shape

df.duplicated().sum()

#checking null values
df.isnull().sum()
# has no null values

# checking outliers
if df.isnull().sum().sum() == 0:
    print("No missing values found in the dataset.")
else:
    # Define colors: Blue for non-missing, Yellow for missing
    plt.figure(figsize=(10, 6))
    sb.heatmap(df.isnull(), cmap=['#000099', '#ffff00'], cbar=False)
    plt.title("Missing Values Heatmap")
    plt.show()


df.columns

# search for outliers for column - industrial_risk
Ir_box=plt.boxplot(df['industrial_risk'])
# there no outliers

# search for outliers for column - management_risk
mr_box=plt.boxplot(df['management_risk'])
# there no outliers

# search for outliers for column - financial_flexibility
ff_box=plt.boxplot(df['financial_flexibility'])
# there no outliers

# search for outliers for column - credibility
c_box=plt.boxplot(df['credibility'])
# there no outliers

# search for outliers for column - competitiveness
comp_box=plt.boxplot(df['competitiveness'])
# there no outliers

# search for outliers for column - operating_risk
or_box=plt.boxplot(df['operating_risk'])
# there no outliers

# checkinig for class imbalance
print(df["class"].value_counts())
#class imbalance is occuring.

"""By the results, Class imbalance occurs when one class (e.g., "bankruptcy")
has significantly fewer instances than the other (e.g., "non-bankruptcy").
This can impact model performance, making it biased towards the majority class."""

# To overcome the class imbalance, we use oversampling SMOTE(Synthetic Minority Over-sampling Technique)
#  -Increase minority clas samples.Because the dataset<250 samples
from imblearn.over_sampling import SMOTE

X = df.drop(columns=["class"])  # Features
y = df["class"]  # Target

smote = SMOTE(sampling_strategy='auto', random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)


# Checking class distribution after SMOTE
print(pd.Series(y_resampled).value_counts())

# Encode the target variable
df["class"] = df["class"].map({"non-bankruptcy": 0, "bankruptcy": 1})

# For model building check the type (relationship)
# of data-linear, non-linear and complex

df

#Checking for Linear Relationship
# Compute correlation matrix
corr_matrix = df.corr()

# Plot heatmap
plt.figure(figsize=(8, 6))
sb.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()
# interpertation - correlations are close to +1 or -1, data is likely linear

# Split features and target

X = df.drop(columns=["class"])
y = df["class"]

# Split into train (80%) and test (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# Standardize the features (important for Logistic Regression)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
#we standardize the features before
# applying Logistic Regression to improve model performance

# Feature scaling (only for SVM and k-NN)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Train the model - Logistic Regression
log_model = LogisticRegression()
log_model.fit(X_train, y_train)
y_pred_log = log_model.predict(X_test)

accuracy_log = accuracy_score(y_test, y_pred_log)

print(" Logistic Regression:")
print(f"Accuracy: {accuracy_log:.2f}")
print("Classification Report:\n", classification_report(y_test, y_pred_log))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_log), "\n")


#Train the model - Decision Tree
tree_model = DecisionTreeClassifier(random_state=42)
tree_model.fit(X_train, y_train)
y_pred_tree = tree_model.predict(X_test)

accuracy_tree = accuracy_score(y_test, y_pred_tree)

print(" Decision Tree Classifier:")
print(f"Accuracy: {accuracy_tree:.2f}")
print("Classification Report:\n", classification_report(y_test, y_pred_tree))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_tree), "\n")


# Train the model - Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

accuracy_rf = accuracy_score(y_test, y_pred_rf)

print(" Random Forest Classifier:")
print(f"Accuracy: {accuracy_rf:.2f}")
print("Classification Report:\n", classification_report(y_test, y_pred_rf))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf), "\n")


# Train the model-SVM
svm_model = SVC()
svm_model.fit(X_train_scaled, y_train)  # Using scaled features
y_pred_svm = svm_model.predict(X_test_scaled)

accuracy_svm = accuracy_score(y_test, y_pred_svm)

print(" Support Vector Machine (SVM):")
print(f"Accuracy: {accuracy_svm:.2f}")
print("Classification Report:\n", classification_report(y_test, y_pred_svm))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_svm), "\n")


# Train the model -  Navie Bayes
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
y_pred_nb = nb_model.predict(X_test)

accuracy_nb = accuracy_score(y_test, y_pred_nb)

print(" Naïve Bayes Classifier:")
print(f"Accuracy: {accuracy_nb:.2f}")
print("Classification Report:\n", classification_report(y_test, y_pred_nb))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_nb), "\n")


# Train the model - k-Nearest Neighbors (k-NN)
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_scaled, y_train)  # Using scaled features
y_pred_knn = knn_model.predict(X_test_scaled)

accuracy_knn = accuracy_score(y_test, y_pred_knn)

print(" k-Nearest Neighbors (k-NN):")
print(f"Accuracy: {accuracy_knn:.2f}")
print("Classification Report:\n", classification_report(y_test, y_pred_knn))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_knn), "\n")


# Train the model-Gradient Boosting (XGBoost)
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)

accuracy_xgb = accuracy_score(y_test, y_pred_xgb)

print(" XGBoost Classifier:")
print(f"Accuracy: {accuracy_xgb:.2f}")
print("Classification Report:\n", classification_report(y_test, y_pred_xgb))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_xgb), "\n")


# Compare Model Performance
models = [
    'Logistic Regression', 'Decision Tree', 'Random Forest',
    'SVM', 'Naïve Bayes', 'k-NN', 'XGBoost'
]
accuracies = [accuracy_log, accuracy_tree, accuracy_rf, accuracy_svm, accuracy_nb, accuracy_knn, accuracy_xgb]

plt.figure(figsize=(10,5))
plt.bar(models, accuracies, color=['blue', 'green', 'red', 'purple', 'orange', 'brown', 'pink'])
plt.xlabel("Classification Models")
plt.ylabel("Accuracy")
plt.title("📊 Comparison of Classification Model Accuracies")
plt.ylim(0, 1)  # Since accuracy is between 0 and 1
plt.xticks(rotation=30)
plt.show()


import sys
import subprocess

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

try:
    from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
    import matplotlib.pyplot as plt
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    import joblib
except ModuleNotFoundError as e:
    missing_module = str(e).split("'")[1]
    print(f"Module {missing_module} not found. Installing...")
    install(missing_module)
    from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
    import matplotlib.pyplot as plt
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    import joblib



from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import pandas as pd

def evaluate_models(models, X_train, X_test, y_train, y_test):
    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        results.append({
            'Model': name,
            'Accuracy': accuracy_score(y_test, y_pred),
            'ROC-AUC': roc_auc_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1 Score': f1_score(y_test, y_pred)
        })
    return results #Fixed: Indentation aligned with the 'for' loop

models = {
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(),
    'Naïve Bayes': GaussianNB(),
    'k-NN': KNeighborsClassifier(n_neighbors=5),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
}

scaler = StandardScaler()

results = evaluate_models(models, X_train, X_test, y_train, y_test)

df_results = pd.DataFrame(results)
df_results = df_results.sort_values(by='Accuracy', ascending=False)

print(df_results)

plt.figure(figsize=(10, 6))
plt.barh(df_results['Model'], df_results['Accuracy'], color='skyblue')
plt.xlabel('Accuracy')
plt.ylabel('Model')
plt.title('Model Performance Comparison')
plt.gca().invert_yaxis()
plt.show()

best_model = df_results.iloc[0]['Model']
print(f"Best model based on Accuracy: {best_model}")
# return statement should only be present inside a function
# return df_results, models[best_model]

tuned_models = {
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'XGBoost': XGBClassifier(),
    'SVM': SVC(),
    'KNN': KNeighborsClassifier()
}

df_results = evaluate_models(tuned_models, X_train, X_test, y_train, y_test)

import joblib
joblib.dump(best_model, 'best_model.pkl')

import subprocess
subprocess.run(["pip", "install", "streamlit"])


import streamlit as st
import joblib
import numpy as np
import pandas as pd

def load_model():
    return joblib.load('best_model.pkl')

def predict(model, input_data):
    prediction = model.predict(input_data)
    return prediction[0]

# Streamlit UI
st.title('Bankruptcy Prediction App')
st.write('Enter the financial details to predict bankruptcy risk.')

 #User Inputs (Modify based on dataset features)
feature1 = st.number_input('Feature 1', min_value=0.0, max_value=100.0, value=50.0)
feature2 = st.number_input('Feature 2', min_value=0.0, max_value=100.0, value=50.0)
feature3 = st.number_input('Feature 3', min_value=0.0, max_value=100.0, value=50.0)
feature4 = st.number_input('Feature 4', min_value=0.0, max_value=100.0, value=50.0)
feature5 = st.number_input('Feature 5', min_value=0.0, max_value=100.0, value=50.0)

# Convert inputs to NumPy array
input_features = np.array([[feature1, feature2, feature3, feature4, feature5]])

if st.button('Predict'):
    model = load_model()
    prediction = predict(model, input_features)
    st.write(f'**Prediction:** {"Bankrupt" if prediction == 1 else "Not Bankrupt"}')

import subprocess

subprocess.run(["pip", "install", "pyngrok"])


with open("app.py", "w") as f:
    f.write("print('Hello, World!')")

import streamlit as st
import joblib
import numpy as np

def load_model():
    return joblib.load('best_model.pkl')

def predict(model, input_data):
    prediction = model.predict(input_data)
    return prediction[0]

st.title('Bankruptcy Prediction App')
st.write('Enter the financial details to predict bankruptcy risk.')

# User Inputs (Modify based on dataset features)
feature1 = st.number_input('Feature 1', min_value=0.0, max_value=100.0, value=50.0)
feature2 = st.number_input('Feature 2', min_value=0.0, max_value=100.0, value=50.0)
feature3 = st.number_input('Feature 3', min_value=0.0, max_value=100.0, value=50.0)
feature4 = st.number_input('Feature 4', min_value=0.0, max_value=100.0, value=50.0)
feature5 = st.number_input('Feature 5', min_value=0.0, max_value=100.0, value=50.0)

# Convert inputs to NumPy array
input_features = np.array([[feature1, feature2, feature3, feature4, feature5]])

if st.button('Predict'):
    model = load_model()
    prediction = predict(model, input_features)
    st.write(f'**Prediction:** {"Bankrupt" if prediction == 1 else "Not Bankrupt"}')

