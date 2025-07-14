import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

#loading the dataset to a pandas dataframe
credit_card_data=pd.read_csv(r'C:\Users\amitk\Desktop\python codes\SEM_5\ML LAB\Project\creditcard.csv')
credit_card_data.info()
# Printing first 5 rows
print(credit_card_data.head())
# Printing last 5 rows
print(credit_card_data.tail())
# Checking for null entries and doing its sum
print(credit_card_data.isnull().sum())
# Counting no. of Columns
print(credit_card_data['Class'].value_counts())

#separating the dataset for analysis
# 0 -> Normal Transaction
# 1 -> Fraudlet Transaction
legit = credit_card_data[credit_card_data.Class == 0]
fraud = credit_card_data[credit_card_data.Class == 1]
print(legit.shape)
print(fraud.shape)
# statistical measures of the data
print(legit.Amount.describe())
print(fraud.Amount.describe())
print(credit_card_data.groupby('Class').mean())
# Under Sampling
legit_sample=legit.sample(n=492)

# Concatenating Two Dataframes
neew_dataframes=pd.concat([legit_sample,fraud], axis=0) 
# axis=0 -> rowwise concatenation
# axis=1 -> columnwise concatenation
print(neew_dataframes.head())
print(neew_dataframes.value_counts())
print(neew_dataframes.groupby('Class').mean())

# Splitting the Data into Features and Targets
X = credit_card_data.drop(columns='Class',axis=1)
Y = credit_card_data['Class']
print(X)
print(Y)

# Splitting the Data intoo training and testing data
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=2)
print(X.shape,X_train.shape,X_test.shape)

# Adding Evaluation Metrics
def evaluate_model(y_true, y_pred, y_prob):
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_prob)
    
    print("\nEvaluation Metrics:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")

# Training and Evaluating on each model
def train_and_evaluate_model(model, model_name, X_train, X_test, Y_train, Y_test):
    print(f"\nTraining {model_name}...")
    model.fit(X_train, Y_train)
    
    # Predictions
    Y_train_pred = model.predict(X_train)
    Y_test_pred = model.predict(X_test)
    Y_test_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    
    # Metrics
    training_accuracy = accuracy_score(Y_train, Y_train_pred)
    test_accuracy = accuracy_score(Y_test, Y_test_pred)
    precision = precision_score(Y_test, Y_test_pred)
    recall = recall_score(Y_test, Y_test_pred)
    f1 = f1_score(Y_test, Y_test_pred)
    roc_auc = roc_auc_score(Y_test, Y_test_prob) if Y_test_prob is not None else "Not applicable"
    
    print(f"\nModel: {model_name}")
    print(f"Training Accuracy: {training_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC-AUC: {roc_auc}\n")
    
    return {
        "Model": model_name,
        "Training Accuracy": training_accuracy,
        "Test Accuracy": test_accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC-AUC": roc_auc
    }

# Models to train
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Support Vector Machine (SVM)": SVC(kernel='linear', probability=True, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
}

# Train and evaluate each model
results = []
for model_name, model_instance in models.items():
    result = train_and_evaluate_model(model_instance, model_name, X_train, X_test, Y_train, Y_test)
    results.append(result)

# Compare results in a DataFrame
results_df = pd.DataFrame(results)
print("\nComparison of Models:")
print(results_df)


def predict_with_all_models(models, transaction_data, feature_columns):
    """
    Predict whether a new transaction is fraudulent or normal for each trained model.

    Parameters:
    - models: Dictionary of trained models with model names as keys
    - transaction_data: A dictionary containing the transaction details (features)
    - feature_columns: List of feature names expected by the models (order matters)

    Returns:
    - predictions: A dictionary containing predictions and probabilities for each model
    """
    # Convert the transaction data into a pandas DataFrame
    transaction_df = pd.DataFrame([transaction_data])
    
    # Ensure the features match the trained model's feature order
    transaction_df = transaction_df[feature_columns]
    
    predictions = {}
    for model_name, model in models.items():
        # Make prediction
        prediction = model.predict(transaction_df)[0]
        probability = (
            model.predict_proba(transaction_df)[0][1]
            if hasattr(model, "predict_proba")
            else None
        )
        
        # Store the results
        predictions[model_name] = {
            "Prediction": "Fraudulent" if prediction == 1 else "Normal",
            "Probability": f"{probability:.4f}" if probability is not None else "N/A",
        }
    
    # Print the predictions
    print("\nPredictions for the given transaction:")
    for model_name, result in predictions.items():
        print(
            f"Model: {model_name}, Prediction: {result['Prediction']}, Probability of Fraudulent: {result['Probability']}"
        )
    
    return predictions

# Example new transaction data
new_transaction = {
    "Time": 150000,
    "V1": -1.359807,
    "V2": -0.072781,
    "V3": 2.536346,
    "V4": 1.378155,
    "V5": -0.338321,
    "V6": 0.462388,
    "V7": 0.239599,
    "V8": 0.098698,
    "V9": 0.363787,
    "V10": 0.090794,
    "V11": -0.551599,
    "V12": -0.617801,
    "V13": -0.991390,
    "V14": -0.311169,
    "V15": 1.468177,
    "V16": -0.470400,
    "V17": 0.207971,
    "V18": 0.025791,
    "V19": 0.403993,
    "V20": 0.251412,
    "V21": -0.018307,
    "V22": 0.277838,
    "V23": -0.110474,
    "V24": 0.066928,
    "V25": 0.128539,
    "V26": -0.189115,
    "V27": 0.133558,
    "V28": 0.021053,
    "Amount": 100.0
}

# List of feature columns
feature_columns = X.columns.tolist()

# Call the function
predict_with_all_models(models, new_transaction, feature_columns)


def predict_new_transaction(model, transaction_data):
    """
    Predict whether a new transaction is fraudulent or not.

    Parameters:
    - model: Trained logistic regression model
    - transaction_data: A dictionary containing the transaction details (features)

    Returns:
    - Prediction: 0 (Normal) or 1 (Fraudulent)
    - Probability: Probability of the transaction being fraudulent
    """
    # Convert the transaction data into a pandas DataFrame
    transaction_df = pd.DataFrame([transaction_data])
    
    # Ensure the features match the trained model's feature order
    transaction_df = transaction_df[X.columns]
    
    # Make prediction
    prediction = model.predict(transaction_df)[0]
    probability = model.predict_proba(transaction_df)[0][1]
    
    print("\nMAKING Prediction on given user transaction")
    # Output the result
    if prediction == 1:
        print(f"The transaction is predicted to be FRAUDULENT with a probability of {probability:.4f}.")
    else:
        print(f"The transaction is predicted to be NORMAL with a probability of {probability:.4f}.")
    
    return prediction, probability


# Example new transaction data (replace these with real values)
new_transaction = {
    "Time": 150000,
    "V1": -1.359807,
    "V2": -0.072781,
    "V3": 2.536346,
    "V4": 1.378155,
    "V5": -0.338321,
    "V6": 0.462388,
    "V7": 0.239599,
    "V8": 0.098698,
    "V9": 0.363787,
    "V10": 0.090794,
    "V11": -0.551599,
    "V12": -0.617801,
    "V13": -0.991390,
    "V14": -0.311169,
    "V15": 1.468177,
    "V16": -0.470400,
    "V17": 0.207971,
    "V18": 0.025791,
    "V19": 0.403993,
    "V20": 0.251412,
    "V21": -0.018307,
    "V22": 0.277838,
    "V23": -0.110474,
    "V24": 0.066928,
    "V25": 0.128539,
    "V26": -0.189115,
    "V27": 0.133558,
    "V28": -0.021053,
    "Amount": 100.0
}

# Call the function
predict_new_transaction(model, new_transaction)
