
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

train_data = pd.read_excel('/content/drive/MyDrive/Train.xlsx')  # Load training data

target_column = 'PCE'   # Identify the target column
                       # Identify the target column# Replace with the actual name of your target column

             # Check if the target column exists in the training dataset
if target_column in train_data.columns:
     # Extract features and target for training set
    X_train = train_data.drop(target_column, axis=1)
    y_train = train_data[target_column]

    # Load test data
    test_data = pd.read_excel('/content/drive/MyDrive/Test.xlsx')

    # Check if the target column exists in the test dataset
    if target_column in test_data.columns:
        # Extract features and target for testing set
        X_test = test_data.drop(target_column, axis=1)
        y_test = test_data[target_column]

        # Scale the data using StandardScaler
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Logistic Regression
        logistic_model = LogisticRegression()
        logistic_model.fit(X_train_scaled, y_train)
        y_pred_logistic = logistic_model.predict(X_test_scaled)
        accuracy_logistic = accuracy_score(y_test, y_pred_logistic)
        conf_matrix_logistic = confusion_matrix(y_test, y_pred_logistic)

        print("Logistic Regression:")
        print(f"Accuracy (Train): {accuracy_score(y_train, logistic_model.predict(X_train_scaled))}")
        print(f"Accuracy (Test): {accuracy_logistic}")
        print(f"Confusion Matrix:\n{conf_matrix_logistic}")
        feature_importance_lr = logistic_model.coef_[0]
        feature_names = X_train.columns  # Correct definition of feature_names
        feature_importance_lr_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance_lr})
        feature_importance_lr_df = feature_importance_lr_df.sort_values(by='Importance', ascending=False)

        print("\nTop Features in Logistic Regression:")
        print(feature_importance_lr_df.head())

        # Support Vector Machine (SVM)
        svm_model = SVC()
        svm_model.fit(X_train_scaled, y_train)
        y_pred_svm = svm_model.predict(X_test_scaled)
        accuracy_svm = accuracy_score(y_test, y_pred_svm)
        conf_matrix_svm = confusion_matrix(y_test, y_pred_svm)

        print("\nSupport Vector Machine:")
        print(f"Accuracy (Train): {accuracy_score(y_train, svm_model.predict(X_train_scaled))}")
        print(f"Accuracy (Test): {accuracy_svm}")
        print(f"Confusion Matrix:\n{conf_matrix_svm}")



        # Random Forest
        rf_model = RandomForestClassifier()
        rf_model.fit(X_train_scaled, y_train)
        y_pred_rf = rf_model.predict(X_test_scaled)
        accuracy_rf = accuracy_score(y_test, y_pred_rf)
        conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)

        print("\nRandom Forest:")
        print(f"Accuracy (Train): {accuracy_score(y_train, rf_model.predict(X_train_scaled))}")
        print(f"Accuracy (Test): {accuracy_rf}")
        print(f"Confusion Matrix:\n{conf_matrix_rf}")
        feature_importance_rf = rf_model.feature_importances_
        feature_importance_rf_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance_rf})
        feature_importance_rf_df = feature_importance_rf_df.sort_values(by='Importance', ascending=False)

        print("\nTop Features in Random Forest:")
        print(feature_importance_rf_df.head())
    else:
        print(f"Error: '{target_column}' column not found in the test dataset.")
else:
    print(f"Error: '{target_column}' column not found in the training dataset.")
