import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def train_and_save_model():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'data', 'ai4i2020.csv')
    model_dir = os.path.join(base_dir, 'models')
    model_path = os.path.join(model_dir, 'rf_model.joblib')
    
    if not os.path.exists(data_path):
        print(f"Error: Dataset not found. Please run download_data.py first.")
        return
        
    print("Dataset Loaded. Preparing for ML architecture...")
    df = pd.read_csv(data_path)
    
    # Feature Engineering and Data Cleaning
    # UDI and Product ID are unique identifiers and will not contribute to prediction.
    # 'Machine failure' is our primary target variable.
    # 'TWF', 'HDF', 'PWF', 'OSF', 'RNF' are specific sub-types of failure. 
    # To prevent data leakage, they must be removed from the input features.
    
    cols_to_drop = ['Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']
    
    X = df.drop(cols_to_drop, axis=1)
    y = df['Machine failure']
    
    # The 'Type' column is categorical (L, M, H) -> Needs One-Hot Encoding.
    categorical_features = ['Type']
    numeric_features = [col for col in X.columns if col != 'Type']
    
    # Data preprocessing steps (Column Transformations)
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
        
    # Machine Learning Pipeline
    # Applying class_weight='balanced' to handle class imbalance (Failure=1 is rare).
    rf_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'))
    ])
    
    # Splitting data (80% training, 20% test); using stratify=y to maintain failure ratio
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("Training Model (Random Forest)... This might take a while.")
    rf_pipeline.fit(X_train, y_train)
    
    # Testing and Evaluation
    y_pred = rf_pipeline.predict(X_test)
    print("\n" + "="*40)
    print("Model Training Results - Classification Report:")
    print("="*40)
    print(classification_report(y_test, y_pred))
    print(f"Overall Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    
    # Generate and Save Confusion Matrix Plot
    cm = confusion_matrix(y_test, y_pred, labels=rf_pipeline.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal (0)", "Failure (1)"])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap=plt.cm.Blues, values_format='d')
    plt.title("Random Forest Confusion Matrix")
    plt.tight_layout()
    cm_path = os.path.join(model_dir, 'confusion_matrix.png')
    
    # Save the trained model and plot
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        
    plt.savefig(cm_path, dpi=300)
    joblib.dump(rf_pipeline, model_path)
    print(f"\nSuccess! Trained model exported to '{model_path}'.")
    print(f"Confusion Matrix saved to '{cm_path}'.")

if __name__ == "__main__":
    train_and_save_model()
