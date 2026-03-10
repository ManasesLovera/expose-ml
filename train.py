import joblib
import os
from pathlib import Path

from ucimlrepo import fetch_ucirepo
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

from sklearn.base import ClassifierMixin
from settings import settings

def train():

    # LOAD DATASET
    
    # fetch dataset 
    spambase = fetch_ucirepo(id=94) 
    
    # data (as pandas dataframes) 
    X = spambase.data.features 
    y = spambase.data.targets 
    
    # metadata 
    # print(spambase.metadata) 
    
    # variable information 
    # print(spambase.variables) 
    
    # Concatenate along columns (axis=1)
    df = pd.concat([X, y], axis=1)

    print(df.head())

    with open('example_data.txt', 'w') as f:
    # Use to_string() to write the entire DataFrame as a formatted table
        f.write(df.head(10).to_string())
    

    # 1. Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y.values.ravel(), test_size=0.2, random_state=42)

    # 2. Scale Features (Crucial for SVM/LogReg)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 3. Define and Evaluate Models
    models: dict[str, ClassifierMixin] = {
        "SVM": SVC(),
        "Random Forest": RandomForestClassifier(),
        "Logistic Regression": LogisticRegression()
    }

    # Create a directory for model artifacts if it doesn't exist.
    models_dir = Path(settings.models_dir)
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(scaler, models_dir / settings.scaler_filename)

    for name, model in models.items():
        # Train
        model.fit(X_train, y_train)
        
        # Save to disk
        # Replaces spaces with underscores for clean filenames
        filename = models_dir / f"{name.lower().replace(' ', '_')}_model.pkl"
        joblib.dump(model, filename)
        
        # Evaluate
        y_pred = model.predict(X_test)
        print(f"\n--- {name} (Saved to {filename}) ---")
        print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    train()
