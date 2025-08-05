import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns 



mlflow.set_tracking_uri("http://localhost:5000")  # Set your MLflow tracking URI

#Load the wine dataset
wine = load_wine()
X = wine.data
y = wine.target



#Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  

#define parameters for the model
params = {
    'n_estimators': 15,
    'max_depth': 5,
    'random_state': 42
}


mlflow.set_experiment("MLflow-learning-Exp1")  # Set your experiment name

with mlflow.start_run():
    #train the model
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_params(params)
    
    
    #Create a cnfusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=wine.target_names, yticklabels=wine.target_names)
    plt.title('Confusion Matrix')   
    plt.xlabel('Predicted')
    plt.ylabel('Actual')    
    plt.savefig("confusion_matrix.png")
    
    signature = infer_signature(X, model.predict(X))
    
    
    mlflow.log_artifact("confusion_matrix.png")
    mlflow.log_artifact(__file__)
    
    mlflow.set_tags({"Author":"Osama Sher","Project":"Wine Classification","model": "RandomForestClassifier"})
    
    mlflow.sklearn.log_model(model, "wine_classifier_rf",input_example=X_train[:5].tolist())  # Log the model with an input example
    
    
    print(f"Model accuracy: {accuracy}")