import mlflow 
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
 
# Set up MLflow experiment
mlflow.set_experiment("iris_classification")
 

 #load iris dataset and split into train/test sets
iris = load_iris()
# print(iris)   #loaded dataset
# print(iris.data)
# print(iris.target)

#let's split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(
    iris.data, #features
    iris.target, #targets
    test_size = 0.2, #20% test data
    random_state = 70 # for reproducibility
)

# print("X_train:", x_train)
# print("Y_train:", y_train)
# print("X_test:", x_test)
# print("Y_test:", y_test)

#start MLflow run
with mlflow.start_run():
    #train mdoel
    model = LogisticRegression(max_iter = 200)
    #fit the model
    model.fit(x_train, y_train)

    #make predictions and calculate accuracy
    preds = model.predict(x_test) #remaining 3 samples
    #preds are the predicted targets(0 / 1 / 2) for the test set

    #comparing predictions vs actual labels to find out accuracy
    acc = accuracy_score(y_test, preds)

    #log to MLflow
    mlflow.log_param("model_type", "logistic_regression")
    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(model, "model")

    print(f"Run logged to mlflow. accuracy: {acc:.3f}")
