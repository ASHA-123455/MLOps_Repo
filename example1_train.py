import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
# Set up MLflow experiment
mlflow.set_experiment("iris_classification")
iris=load_iris()
# print(iris)
# print(iris.data)
# print(iris.target)
X_train,X_test,y_train,y_test = train_test_split(
    iris.data,
    iris.target,
    test_size=0.2,
    random_state=40
)
with mlflow.start_run():
    model = LogisticRegression(max_iter=200)
    model.fit(X_train,y_train)
    preds=model.predict(X_test)
    acc=accuracy_score(y_test,preds)
    mlflow.log_param("model_type","logistic_regression")
    mlflow.log_metric("accuracy",acc)
    mlflow.sklearn.log_model(model,"model")
    print(f"Run logged to MLflow. \nAccuracy:{acc:.3f}")