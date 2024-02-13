# CustomerChurn
End to end ML production pipeline for Customer Churn Prediction 

Machine Learning Case Study: Telco Customer Churn Prediction
Dataset : https://www.kaggle.com/datasets/blastchar/telco-customer-churn

#### **About Dataset**:
Customers who left within the last month – the column is called Churn


The objective is to predict whether users are likely to churn in the future based on their characteristics derived from historical data. By identifying potential churners, targeted campaigns can be deployed to retain customers, as the cost of retention is typically lower than acquisition.

To establish the periodicity of churn, we assume a monthly basis for customers on monthly contracts and a yearly basis for those on annual contracts. However, as contract end dates are unavailable in the data, we opt for a monthly churn assumption.

From the Data Analysis conducted in notebooks/Telco_churn_prediction.ipynb, it was observed that tenure exhibits a strong correlation with churn. However, considering tenure for active customers introduces potential data leakage. Therefore, we assume tenure to be the duration (in months) a customer has been active or was active before churning.

#### **Hypotheses**:
1. Customers more likely to churn tend to have lower tenure.
2. Customers more likely to churn have lower tenure and higher monthly charges.


#### **Design Choices**:
Batch processing of user data for model training and periodic inference.
Reusable components for feature encoding and pipeline deployment, facilitating training and inference across different user cohorts.
Utilization of Airflow for pipeline orchestration, enabling dynamic cohort selection.
Inclusion of additional parameters during inference, such as contract end date, for prioritizing users in campaign targeting.

#### **Pipeline Description**:
Training Pipeline: Data Preprocessing (Clean data, Preprocessing, Train-Test Split) -> Model Training (Logistic Regression) -> Model Evaluation (ROC AUC) -> Model Saving and Metric Tracking via MLFlow.
Inference Pipeline: Similar preprocessing with inference flag set to true to skip unnecessary steps -> Model Loading and Inference Execution. Predictions can be stored in a SQL database.

#### **Model Performance Evaluation**:
Logistic Regression with selected features (['tenure', 'InternetService', 'Dependents', 'TotalCharges', 'MonthlyCharges']) achieved 79% accuracy. However, high VIF values for TotalCharges and tenure indicate correlated features.
Model alone with tenure doesnt really perform well, and 2nd Hypothesis is supported as this gives a better performance of 79% accuracy.

#### **Hyperparameter Tuning and evaluation metrics**:
Precision: 0.82, Recall: 0.93, F1-Score: 0.87 for non-churners, Precision: 0.65, Recall: 0.40, F1-Score: 0.49 for churners, with an overall accuracy of 79%.

#### **Scaling Up the Pipeline Discussion**:
Airflow DAGs can be executed on a Kubernetes cluster to enable large-scale inference on extensive user databases.

#### **Future Work**:
Investigate feature engineering techniques to address multicollinearity.
Explore advanced machine learning models to improve predictive performance.
Implement real-time inference capabilities for immediate campaign deployment.
Enhance scalability by integrating with cloud-based solutions for resource optimization.


++++++ STEPS TO RUN +++++++

```
cd customer_churn
docker-compose build

docker-compose run airflow
docker-compose run mlflow
docker-compose run python
```

Airflow Training and inference pipeline can be scheduled at different time intervals
Based on target customer (cohort type). For monthly contract type -> pipeline can be run weekly or daily and for yearly contract type, same pipeline can be run monthly

<img width="1416" alt="image" src="https://github.com/harshita23sharma/CustomerChurn/assets/16293041/b3149d84-e1fa-4bf4-bcdc-3c2672385aa5">

Training pipeline:
<img width="1435" alt="image" src="https://github.com/harshita23sharma/CustomerChurn/assets/16293041/01f57162-7a2d-4574-81d4-e4f8d05b095b">

Training Logs:
<img width="1423" alt="image" src="https://github.com/harshita23sharma/CustomerChurn/assets/16293041/df48a7c3-9fed-43da-b044-2170e46a3d25">


For tracking the parameters and metrics, MLFLOW has been setup
It can also be used for selecting best model on the fly and deploying the same.




