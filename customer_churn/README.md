# CustomerChurn
End to end ML production pipeline for Customer Churn Prediction 
```

inside customer_churn:
mlflow server --backend-store-uri mlflow/ --artifacts-destination mlflow/ --port 8000


cd customer_churn/airflow
pip install apache-airflow==2.8.1 --constraint https://raw.githubusercontent.com/apache/airflow/constraints-2.8.1/constraints-3.10.txt
export AIRFLOW_HOME=/Users/harshita/Documents/github_repos/CustomerChurn/customer_churn/airflow
airflow db migrate
airflow db init
airflow users create --username harshita --password harshita --firstname harshita --lastname sharma --role Admin --email harshita23sharma@gmail.com
airflow standalone
```