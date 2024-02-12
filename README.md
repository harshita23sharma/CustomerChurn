# CustomerChurn
End to end ML production pipeline for Customer Churn Prediction 

Machine Learning Case Study: Telco Customer Churn Prediction
Dataset : https://www.kaggle.com/datasets/blastchar/telco-customer-churn

About Dataset:
Customers who left within the last month – the column is called Churn

Hypothesis:
1. Customers who stayed with the company longer (longer tenure) are more likely to stay with the company. OR Customers more likely to churn have lower tenure
2. Customers more likely to churn have higher monthly charges

#### Model & Feature Selection
Based on feature importance of model (based on feature weights), following features were selected for model building: ['Contract', 'tenure', 'InternetService', 'Dependents', 'TotalCharges', 'MonthlyCharges]

vif_data::            feature       VIF
0                      tenure  5.966841
1                      MonthlyCharges  3.235405
2                      TotalCharges  9.593835
3                      Dependents  1.272079
4                      InternetService  1.284656

TotalCharges and tenure are bit correlated


### Finally logistic regression was chosen features
Hyper param tuning
              precision    recall  f1-score   support

           0       0.82      0.93      0.87      1311
           1       0.65      0.40      0.49       447

    accuracy                           0.79      1758
   macro avg       0.73      0.66      0.68      1758
weighted avg       0.77      0.79      0.77      1758

## Model Evaluation Metrics
```
For performance assessment of the chosen models, various metrics are used:

Feature weights: Indicates the top features used by the model to generate the predictions
Confusion matrix: Shows a grid of true and false predictions compared to the actual values
Accuracy score: Shows the overall accuracy of the model for training set and test set
ROC Curve: Shows the diagnostic ability of a model by bringing together true positive rate (TPR) and false positive rate (FPR) for different thresholds of class predictions (e.g. thresholds of 10%, 50% or 90% resulting to a prediction of churn)
AUC (for ROC): Measures the overall separability between classes of the model related to the ROC curve
Precision-Recall-Curve: Shows the diagnostic ability by comparing false positive rate (FPR) and false negative rate (FNR) for different thresholds of class predictions. It is suitable for data sets with high class imbalances (negative values overrepresented) as it focuses on precision and recall, which are not dependent on the number of true negatives and thereby excludes the imbalance
F1 Score: Builds the harmonic mean of precision and recall and thereby measures the compromise between both.
AUC (for PRC): Measures the overall separability between classes of the model related to the Precision-Recall curve
```

++++++

RUN
```
cd customer_churn
docker-compose build python
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




