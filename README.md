# BITS-ML-AS-2-V1.1
BITS MTECH AIML Semester 1 ML assignment 2 (15-Feb-2026)

a. Problem statement
====================
Step 1: Dataset choice
Choose ONE classification dataset of your choice from any public repository -
Kaggle or UCI. It may be a binary classification problem or a multi-class
classification problem.
Minimum Feature Size: 12
Minimum Instance Size: 500
Step 2: Machine Learning Classification models and Evaluation metrics
Implement the following classification models using the dataset chosen above. All
the 6 ML models have to be implemented on the same dataset.
1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbor Classifier
4. Naive Bayes Classifier - Gaussian or Multinomial
5. Ensemble Model - Random Forest
6. Ensemble Model - XGBoost
For each of the models above, calculate the following evaluation metrics:
1. Accuracy
2. AUC Score
3. Precision
4. Recall
5. F1 Score
6. Matthews Correlation Coeﬃcient (MCC Score)

b. Dataset description [ 1 mark ]
====================================
Seven different types of dry bean’s features such as form, shape, type, and structure are present. And a total of 16 features; 12 dimensions and 4 shape forms, were obtained from the grains.
The Dry Bean dataset was selected because it is a well-known multiclass classification
dataset with sufficient features and instances to evaluate multiple machine learning
algorithms. Its clean tabular structure and balanced classes make it suitable for
comparing model performance using metrics such as Accuracy, AUC, Precision, Recall,
F1 Score, and MCC.

c. Models used: [ 6 marks - 1 marks for all the metrics for each model ]
========================================================================
Make a Comparison Table with the evaluation metrics calculated for all the 6
models as below:

| Model               | Accuracy   | Precision  | Recall    | F1 Score  | MCC        | ROC AUC   |
|---------------------|-----------:|-----------:|----------:|----------:|-----------:|----------:|
| Logistic Regression | 0.9218903  | 0.9224989  | 0.9218903 | 0.9220339 | 0.90563999 | 0.9947419 |
| Decision Tree       | 0.8929971  | 0.8927592  | 0.8929971 | 0.8927802 | 0.87057570 | 0.9438949 |
| KNN                 | 0.9174829  | 0.9178084  | 0.9174829 | 0.9175374 | 0.90019629 | 0.9845730 |
| Naive Bayes         | 0.8969148  | 0.8993155  | 0.8969148 | 0.8969500 | 0.87606021 | 0.9918335 |
| Random Forest       | 0.9209109  | 0.9209190  | 0.9209109 | 0.9208589 | 0.90435128 | 0.9918000 |
| XGBoost             | 0.9209109  | 0.9210542  | 0.9209109 | 0.9208573 | 0.90432187 | 0.9940016 |

- Add your observations on the performance of each model on the chosen
dataset. [ 3 marks ]

| ML Model | Observation about model performance |
|----------|------------------------------------|
| Logistic Regression | Reliable and stable performance. High overall performance. |
| Decision Tree | Lowest accuracy and MCC, AUC is lower, weaker generalization. |
| kNN | High AUC means strong class discrimination. |
| Naive Bayes | Very high AUC despite lower Accuracy. |
| Random Forest (Ensemble) | Ensemble reduces overfitting, one of the top performers. |
| XGBoost (Ensemble) | Performance close to Random Forest, highest AUC among ensemble models. |

Streamlit URL:
https://bits-ml-as-2-v11-afwxmwqthlowzsodwczcs6.streamlit.app/

Note:
Instead of XGBoost, used Gradient Boosting. Kept the naming as XGBoost due to proble requirement. 
XGBoost should be read as Grandient Boosting through out all the documents, and soruce codes.
Reason: As I am using the office laptop, the security policy does not allow to use brew to install the dependencies required for the package XGBoost. Hence, used Gradient Boosting.