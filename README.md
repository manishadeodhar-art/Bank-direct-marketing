# Bank-direct-marketing
Executive Summary
This project evaluates four supervised learning models—Logistic Regression, K‑Nearest Neighbors (KNN), Decision Tree, and Support Vector Classifier (SVC)—to determine their effectiveness in predicting customer subscription to a term deposit as part of a bank’s telemarketing campaign. Using the Bank Marketing dataset from the UCI Machine Learning Repository, the analysis compares each model under default settings and after hyperparameter tuning, focusing on training time, training accuracy, and test accuracy.

Across the evaluation, default models consistently outperformed tuned models in predictive accuracy, with Logistic Regression and SVC emerging as the strongest performers. Logistic Regression demonstrated excellent generalization with nearly identical train and test accuracy, while SVC delivered the highest test accuracy overall, albeit with significantly longer training time. KNN and Decision Tree also performed well in their default configurations, though the Decision Tree showed clear signs of overfitting. Hyperparameter tuning reduced overfitting in the Decision Tree and improved training efficiency for SVC, but for most models, tuning introduced higher computational cost and lower accuracy.

The findings highlight that tuning does not always yield performance gains and must be applied thoughtfully, especially when working with imbalanced datasets. For practical deployment, Logistic Regression (default) offers the best balance of accuracy, efficiency, and interpretability, making it the most suitable model for supporting targeted marketing decisions. SVC remains a strong alternative when maximum accuracy is prioritized and computational resources are less constrained.

Project Summary: Classifier Comparison for Bank Marketing Dataset
1. Overview
This project evaluates and compares the performance of four supervised machine learning classifiers—Logistic Regression, K‑Nearest Neighbors (KNN), Decision Tree, and Support Vector Classifier (SVC)—using the Bank Marketing dataset. The goal is to understand how each model performs under default settings and how performance changes after hyperparameter tuning. The analysis includes accuracy, generalization behavior, and computational efficiency.

2. Business Objective
Banks frequently run telemarketing campaigns to promote term deposit subscriptions. The business objective of this project is to predict whether a customer will subscribe to a term deposit, enabling the bank to:

    Prioritize high‑probability leads

    Reduce operational costs

    Improve campaign efficiency

    Increase conversion rates

    Accurate classification models can significantly enhance targeting strategies and overall marketing ROI.

3. Data Source
The dataset used in this project is the Bank Marketing Dataset from the UCI Machine Learning Repository.
It contains information collected from direct marketing campaigns conducted by a Portuguese banking institution.

4. Key Characteristics of the Data
Rows: ~45,000 customer records

Target Variable: y (binary: yes or no for term deposit subscription)

Features include:

Demographics: age, job, marital status, education

Financial attributes: balance, loan status, housing loan

Campaign details: contact type, number of contacts, previous outcomes

Economic indicators: employment variation rate, consumer confidence index

Class Imbalance:  
The dataset is highly imbalanced, with the majority of customers not subscribing. This makes accuracy alone an unreliable metric and motivates the use of recall, F1, and AUC in extended evaluations.

5. Methodology
The project follows a structured machine‑learning workflow:

5.1 Data Preparation
Loaded and cleaned dataset

Encoded categorical variables

Split into training and testing sets

Standardized numerical features where appropriate

5.2 Baseline Modeling
Trained four default models:

Logistic Regression

KNN

Decision Tree

SVC

Captured:

Training time

Training accuracy

Test accuracy
5.3 Comparative Analysis
Compared:

Default vs tuned performance

Accuracy vs generalization

Training time trade‑offs

6. Observations
6.1 Default Models
SVC (Default) achieved the highest test accuracy (0.9020).

Logistic Regression (Default) showed strong generalization with nearly equal train/test accuracy.

Decision Tree (Default) severely overfit (train: 0.9954, test: 0.8384).

KNN (Default) performed well with minimal training time.

6.2 Tuned Models
Tuning reduced accuracy for most models (Logistic Regression, KNN, SVC).

Decision Tree (Tuned) improved test accuracy (0.8495), showing reduced overfitting.

Training time increased significantly for Logistic Regression and KNN.

SVC (Tuned) became dramatically faster (57.6 sec → 2.03 sec) but lost accuracy.

6.3 Default vs Tuned Comparison
Default models consistently outperformed tuned models in accuracy, except for the Decision Tree.

Tuning often increased computational cost without improving performance.

The only meaningful gain from tuning was better generalization in the Decision Tree and faster training for SVC.

7. Findings
Accuracy: Default SVC and Logistic Regression are the strongest performers.

Generalization: Logistic Regression (Default) shows the most stable behavior.

Efficiency: KNN and Logistic Regression train extremely fast in default form.

Impact of Tuning:

Did not improve accuracy for most models

Increased training time substantially

Helped reduce overfitting only in the Decision Tree

Overall, tuning did not yield the expected performance improvements and often introduced unnecessary computational overhead.

8. Recommendation
Based on accuracy, generalization, and efficiency, the recommended model for deployment is Logistic Regression (Default). It offers:

Strong and balanced train/test accuracy

Fast training time

Excellent generalization

Interpretability for business stakeholders

Stability across different configurations

If the business prioritizes maximum accuracy and can tolerate long training times, SVC (Default) is a strong alternative.
If interpretability and operational efficiency are priorities, Logistic Regression remains the best choice.
