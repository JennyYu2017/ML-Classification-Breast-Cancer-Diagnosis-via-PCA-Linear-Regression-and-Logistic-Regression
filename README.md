# Machine Learning--Breast Cancer Diagnosis via Different Machine Learning Technique(PCA, Linear Regression and Logistics Regression)

The goal of this project is to compare the performance of different Machine Learning models for automatic diagnosis of breast cancer.

In this project, the dataset contains 30 carefully selected features from each of ​569 patients. The same dataset has also been made available from the UCI
Machine Learning Repository(https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+%28original%29).

# Programming Language
Matlab

# Conclusion
![Performance Comparation of Different Machine Learning Models](https://github.com/JennyYu2017/Machine-Learning--Breast-Cancer-Diagnosis-via-Different-Machine-Learning-Techniques/blob/master/Performance%20Comparation%20on%20Different%20Machine%20Learning%20Models.png)

Overall speaking, both PCA and Logistic Regression(applying BFGS) perform well with small error rate for the problem. However, PCA is more efficient in terms of CPU time in both training stage and testing stage compared to BFGS. 

Linear regression also perform good with only a minor greater error rate compared to PCA and BFGS. 

Logistic Regression(applying NAG)’s result is not satisfying.It is error rate is relatively high comparing to other models. 
