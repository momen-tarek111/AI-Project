# AI_project

1. Introduction:
The objective of this project is to classify emails as "spam" or "not spam" based on text features such as word frequencies, presence of spam keywords, and sender information. We aim to build and evaluate multiple classification models using machine learning algorithms.

2. Data Preprocessing:

The dataset used for this project is "spam_ham_dataset.csv".
We performed the following preprocessing steps:
Removed rows with missing values.
Dropped the 'label' column and kept only the 'label_num' column for classification.
Eliminated duplicate rows to ensure data cleanliness.
3. Feature Selection:

We employed CountVectorizer to convert text data into numerical features.
Further feature selection techniques such as correlation analysis or feature importance were not explicitly applied in this project.
4. Classification Models:

Logistic Regression:

Hyperparameters: Solver='liblinear', C=10.
Explanation: Logistic Regression is a linear classification algorithm that predicts the probability of a binary outcome. In this model, the regularization parameter C is set to 10 to control overfitting.
Random Forest Classifier:

Hyperparameters: n_estimators=100, criterion='gini'.
Explanation: Random Forest is an ensemble learning method that constructs multiple decision trees during training. The number of trees (n_estimators) is set to 100, and the Gini impurity criterion is used for splitting nodes.
Support Vector Classifier (SVM):

Hyperparameters: Kernel='linear', C=1.
Explanation: SVM is a powerful classification algorithm that finds the hyperplane that best separates classes in a high-dimensional space. We use a linear kernel and set the regularization parameter C to 1 to balance margin width and classification error.
Naive Bayes:

Hyperparameters: Alpha=1.9.
Explanation: Naive Bayes is a probabilistic classifier based on Bayes' theorem. The Laplace smoothing parameter alpha is set to 1.9 to handle unseen features in the test data effectively.
Decision Tree Classifier:

Hyperparameters: Criterion='entropy'.
Explanation: Decision Tree is a simple yet effective classification algorithm that creates a tree-like model of decisions. The criterion is set to 'entropy' to measure the quality of splits based on information gain.
5. Model Training and Evaluation:

Each model was trained on the training set and evaluated on the testing set.
Evaluation metrics used:
Accuracy: Measures the overall correctness of the model.
6. Results:

Accuracy of each model on the testing set:
Logistic Regression: 0.981
Random Forest Classifier: 0.969
Support Vector Classifier: 0.967
Naive Bayes: 0.969
Decision Tree Classifier: 0.946
7. Conclusion:

The Logistic Regression model achieved the highest accuracy of 98.1%, demonstrating superior performance in classifying emails as spam or not spam.
Random Forest Classifier and Naive Bayes models also performed well, with accuracies of 96.9%.
Decision Tree Classifier had the lowest accuracy among the models, indicating that it may not generalize as effectively to unseen data.
