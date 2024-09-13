
import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
import seaborn as sns
import matplotlib.pyplot as plt
from tkinter import  *
from tkinter import  messagebox
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier , plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics , svm
from sklearn.model_selection import train_test_split , cross_val_score
from sklearn.metrics import accuracy_score , classification_report,confusion_matrix
from sklearn import preprocessing

# Load the dataset
data = pd.read_csv('spam_ham_dataset.csv')

# Drop rows with any missing values
data.dropna(how='any', inplace=True)

# Remove the 'label' column as we'll use 'label_num' for classification
data = data.drop(["label"], axis=1)

# Remove duplicate rows
data = data.drop_duplicates()

# Split data into features (X) and target (Y)
X = data["text"]
Y = data["label_num"]

# Split data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=44)

# Vectorize text data
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Initialize models
LG_model = LogisticRegression(solver='liblinear', C=10, random_state=0)
SV_model = svm.SVC(kernel='linear', C=1, random_state=0)
DT_model = DecisionTreeClassifier(criterion='entropy', random_state=0)
mnb = MultinomialNB(alpha=1.9)
rfc = RandomForestClassifier(n_estimators=100, criterion='gini')

# Train models
LG_model.fit(X_train, Y_train)
SV_model.fit(X_train, Y_train)
DT_model.fit(X_train, Y_train)
mnb.fit(X_train, Y_train)
rfc.fit(X_train, Y_train)

# Model evaluation
models = {
    'Logistic Regression': LG_model,
    'Random Forest': rfc,
    'Support Vector Classifier': SV_model,
    'Naive Bayes': mnb,
    'Desicion Tree': DT_model
}

accuracies = {}
for name, model in models.items():
    Y_pred = model.predict(X_test)
    acc = accuracy_score(Y_test, Y_pred)
    accuracies[name] = acc

# Print accuracies
for name, acc in accuracies.items():
    print(f'{name} Accuracy:', acc)

# GUI
root = Tk()
root.maxsize(800, 800)
root.configure(width="600", height="600", bg="lightblue")
root.minsize(200, 200)
root.title("Email spam detection")

label = Label(root, text="Email spam project ")
label.configure(bg="Lightblue", foreground="white", font=("Arial", 20, "bold"))
label.pack()

label2 = Label(root, text="Enter the Email to detect")
label2.configure(bg="blue", foreground="white", font=("Arial", 15, "bold"))
label2.pack()

my_entry = Entry(root, width=40, foreground="white", bg="gray")
my_entry.pack(pady=10)

def pre():
    text = [my_entry.get()]
    Y_pred = LG_model.predict(vectorizer.transform(text))
    result = "spam" if Y_pred[0] == 1 else "ham"
    messagebox.showinfo("Detection", result)

my_button = Button(text="Detect", command=pre, bg="black", foreground="white", activebackground="gray")
my_button.pack(pady=10)

root.mainloop()
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Visualize the distribution of classes (spam vs. not spam)
plt.figure(figsize=(8, 6))
sns.countplot(x='label_num', data=data)
plt.title('Distribution of Classes')
plt.xlabel('Class')
plt.ylabel('Count')
plt.xticks([0, 1], ['Not Spam', 'Spam'])
plt.show()

# Visualize the accuracy of each model
models = {
    'Logistic Regression': LG_model,
    'Random Forest': rfc,
    'Support Vector Classifier': SV_model,
    'Naive Bayes': mnb,
    'Decision Tree': DT_model
}
accuracies_values = list(accuracies.values())
plt.figure(figsize=(10, 6))
sns.barplot(x=models.keys(), y=accuracies_values)
plt.title('Model Accuracies')
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.ylim(0.9, 1.0)
plt.xticks(rotation=45)
plt.show()

# Confusion matrix for each model
for name, model in models.items():
    Y_pred = model.predict(X_test)
    cm = confusion_matrix(Y_test, Y_pred)

    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title(f"Confusion Matrix - {name}")
    plt.show()

    # ROC curve for each model
    if name != 'Decision Tree':
        fpr, tpr, thresholds = roc_curve(Y_test, Y_pred)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {name}')
        plt.legend(loc="lower right")
        plt.show()

# Decision Tree Visualization
plt.figure(figsize=(15, 10))
plot_tree(DT_model, filled=True)
plt.title("Decision Tree Visualization")
plt.show()
plt.figure(figsize=(10, 6))
feat_importances = pd.Series(rfc.feature_importances_, index=vectorizer.get_feature_names_out())
feat_importances.nlargest(20).plot(kind='barh')
plt.title('Feature Importance - Random Forest')
plt.xlabel('Relative Importance')
plt.ylabel('Feature')
plt.show()

# Plot feature importance for Decision Tree
plt.figure(figsize=(10, 6))
feat_importances = pd.Series(DT_model.feature_importances_, index=vectorizer.get_feature_names_out())
feat_importances.nlargest(20).plot(kind='barh')
plt.title('Feature Importance - Decision Tree')
plt.xlabel('Relative Importance')
plt.ylabel('Feature')
plt.show()

