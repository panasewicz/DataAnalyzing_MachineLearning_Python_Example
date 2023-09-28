import sklearn
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
import seaborn as sns
import matplotlib.pyplot as plt


iris = load_iris(return_X_y=False, as_frame=True)
# print(data.target[[10, 25, 50]])
# print(list(data.target_names))

print("Object attributes load_iris: " + str(dir(iris)))

data = iris.data
rows_number = data.shape[0]
columns_number = data.shape[1]

print("Number of data rows: " + str(rows_number))
print("Number of data columns: " + str(columns_number))

target = iris.target

df = pd.DataFrame(data, columns=iris.feature_names)
df['target'] = target

#df.to_csv('iris.csv', index=False)

#sns.scatterplot(x='sepal length (cm)', y='sepal width (cm)', hue='target', data=df)
sns.pairplot(df, hue='target')
plt.show()

#DIVIDING DATASET FOR TRAIN AND TEST DATA WITH 4:1 RATIO
X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(iris.data, iris.target, test_size=0.2) #random_state=0)

print(f'Number of samples in the training set: {len(X_train)}')
print(f'Number of samples in the test set: {len(X_test)}')


# KNN Training Method
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, Y_train)
y_knn_pred = knn.predict(X_test)

# KNN Cross-Validation
knn_scores = cross_val_score(knn, data, target, cv=5)

# SVM Training Method
svm = SVC(kernel='linear')
svm.fit(X_train, Y_train)
y_svm_pred = svm.predict(X_test)

# SVM Cross-Validation
svm_scores = cross_val_score(svm, data, target, cv=5)

# Decision Tree Training Method
dtree = DecisionTreeClassifier()
dtree.fit(X_train, Y_train)
y_dtree_pred = dtree.predict(X_test)

# Decision Tree Cross-Validation
dtree_scores = cross_val_score(dtree, data, target, cv=5)

print("METHODS REPORT KNN: ")
print(classification_report(Y_test, y_knn_pred))

print("METHODS REPORT SVM: ")
print(classification_report(Y_test, y_svm_pred))

print("METHODS REPORT DECISION TREE: ")
print(classification_report(Y_test, y_dtree_pred))


print("CROSS-VALIDATION KNN: ")
print("Prediction accuracy KNN: %0.2f (+/- %0.2f)" % (knn_scores.mean(), knn_scores.std()*2))

print("CROSS-VALIDATION SVM: ")
print("Prediction accuracy SVM: %0.2f (+/- %0.2f)" % (svm_scores.mean(), svm_scores.std()*2))

print("CROSS-VALIDATION DECISION TREE: ")
print("Prediction accuracy dtree: %0.2f (+/- %0.2f)" % (dtree_scores.mean(), dtree_scores.std()*2))

fig, axs = plt.subplots(1, 3, figsize=(12, 4))

axs[0].scatter(X_test["sepal length (cm)"], X_test["sepal width (cm)"], c=Y_test, label='KNN')
axs[0].scatter(X_test["sepal length (cm)"], X_test["sepal width (cm)"], c=y_knn_pred, marker='x', label='KNN')
axs[0].set_xlabel('sepal length (cm)')
axs[0].set_ylabel('sepal width (cm)')
axs[0].legend(['Test', 'Predicted'])
axs[0].set_title("Split KNN")
axs[1].scatter(X_test["sepal length (cm)"], X_test["sepal width (cm)"], c=Y_test, label='SVM')
axs[1].scatter(X_test["sepal length (cm)"], X_test["sepal width (cm)"], c=y_svm_pred, marker='x', label='SVM')
axs[1].set_xlabel('sepal length (cm)')
axs[1].set_ylabel('sepal width (cm)')
axs[1].legend(['Test', 'Predicted'])
axs[1].set_title("Split SVM")
axs[2].scatter(X_test["sepal length (cm)"], X_test["sepal width (cm)"], c=Y_test, label='dTree')
axs[2].scatter(X_test["sepal length (cm)"], X_test["sepal width (cm)"], c=y_dtree_pred, marker='x', label='dTree')
axs[2].set_xlabel('sepal length (cm)')
axs[2].set_ylabel('sepal width (cm)')
axs[2].legend(['Test', 'Predicted'])
axs[2].set_title("Split dTree")

plt.show()

