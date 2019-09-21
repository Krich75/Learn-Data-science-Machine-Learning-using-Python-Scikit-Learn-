from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt


iris = load_iris()
print(type(iris))
print(iris.feature_names)
print(iris.target_names)
print(iris.target)
print(type(iris.data))
print(type(iris.target))

X = iris.data
y = iris.target

print(X.shape)
print(y.shape)
neigh = KNeighborsClassifier(n_neighbors=1)
neigh.fit(X, y) 
print(neigh)
prediction = neigh.predict([[2,4,3,1]])
print(prediction)
print(iris.target_names)
prediction = neigh.predict([[2,4,3,1],[6,5,3,4]])
print(prediction)
neigh5 = KNeighborsClassifier(n_neighbors=5)
neigh5.fit(X, y) 
prediction = neigh.predict([[2,4,3,1]])
prediction = neigh.predict([[2,4,3,1],[6,5,3,4]])
print(prediction)






clf = LogisticRegression()
clf.fit(X,y)
prediction_logestic = clf.predict([[5.1,3.5,1.4,0.1]])
print(prediction_logestic)
prediction_logestic 	 = neigh.predict([[2,4,3,1],[6,5,3,4]])


print(prediction_logestic)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

print(X.shape)
print(X_test.shape,X_train.shape)
print(y.shape)
print(y_test.shape,y_train.shape)

logisticreg = LogisticRegression()

logisticreg.fit(X_test,y_test)

y_pred = logisticreg.predict(X_test)

print(metrics.accuracy_score(y_test,y_pred))


neigh5 = KNeighborsClassifier(n_neighbors=5)
neigh5.fit(X_test,y_test) 
y_pred = neigh5.predict(X_test)
print("KNN WITH 5")
print(metrics.accuracy_score(y_test,y_pred))

neigh = KNeighborsClassifier(n_neighbors=1)
scores = cross_val_score(neigh,X, y, cv=10, scoring = "accuracy")
print(scores)
k_scores = []
k_range = (1,50)
for x in k_range:
	neigh8 = KNeighborsClassifier(n_neighbors=x)
	scores = cross_val_score(neigh,X, y, cv=10, scoring = "accuracy")
	k_scores.append(scores.mean())
	
print(k_scores)

plt.plot(k_range,k_scores)
plt.xlabel('Accuracy')
plt.ylabel('KNN n_neighbors')


plt.show()
