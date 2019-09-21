from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
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


for x in range(1,50):
	neigh8 = KNeighborsClassifier(n_neighbors=x)
	print(neigh8)
	neigh8.fit(X, y) 
	prediction = neigh.predict([[2,4,3,1]])
	prediction = neigh.predict([[2,4,3,1],[6,5,3,4]])
	print(prediction)



clf = LogisticRegression()
clf.fit(X,y)
prediction_logestic = clf.predict([[5.1,3.5,1.4,0.1]])
print(prediction_logestic)
prediction_logestic 	 = neigh.predict([[2,4,3,1],[6,5,3,4]])
print(prediction_logestic)
