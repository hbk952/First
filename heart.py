import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
#loading the data
heart=pd.read_csv(r'C:\Users\haris\heart.csv')

#data description
print(heart.head(5))
print(heart.shape)

info = ["age","1: male, 0: female","chest pain type, 1: typical angina, 2: atypical angina, 3: non-anginal pain, 4: asymptomatic","resting blood pressure"," serum cholestoral in mg/dl","fasting blood sugar > 120 mg/dl","resting electrocardiographic results (values 0,1,2)"," maximum heart rate achieved","exercise induced angina","oldpeak = ST depression induced by exercise relative to rest","the slope of the peak exercise ST segment","number of major vessels (0-3) colored by flourosopy","thal: 3 = normal; 6 = fixed defect; 7 = reversable defect"]

for i in range(len(info)):
    print(heart.columns[i]+":\t\t\t"+info[i])

print(heart.describe())

#Visualization
heart.hist(figsize=(14,14))
plt.show()

sns.barplot(x=heart['sex'],y=heart['age'],hue=heart['target'])
plt.show()

sns.barplot(heart["sex"],heart['target'])
plt.show()

target_temp = heart['target'].value_counts()
print(target_temp)

# create a correlation heatmap
numeric_columns=['trestbps','chol','thalach','age','oldpeak']
sns.heatmap(heart[numeric_columns].corr(),annot=True, cmap='terrain', linewidths=0.1)
fig=plt.gcf()
fig.set_size_inches(8,6)
plt.show()

# create four distplots
plt.figure(figsize=(12,12))
plt.subplot(221)
sns.distplot(heart[heart['target']==0].age)
plt.title('Age of patients without heart disease')
plt.subplot(222)
sns.distplot(heart[heart['target']==1].age)
plt.title('Age of patients with heart disease')
plt.subplot(223)
sns.distplot(heart[heart['target']==0].thalach )
plt.title('Max heart rate of patients without heart disease')
plt.subplot(224)
sns.distplot(heart[heart['target']==1].thalach )
plt.title('Max heart rate of patients with heart disease')
plt.show()

#data preprocessing
print(heart['target'].value_counts())
print(heart.isnull().sum())

#Storing in X and y
x, y = heart.iloc[:, :-1], heart.iloc[:, -1]
X_train,X_test,y_train,y_test=train_test_split(x,y,random_state=10,test_size=0.3,shuffle=True)

print ("train_set_x shape: " + str(X_train.shape))
print ("train_set_y shape: " + str(y_train.shape))
print ("test_set_x shape: " + str(X_test.shape))
print ("test_set_y shape: " + str(y_test.shape))

#Random Tree Classifier
clf=RandomForestClassifier(n_estimators=100,random_state=0)
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
accuracy_clf=accuracy_score(y_test,y_pred)*100
print(accuracy_clf)
print("Accuracy on training set: {:.3f}".format(clf.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(clf.score(X_test, y_test)))

#Feature Importance in Decision Trees
print("Feature importances:\n{}".format(clf.feature_importances_))

def plot_feature_importance(model):
    plt.figure(figsize=(8,6))
    n_features = 13
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), x)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    plt.ylim(-1, n_features)
plot_feature_importance(clf)
plt.show()

# KNN
sc=StandardScaler()
sc.fit(X_train)
X_train_std=sc.transform(X_train)
X_test_std=sc.transform(X_test)


k_range=range(1,26)
scores={}
scores_list=[]

for k in k_range:
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_std,y_train)
    prediction_knn=knn.predict(X_test_std)
    scores[k]=accuracy_score(y_test,prediction_knn)
    scores_list.append(accuracy_score(y_test,prediction_knn))

print(scores)

plt.plot(k_range,scores_list)
plt.show()


knn=KNeighborsClassifier(n_neighbors=(scores_list.index(max(scores_list))))
knn.fit(X_train_std,y_train)
y_pred=knn.predict(X_test_std)
accuracy_knn=accuracy_score(y_test,y_pred)*100

print(accuracy_knn)
print("Accuracy on training set: {:.3f}".format(knn.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(knn.score(X_test, y_test)))

#Algo compare
algorithms=['Random Forest','KNN']
scores=[accuracy_clf,accuracy_knn]

sns.set(rc={'figure.figsize':(15,7)})
plt.xlabel("Algorithms")
plt.ylabel("Accuracy score")

sns.barplot(algorithms,scores)
plt.show()

#Classification Report
report=classification_report(y_test,y_pred)
print(report)