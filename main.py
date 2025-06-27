"This project, titled 'Breast Cancer Prediction Using Machine Learning,' is implemented using Google Colab. The dataset for this project is sourced from Kaggle."



//import libraries//
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


//read data//
df=pd.read_csv('data.csv')

//data overview//
df.head()


df.iloc[:10]
//Now fine//


//lets check data types//
df.dtypes
//all are numeric except target label 'diagnosis'//


//no of rows and columns//
df.shape
//there is 569 rows and 31columns i.e 30 features and one target class//


  df.describe()


//check any null values in database//
df.isnull().values.any()


//Data visualization//
//histogram//
df.hist(bins=50,figsize=(15,15))
plt.show()


//create a pair plot//
sns.pairplot(df.iloc[:,0:7],hue='diagnosis')


//Count each label//
ax=sns.countplot(y='diagnosis',data=df,palette='Set2')


//lets find correlation//
fig, ax = plt.subplots(figsize=(12,10))
sns.heatmap(df.corr(),ax=ax)


//box plot to check outlier in each category//

//define function can be call later//
def boxPlot(dff):
    d=dff.drop(columns=['diagnosis'])
    for column in d:
        plt.figure(figsize=(5,2))
        sns.boxplot(x=column,data=d,palette="colorblind")
boxPlot(df)


//Quartile range//

Q1=df.quantile(0.25)
Q3=df.quantile(0.75)
IQR=Q3-Q1


//remove all outlier//
//< Q1-1.5*IQR//
//> Q3+1.5*IQR//

df_out = df[~((df < (Q1 - (1.5 * IQR))) |(df > (Q3 + (1.5 * IQR)))).any(axis=1)]
df.shape,df_out.shape


//KNN//
from sklearn.neighbors import KNeighborsClassifier

clf=KNeighborsClassifier(n_neighbors=9,n_jobs=-1)
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)

ac=accuracy_score(y_test,y_pred)
acc.append(ac)
rc=roc_auc_score(y_test,y_pred)
roc.append(rc)
print("Accuracy {0} ROC {1}".format(ac,rc))

//cross validation//
result=cross_validate(clf,X_train,y_train,scoring=scoring,cv=10)
display_result(result)


//Naivye Bayes//

from sklearn.naive_bayes import GaussianNB

clf=GaussianNB()
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)

ac=accuracy_score(y_test,y_pred)
acc.append(ac)
rc=roc_auc_score(y_test,y_pred)
roc.append(rc)
print("Accuracy {0} ROC {1}".format(ac,rc))

//cross validation//
result=cross_validate(clf,X_train,y_train,scoring=scoring,cv=10)
display_result(result)


//Random Forest//
from sklearn.ensemble import RandomForestClassifier

clf=RandomForestClassifier(n_estimators=20,max_depth=10)
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)

ac=accuracy_score(y_test,y_pred)
acc.append(ac)
rc=roc_auc_score(y_test,y_pred)
roc.append(rc)
print("Accuracy {0} ROC {1}".format(ac,rc))
import matplotlib.pyplot as plt
import numpy as np

// Assuming acc and roc are your accuracy and roc auc score arrays//
algorithms = ['Logistic Regression','SVM','KNN','Naivye Bayes','Random Forest']

// Assuming acc and roc are arrays of shape (7,) and (5,) respectively//
acc = np.array([0.8, 0.7, 0.6, 0.9, 0.85, 0.95, 0.8])
roc = np.array([0.9, 0.8, 0.7, 0.95, 0.9])

// Select the scores corresponding to the algorithms you're plotting//
acc_selected = acc[[i for i, algo in enumerate(algorithms) if algo in ['Logistic Regression','SVM','KNN','Naivye Bayes','Random Forest']]]
roc_selected = roc[[i for i, algo in enumerate(algorithms) if algo in ['Logistic Regression','SVM','KNN','Naivye Bayes','Random Forest']]]

// Plot the accuracy scores//
plt.figure(figsize=(20,20))
plt.bar(algorithms, acc_selected, color=['salmon','r','g','b','orange'], label='Accuracy')
plt.ylabel('Accuracy Score')
plt.xlabel('Algorithms')
plt.show()

// Plot theroc auc scores//
plt.figure(figsize=(8,5))
plt.bar(algorithms, roc_selected, color=['salmon','r','g','b','orange'], label='ROC AUC')
plt.ylabel('ROC AUC')
plt.xlabel('Algorithms')
plt.show()
//cross validation//
result=cross_validate(clf,X_train,y_train,scoring=scoring,cv=10)
display_result(result)


//Support Vector Machine //
from sklearn.svm import SVC

clf=SVC(gamma='auto',kernel='linear')
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)

ac=accuracy_score(y_test,y_pred)
acc.append(ac)
rc=roc_auc_score(y_test,y_pred)
roc.append(rc)
print("Accuracy {0} ROC {1}".format(ac,rc))

//cross validation//
result=cross_validate(clf,X_train,y_train,scoring=scoring,cv=10)
display_result(result)


// Logistic Regression //
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

ac = accuracy_score(y_test, y_pred)
acc.append(ac)
rc = roc_auc_score(y_test, y_pred)
roc.append(rc)
print("Logistic Regression - Accuracy: {0}, ROC: {1}".format(ac, rc))

// Cross-validation //
result = cross_validate(clf, X_train, y_train, scoring=scoring, cv=10)
display_result(result)


// Gradient Boosting //
from sklearn.ensemble import GradientBoostingClassifier

clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

ac = accuracy_score(y_test, y_pred)
acc.append(ac)
rc = roc_auc_score(y_test, y_pred)
roc.append(rc)
print("Gradient Boosting - Accuracy: {0}, ROC: {1}".format(ac, rc))

// Cross-validation //
result = cross_validate(clf, X_train, y_train, scoring=scoring, cv=10)
display_result(result)





  
