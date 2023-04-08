import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


train = pd.read_csv("titanic_train.csv")

#Check Data(summary)
print(train.head())

#print(train.info()) 

#Exploratory Data Analysis
#heatmap to show the distribution of data
#sns.heatmap(train.isnull(), yticklabels = False, cbar = False, cmap = 'viridis')
'''
missing some age information(this info can be fixed later due to small number missing).
missing a lot of cabin information which may have to be removed later(too much missing info)
'''

#exploring survived column
#sns.set_style('whitegrid')
#sns.countplot(x = 'Survived', data = train, palette = 'RdBu_r')*
'''
There were significantly less Survivors
'''

#diving into the the demographics of survivors
#distribution of survivor between sex
#sns.countplot(x = 'Survived', data = train, hue = 'Sex', palette = 'RdBu_r')*
'''
Vastly more males died than females
'''

#distribution of survivors by class
#sns.countplot(x = 'Survived' , data = train, hue = 'Pclass', palette = 'rainbow')*
'''
Most people in the 3rd class did not survive compared to those in 1st and 2nd.
It also looks as though there were vastly more people in 3rd class as compared to the other classes(This might skew the data).
'''

#age distribution of passengers that survived
#sns.countplot(x = 'Age' , data = train, hue = 'Survived', palette = 'RdBu_r')*
'''
We can also infere Age distribution from this as well: Most people on the Titanic were between the ages of 16 to 36
This age range also had the most number of casualties.
'''

################################################################################################################################################################

#Data Cleaning
#From the analysis we noticed we were missing some data in the Age, Cabin and Embarked columns respectively, I will address them here
#Age => using imputation we can either take the average age of passengers and fill empty cells or in this case fill the empty cells by the avg age of passengers by class
#lets visualize this information with a box plot 
sns.boxplot(x = 'Pclass', y = 'Age', data = train, palette = 'winter')
#plt.show()
'''
First class passengers are generally older than the other 2 classes and the avg age in this class is about 37yrs
Second class passangers are slightly older than 3rd class passengers. The avg age in this class is about 29yrs
Third class passengers are the youngest group of passengers. The avg age in this class is about 24yrs
NB: this are estimated based on the boxplot
'''

#Due to the small number of missing ages, we can fill that column with avgs using function
def impute_age(cols):
    Age    = cols[0]
    Pclass = cols[1]

    if pd.isnull(Age):

        if Pclass == 1:
            return 37

        elif Pclass == 2:
            return 29

        else:
            return 24

    else:
        return Age


#call function => impute_age
train['Age'] = train[['Age', 'Pclass']].apply(impute_age, axis = 1)

#There is a significant number of cabin info missing which could be filled in but may skew te data greatly(I am going to drop this column)
train.drop('Cabin', axis = 1, inplace = True)

#We need to convert and/or drop categorial features within the dataframe(Sex, Name, Ticket, Embarked etc)
sex = pd.get_dummies(train['Sex'], drop_first = True) # we dropped one clolumn
embark = pd.get_dummies(train['Embarked'], drop_first = True)

train.drop(['Sex', 'Embarked', 'Name', 'Ticket'], axis = 1, inplace = True)

train = pd.concat([train, sex, embark], axis = 1)
print(train.info()) #age is now good to go



#############################################################################################################################################################
#Building Logistic Regression Model
#Splitting the data infto training and test splits
X = train.drop('Survived', axis = 1)
y = train['Survived']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3, random_state = 101)


#Train the model on the data
from sklearn.linear_model import LogisticRegression
LogModel = LogisticRegression()
LogModel.fit(X_train, y_train)

#Prediction
Prediction = LogModel.predict(X_test)

#Evaluation
from sklearn.metrics import classification_report
print(classification_report(y_test,Prediction))


#Some Error were encountered with the scikit learn library will need to investigate
'''
Error/Warning Summary:

opt/anaconda3/lib/python3.7/site-packages/sklearn/linear_model/_logistic.py:940: ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)
              precision    recall  f1-score   support
'''