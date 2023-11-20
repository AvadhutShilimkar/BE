1]

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df  = pd.read_csv("uber.csv")

df = df.drop(['Unnamed: 0', 'key'], axis= 1)
df.head()
df.isnull().sum() 

df['dropoff_latitude'].fillna(value=df['dropoff_latitude'].mean(),inplace = True)
df['dropoff_longitude'].fillna(value=df['dropoff_longitude'].median(),inplace = True)

df.pickup_datetime = pd.to_datetime(df.pickup_datetime, errors='coerce') 

df= df.assign(hour = df.pickup_datetime.dt.hour,
             day= df.pickup_datetime.dt.day,
             month = df.pickup_datetime.dt.month,
             year = df.pickup_datetime.dt.year,
             dayofweek = df.pickup_datetime.dt.dayofweek)

df = df.drop('pickup_datetime',axis=1)

df.plot(kind = "box",subplots = True,layout = (7,2),figsize=(15,20))

#Using the InterQuartile Range to fill the values
def remove_outlier(df1 , col):
    Q1 = df1[col].quantile(0.25)
    Q3 = df1[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_whisker = Q1-1.5*IQR
    upper_whisker = Q3+1.5*IQR
    df[col] = np.clip(df1[col] , lower_whisker , upper_whisker)
    return df1

def treat_outliers_all(df1 , col_list):
    for c in col_list:
        df1 = remove_outlier(df , c)
    return df1

df = treat_outliers_all(df , df.iloc[: , 0::])

df.plot(kind = "box",subplots = True,layout = (7,2),figsize=(15,20))

#pip install haversine
import haversine as hs  #Calculate the distance using Haversine to calculate the distance between to points. Can't use Eucladian as it is for flat surface.
travel_dist = []
for pos in range(len(df['pickup_longitude'])):
        long1,lati1,long2,lati2 = [df['pickup_longitude'][pos],df['pickup_latitude'][pos],df['dropoff_longitude'][pos],df['dropoff_latitude'][pos]]
        loc1=(lati1,long1)
        loc2=(lati2,long2)
        c = hs.haversine(loc1,loc2)
        travel_dist.append(c)
    
print(travel_dist)
df['dist_travel_km'] = travel_dist
df.head()

#Uber doesn't travel over 130 kms so minimize the distance 
df= df.loc[(df.dist_travel_km >= 1) | (df.dist_travel_km <= 130)]
print("Remaining observastions in the dataset:", df.shape)




#Finding inccorect latitude (Less than or greater than 90) and longitude (greater than or less than 180)
incorrect_coordinates = df.loc[(df.pickup_latitude > 90) |(df.pickup_latitude < -90) |
                                   (df.dropoff_latitude > 90) |(df.dropoff_latitude < -90) |
                                   (df.pickup_longitude > 180) |(df.pickup_longitude < -180) |
                                   (df.dropoff_longitude > 90) |(df.dropoff_longitude < -90)
                                    ]

df.drop(incorrect_coordinates, inplace = True, errors = 'ignore')

sns.heatmap(df.isnull())

corr = df.corr()

fig,axis = plt.subplots(figsize = (10,6))
sns.heatmap(df.corr(),annot = True)

x = df[['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude','passenger_count','hour','day','month','year','dayofweek','dist_travel_km']]
y = df['fare_amount']

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(x,y,test_size = 0.33)

from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(X_train,y_train)
regression.intercept_
regression.coef_
prediction = regression.predict(X_test)
print(prediction)
y_test

from sklearn.metrics import r2_score 
r2_score(y_test,prediction)
from sklearn.metrics import mean_squared_error
MSE = mean_squared_error(y_test,prediction)
RMSE = np.sqrt(MSE)

from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=100) 
rf.fit(X_train,y_train)
y_pred = rf.predict(X_test)
y_pred
R2_Random = r2_score(y_test,y_pred)
MSE_Random = mean_squared_error(y_test,y_pred)
RMSE_Random = np.sqrt(MSE_Random)


2]

pip install sympy

import matplotlib as plot
import numpy as np
import sympy as sym 
from matplotlib import pyplot

def objective(x):
    return (x+3)**2

def derivative(x):
    return 2*(x + 3)

def gradient_descent(alpha, start, max_iter):
    x_list = list()
    x= start;
    x_list.append(x)
    for i in range(max_iter):
        gradient = derivative(x);
    x = x - (alpha*gradient);
    x_list.append(x);
    return x_list

x = sym.symbols('x')
expr = (x+3)**2.0;
grad = sym.Derivative(expr,x)
print("{}".format(grad.doit()) )
grad.doit().subs(x,2)

def gradient_descent1(expr,alpha, start, max_iter):
    x_list = list()
    x = sym.symbols('x')
    grad = sym.Derivative(expr,x).doit()
    x_val= start;
    x_list.append(x_val)
    for i in range(max_iter):
        gradient = grad.subs(x,x_val);
        x_val = x_val - (alpha*gradient);
        x_list.append(x_val);
    return x_list

alpha = 0.1 #Step_size
start = 2 #Starting point
max_iter = 30 #Limit on iterations
x = sym.symbols('x')
expr = (x+3)**2; #target function

x_cordinate = np.linspace(-15,15,100)
pyplot.plot(x_cordinate,objective(x_cordinate))
pyplot.plot(2,objective(2),'ro')

X = gradient_descent(alpha,start,max_iter)
x_cordinate = np.linspace(-5,5,100)
pyplot.plot(x_cordinate,objective(x_cordinate))
X_arr = np.array(X)
pyplot.plot(X_arr, objective(X_arr), '.-', color='red')
pyplot.show()

X= gradient_descent1(expr,alpha,start,max_iter)
X_arr = np.array(X)
x_cordinate = np.linspace(-5,5,100)
pyplot.plot(x_cordinate,objective(x_cordinate))
X_arr = np.array(X)
pyplot.plot(X_arr, objective(X_arr), '.-', color='red')
pyplot.show()


3]

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import metrics

df=pd.read_csv('diabetes.csv')

X = df.drop('Outcome',axis = 1)
y = df['Outcome']

from sklearn.preprocessing import scale
X = scale(X)
# split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=7)
 
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

print("Confusion matrix: ")
cs = metrics.confusion_matrix(y_test,y_pred)
print(cs)

print("Acccuracy ",metrics.accuracy_score(y_test,y_pred))

total_misclassified = cs[0,1] + cs[1,0]
print(total_misclassified)
total_examples = cs[0,0]+cs[0,1]+cs[1,0]+cs[1,1]
print(total_examples)
print("Error rate",total_misclassified/total_examples)
print("Error rate ",1-metrics.accuracy_score(y_test,y_pred))

print("Precision score",metrics.precision_score(y_test,y_pred))
print("Recall score ",metrics.recall_score(y_test,y_pred))
print("Classification report ",metrics.classification_report(y_test,y_pred))


4]

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, k_means #For clustering
from sklearn.decomposition import PCA #Linear Dimensionality reduction.
df = pd.read_csv("sales_data_sample.csv") #Loading the dataset.

df.head()
df.shape 
df.describe()
df.info()
df.isnull().sum()

df_drop  = ['ADDRESSLINE1', 'ADDRESSLINE2', 'STATUS','POSTALCODE', 'CITY', 'TERRITORY', 'PHONE', 'STATE', 'CONTACTFIRSTNAME', 'CONTACTLASTNAME', 'CUSTOMERNAME', 'ORDERNUMBER']
df = df.drop(df_drop, axis=1) #Dropping the categorical uneccessary columns along with columns having null values. Can't fill the null values are there are alot of null values.

df.isnull().sum()
df.dtypes

df['COUNTRY'].unique()
df['PRODUCTLINE'].unique()
df['DEALSIZE'].unique()

productline = pd.get_dummies(df['PRODUCTLINE']) #Converting the categorical columns. 
Dealsize = pd.get_dummies(df['DEALSIZE'])
df = pd.concat([df,productline,Dealsize], axis = 1)

df_drop  = ['COUNTRY','PRODUCTLINE','DEALSIZE'] #Dropping Country too as there are alot of countries. 
df = df.drop(df_drop, axis=1)

df['PRODUCTCODE'] = pd.Categorical(df['PRODUCTCODE']).codes #Converting the datatype.
df.drop('ORDERDATE', axis=1, inplace=True)
df.dtypes

distortions = [] # Within Cluster Sum of Squares from the centroid
K = range(1,10)
for k in K:
    kmeanModel = KMeans(n_clusters=k)
    kmeanModel.fit(df)
    distortions.append(kmeanModel.inertia_)   #Appeding the intertia to the Distortions 

plt.figure(figsize=(16,8))
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()

X_train = df.values #Returns a numpy array.
X_train.shape

model = KMeans(n_clusters=3,random_state=2) #Number of cluster = 3
model = model.fit(X_train) #Fitting the values to create a model.
predictions = model.predict(X_train) #Predicting the cluster values (0,1,or 2)

unique,counts = np.unique(predictions,return_counts=True)
counts = counts.reshape(1,3)
counts_df = pd.DataFrame(counts,columns=['Cluster1','Cluster2','Cluster3'])
counts_df.head()


pca = PCA(n_components=2) 
reduced_X = pd.DataFrame(pca.fit_transform(X_train),columns=['PCA1','PCA2'])

reduced_X.head()

#Plotting the normal Scatter Plot
plt.figure(figsize=(14,10))
plt.scatter(reduced_X['PCA1'],reduced_X['PCA2'])

model.cluster_centers_
reduced_centers = pca.transform(model.cluster_centers_)
reduced_centers

plt.figure(figsize=(14,10))
plt.scatter(reduced_X['PCA1'],reduced_X['PCA2'])
plt.scatter(reduced_centers[:,0],reduced_centers[:,1],color='black',marker='x',s=300) #Plotting the centriods

reduced_X['Clusters'] = predictions
reduced_X.head()

#Plotting the clusters 
plt.figure(figsize=(14,10))
#                     taking the cluster number and first column           taking the same cluster number and second column      Assigning the color
plt.scatter(reduced_X[reduced_X['Clusters'] == 0].loc[:,'PCA1'],reduced_X[reduced_X['Clusters'] == 0].loc[:,'PCA2'],color='slateblue')
plt.scatter(reduced_X[reduced_X['Clusters'] == 1].loc[:,'PCA1'],reduced_X[reduced_X['Clusters'] == 1].loc[:,'PCA2'],color='springgreen')
plt.scatter(reduced_X[reduced_X['Clusters'] == 2].loc[:,'PCA1'],reduced_X[reduced_X['Clusters'] == 2].loc[:,'PCA2'],color='indigo')

plt.scatter(reduced_centers[:,0],reduced_centers[:,1],color='black',marker='x',s=300)

