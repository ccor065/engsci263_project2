import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.datasets import load_iris
'''
## This file creates a data frame where each row represents a different store
## The attributes obtained are:
    Type            - Countdown, SuperValue, FreshChoice
    Location        - General area where the store is located in
    Store           - Store name
    Lat             - latitude of store
    Long            - Longitude of store
    Mon...Sat       - Mean demand of pallets per day (rounded up)
    Region          - General region the store occupys.
NOTE: Daily demands do not include sunday as no deliverys are done on sundays.
'''

# Load location data from csv file
locations  = pd.read_csv("assignment_resources/WoolworthsLocations.csv")
# Load average demnd per day for store 
demand = pd.read_csv("assignment_resources/MeanDemandperWeek.csv")
# region data
region = pd.read_csv("assignment_resources/supermarket_regions.csv")
        #### Maybe regionally partition using lng lat to create a regional square..?

## Merge data
merged = pd.merge(locations, demand, how = 'inner', on ='Store')
merged = pd.merge(merged, region, how = 'inner', on ='Store')
'''Uncomment to see head'''
#print(merged.head())
'''Uncomment to check for null values'''
# merged.isnull().any()

# Save to csv file.
merged.to_csv("stores_df.csv", index=False)

iris=load_iris()

treeReg = DecisionTreeClassifier(max_depth = 3)
treeReg.fit(iris.data, iris.target)
r = export_text(treeReg, feature_names=iris['feature_names'])

target = np.zeros(5)
for i in range(1,13):
    target = np.append(target,np.ones(5)*i)



print(target)
treeReg = DecisionTreeClassifier(max_depth = 5)
vars = merged[['Lat', 'Long']].to_numpy()
tree = treeReg.fit(vars, target)
print(tree.n_classes_)
r = export_text(treeReg)
print(r)