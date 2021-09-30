import pandas as pd
import numpy as np
'''
## This file creates a data frame where each row represents a different store
## The attributes obtained are:
    Type            - Countdown, SuperValue, FreshChoice
    Location        - General area where the store is located in
    Store           - Store name
    Lat             - latitude of store
    Long            - Longitude of store
    Mon...Sat       - Mean demand of pallets per day (rounded up)
'''


# Load location data from csv file
locations  = pd.read_csv("assignment_resources/WoolworthsLocations.csv")

# Load average demnd per day for store 
demand = pd.read_csv("assignment_resources/MeanDemandperWeek.csv")
## Merge data
merged = pd.merge(locations, demand, how = 'left', on ='Store')
##Uncomment to see head
#print(merged.head())
##Uncomment to check for null values
#merged.isnull().any()

# Save to csv file.
merged.to_csv("stores_df.csv")
