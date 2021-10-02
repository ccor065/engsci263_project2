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
region_names =["Central Region","South Region","North Region","East Region","West Region","Southern Most Region"]

## reg_boundaries columns = left lng, right lng, upper lat, lower
##                rows = region
                #central      south            north        east                west               south most
lng = [[174.7,174.8],[174.75, 174.9],[174.7,174.78],[174.81,174.95],[174.59, 174.6999999],[174.8, 175]]
                #central      south            north        east                west               south most
lat = [[-36.83,-36.93],[-36.935, -37.02],[-36.70,-36.82],[-36.85,-36.95],[-36.75, -36.95],[-37.04, -37.07]]


region_boundaires = pd.DataFrame({'lng':lng, 'lat':lat,'region':region_names})
print(region_boundaires.head())

## Merge data
merged = pd.merge(locations, demand, how = 'inner', on ='Store')
#merged = pd.merge(merged, region, how = 'inner', on ='Store')
region_add = ["invalid"]*len(merged.Store)
for i in range(len(merged.Store)):
    for j in range(len(region_boundaires.lng)):
        lng = region_boundaires.lng[j]
        lat = region_boundaires.lat[j]
        if merged.Long[i] >= lng[0] and merged.Long[i] <= lng[1] and merged.Lat[i] <= lat[0] and merged.Lat[i] >= lat[1]:
            region_add[i] = ( region_boundaires.region[j])
            added = True

merged["Region"] = region_add
'''Uncomment to see head'''
print(merged.head())
'''Uncomment to check for null values'''
# merged.isnull().any()

# Save to csv file.
merged.to_csv("stores_df.csv", index=False)

