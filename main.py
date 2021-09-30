import numpy as np
import pandas as pd
import folium

ORSkey = '88'
## Read in store data
stores_df = pd.read_csv('stores_df.csv')
region_names = np.array(["Central Region","South Region","North Region","East Region","West Region","Southern Most Region"])

## Construct Data frame for each region.

central =  stores_df[stores_df.loc[:]["Region"] == region_names[0]]
south =  stores_df[stores_df.loc[:]["Region"] == region_names[1]]
north =  stores_df[stores_df.loc[:]["Region"] == region_names[2]]
east =  stores_df[stores_df.loc[:]["Region"] == region_names[3]]
west =  stores_df[stores_df.loc[:]["Region"] == region_names[4]]
southernMost =  stores_df[stores_df.loc[:]["Region"] == region_names[5]]

## Construct a3D array  where each value of z is a different region e.g z=0=central region
#                 z =    0         1       2       3       4       5
regions = np.array([[central], [south], [north], [east], [west], [southernMost]])
# access region through regions[:][:][index]
print(regions[:][:][0])

