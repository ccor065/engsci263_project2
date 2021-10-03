import pulp
import pandas as pd
import numpy as np
from generate_routes import *

# This file creates a set parttion

## Read in store data
stores_df = pd.read_csv('stores_df.csv')
## region names
region_names = np.array(["Central Region","South Region","North Region","East Region","West Region","Southern Most Region"])
monday, tuesday, wednesday, thursday, friday,saturday = generate_routes(stores_df, region_names)
mon_costs = getCosts(monday)
tueCosts = getCosts(tuesday)
wedCosts = getCosts(wednesday)
thurCosts = getCosts(thursday)
friCosts = getCosts(friday)
satCosts = getCosts(saturday)
## Test
print(len(satCosts) == len(saturday))
print(len(tueCosts) == len(tuesday))
print(len(mon_costs) == len(monday))
print(len(wedCosts) == len(wednesday))
print(len(thurCosts) == len(thursday))
print(len(friCosts) == len(friday))

weekday_stores = stores_df['Store']
saturday_stores = stores_df.loc[stores_df['Saturday']!=0,  ['Store']]
nTrucks = 30
'''
x = pulp.LpVariable.dicts('Monday', monday, 
                            lowBound = 0,
                            upBound = 1,
                            cat = pulp.LpInteger)

mon_routing = pulp.LpProblem("Monday Routing", pulp.LpMinimize)

mon_routing += sum([x[route] for route in monday]) <= nTrucks, "Maximum_number_of_trucks"
#Store should only be visited once
for store in weekday_stores:
    mon_routing += sum([x[route] for route in monday
                                if store in route]) == 1, "storez %s"%store
mon_routing.solve()
'''