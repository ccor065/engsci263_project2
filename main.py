import numpy as np
import pandas as pd
import folium
import itertools

ORSkey = '88'
## Read in store data
stores_df = pd.read_csv('stores_df.csv')
## region names
region_names = np.array(["Central Region","South Region","North Region","East Region","West Region","Southern Most Region"])

## Construct Data frame for each region.
central =  stores_df[stores_df.loc[:]["Region"] == region_names[0]]
south =  stores_df[stores_df.loc[:]["Region"] == region_names[1]]
north =  stores_df[stores_df.loc[:]["Region"] == region_names[2]]
east =  stores_df[stores_df.loc[:]["Region"] == region_names[3]]
west =  stores_df[stores_df.loc[:]["Region"] == region_names[4]]
southernMost =  stores_df[stores_df.loc[:]["Region"] == region_names[5]]
print(len(stores_df))
## Construct a3D array  where each value of z is a different region e.g z=0=central region
#                 z =    0         1       2       3       4       5
regions = np.array([[central], [south], [north], [east], [west], [southernMost]], dtype = object)
# access region through regions[:][:][z] -- eg get centeral though regions[:][:][0]

def generate_routes(region, min_trucks, max_trucks):
    # CENTRAL STORES
    region_stores = np.array(region.Store)
    # intialise valid routes for central regions for each day
    mc = []
    tuc = []
    wc = []
    thc = []
    fc = []
    sc = []
    # generates routes for weekdays in size 2 to 4
    for i in range (min_trucks, max_trucks):
        for subset in itertools.combinations(region_stores, i):
            dm = 0
            dtu = 0
            dw = 0
            dth = 0
            df = 0
            for node in subset:
                z = region[region.Store == node]
                dm += z.Monday.values[0]
                dtu += z.Tuesday.values[0]
                dw += z.Wednesday.values[0]
                dth += z.Thursday.values[0]
                df += z.Friday.values[0]
            # Check freasblity of specfic route
            if dm <= 26:
                mc.append(subset)
            if dtu <= 26:
                tuc.append(subset)
            if dw <= 26:
                wc.append(subset)
            if dth <= 26:
                thc.append(subset)
            if df <= 26:
                fc.append(subset)
    # generates routes on saturdays, array below stores that get stock on saturday 
    satC_stores = []

    for store in region_stores:
        z = region[region.Store == store]
        if z.Saturday.values[0] != 0:
            satC_stores.append(store)

    for i in range (min_trucks,max_trucks):
        for subset in itertools.combinations(satC_stores, i):
            ds = 0
            for node in subset:
                z = region[region.Store == node]
                ds += z.Saturday.values[0]
            if ds <= 26:
                sc.append(subset)
    return [mc, tuc, wc,thc,fc,sc]


def generate_routes2(region, n_routes,nTrucks):

    feasible_routes = []
    indexes = np.arange(1, 15)
    region_stores = np.array(region.Store)
    #for i in range(2, nTrucks+1):
    for j in range(n_routes):
        np.random.shuffle(region_stores)
        split_indices = np.sort(np.random.choice(indexes, nTrucks))
        #split_indices = np.random.randint(1, 17,nTrucks )
        #split_indices = np.sort( np.random.randint(1, 17,nTrucks ))
        routes = np.split(region_stores,split_indices)
        feasible = True
        
        for route in routes:
            sum = 0
            for store in route:
                sum += (region[region.Store == store]).Monday.values[0]
            if sum >= 26:
                feasible = False
        if feasible:
            feasible_routes.append(list(routes))
        
        feasible_routes.append(list(routes))
    return feasible_routes


        
central_routes = generate_routes2(central, 1000, 5)

for route in central_routes[0]:
    print(route)
'''
central_routes = generate_routes(central, 2, 5) #number of stores 17
northern_routes = generate_routes(north, 2, 5)
southern_routes = generate_routes(south, 2, 5)
eastern_routes = generate_routes(east, 2, 5)
western_routes = generate_routes(west, 2, 5)
southern_routes= generate_routes(southernMost, 2, 5)

'''



