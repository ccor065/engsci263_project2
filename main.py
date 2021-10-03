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



def generate_routes(region, min, max):
    """
    This function generates sub-routes within a given region and checks the feasiblity of a sub-route based
    of truck capacities based on the mean daily demands of each store in the sub-route.
    ------------------------------------------------------------
    INPUTS:
        Region: pandas df
                Dataframe containing onfromation on all the stores in a given region.
        min:     int
                Minimum number of stores per route for generation
        max:     int 
                Maximum number of stores per sub-route.
    -------------------------------------------------------------
    RETURNS:
        [mc, tuc, wc, thc, fc, sc]: list of lists
        Where: mc, tuc, wc, thc, fc, sc 
                are lists containing feasible routes on Monday, Tuesday, Wednesday, Thursday
                Friday and Saturday respectively.
    -------------------------------------------------------------
    NOTE: Max should be actual max not actual max+1 as this is accounted for in the looping.
            
    """
    max +=1
    region_stores = np.array(region.Store)
    # intialise lists that store valid sub-routes per day
    mc = []         #monday
    tuc = []        #tuesday .. etc
    wc = []
    thc = []
    fc = []
    sc = []
    # generates sub-routes per day
    """ WEEK DAY ROUTES"""
    ## Loop through number of stores per sub-route
    for i in range (min, max):
        # get every possible combintaion of stores based on how many are in each route.
        for subset in itertools.combinations(region_stores, i):
            # initalise counters, these count demands for routes.
            dm = 0
            dtu = 0
            dw = 0
            dth = 0
            df = 0
            # cycle through each store in the sub-route and sum their demands.
            for node in subset:
                # add to counter based on day/
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

    """SATURDAY ROUTE GEN"""
    # Initailise list that will store the stores that get delivery on saturdays.
    satC_stores = []
    # Add stores to list
    for store in region_stores:
        z = region[region.Store == store]
        if z.Saturday.values[0] != 0:
            satC_stores.append(store)
    #Generate sub-routes say way as week-day
    for i in range (min,max):
        for subset in itertools.combinations(satC_stores, i):
            ds = 0 #Initalse demand counter
            for node in subset:
                z = region[region.Store == node]
                ds += z.Saturday.values[0]
            #Check feaislblity
            if ds <= 26:
                sc.append(subset)
    # return fealisble sub-routes for each day.
    return [mc, tuc, wc,thc,fc,sc]



''' get number of sotes in each region
print(len(north.index))
print(len(south.index))
print(len(east.index))
print(len(west.index))
print(len(southernMost.index))
'''

'''
central_routes = generate_routes(central, 2, 4) #number of stores 17
northern_routes = generate_routes(north, 2, 4) #9 stores
southern_routes = generate_routes(south, 2, 4) #9 stores
eastern_routes = generate_routes(east, 2, 4) #12 stores
western_routes = generate_routes(west, 2, 4) # 14 stores
southern_routes= generate_routes(southernMost, 2, 4) #4 stores

'''



