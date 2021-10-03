import numpy as np
import pandas as pd
import itertools
"""
This file generates feasible sub-routes per day, generates data frames per day.
"""


def generate_routes(stores_df, region_names):
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
    min = 2
    max = 5
    # intialise lists that store valid sub-routes per day
    monday_routes = []         #monday
    tuesday_routes = []        #tuesday .. etc
    wednesday_routes = []
    thursday_routes = []
    friday_routes = []
    saturday_routes = []
    # all routes must begin at disrubtion centre
    dc = 'Distribution Centre Auckland'
    # generates sub-routes per day
    """ WEEK DAY ROUTES"""
    for i in range(len(region_names)):
        region = stores_df[stores_df.loc[:]["Region"] == region_names[i]]
        region_stores = np.array(region.Store) 
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
                
                # Add duration to end of sub-route
                subset = (dc,) + subset + (dc,) 
                # Check freasblity of specfic route
                if dm <= 26:
                    
                    monday_routes.append(subset)
                if dtu <= 26:
                    tuesday_routes.append(subset)
                if dw <= 26:
                    wednesday_routes.append(subset)
                if dth <= 26:
                    thursday_routes.append(subset)
                if df <= 26:
                    friday_routes.append(subset)
        #Generate saturdays
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
                    # add duration to end of subset
                    subset = (dc,) + subset + (dc,) 
                    saturday_routes.append(subset)

    # return fealisble sub-routes for each day.


    return  monday_routes, tuesday_routes, wednesday_routes,thursday_routes, friday_routes,saturday_routes


def getCosts(route_set):
    """
    This function calculates the duration of a specfic route

    """
    cost_set =[]
    for route in route_set:
        nStore = len(route)
        duration=450*nStore
        cost =0
        for i in range(len(route)-1):
            currentStore = route[i]
            nextStore = route[i+1]
            row = durations.loc[durations['Store'] == currentStore]
            duration += row[nextStore].values[0]

        #less than four hours
        if duration <= 14400:
            cost = duration*0.0625
        if duration > 14400:
            cost = 14400*0.0625
            extraTime = duration - 14400
            cost += extraTime * (275/3600)
        cost_set.append(cost)
    return cost_set



if __name__ == "__main__":
    durations = pd.read_csv('assignment_resources/WoolworthsTravelDurations.csv')
    ## Read in store data
    stores_df = pd.read_csv('stores_df.csv')
    ## region names
    region_names = np.array(["Central Region","South Region","North Region","East Region","West Region","Southern Most Region"])
    monday, tuesday, wednesday, thursday, friday,saturday = generate_routes(stores_df, region_names)

    # Create data frame and save to file
    monday_df = pd.DataFrame({'Route': monday, 'Cost': getCosts(monday)})
    monday_df.to_csv("generated_routes/monday_routes.csv", index=False)

    tuesday_df = pd.DataFrame({'Route': tuesday, 'Cost':getCosts(tuesday)})
    tuesday_df.to_csv("generated_routes/tuesday_routes.csv", index=False)

    wednesday_df = pd.DataFrame({'Route': wednesday, 'Cost':getCosts(wednesday)})
    wednesday_df.to_csv("generated_routes/wednesday_routes.csv", index=False)

    thursday_df = pd.DataFrame({'Route': thursday, 'Cost':getCosts(thursday)})
    thursday_df.to_csv("generated_routes/thursday_routes.csv", index=False)

    friday_df = pd.DataFrame({'Route': friday, 'Cost': getCosts(friday)})
    friday_df.to_csv("generated_routes/friday_routes.csv", index=False)
    
    sat_df = pd.DataFrame({'Route': saturday, 'Cost':getCosts(saturday)})
    sat_df.to_csv("generated_routes/saturday_routes.csv", index=False)






