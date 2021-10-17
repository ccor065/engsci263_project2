import pandas as pd
import numpy as np
import itertools
"""
This file generates sub-routes within a given region and checks the feasiblity of a sub-route based
of truck capacities based on the mean daily demands of each store in the sub-route. It then saves a dataframe
which contains these feasible routes, their asocciated cost and what region it is in.
"""
def getDurations(stores_df, route_set, day):
    """
    This function calculates the time taken of a set of valid routes
    -------------------------------------------------------
    Inputs :
            stores_df: pandas df
            Dataframe containing onfromation on all the stores.
            route_set: list of tuples
            List of all the vild routes for a given day
            day: str
            corresponding day to set of routes 
    --------------------------------------------------------
    Returns: 
        duration_set: list
        Correspoding list of the duration of each route.
    ----------------------------------------------------
    NOTE: Indexes will match in the return so no further manipulation should be nesissary,
        Each route should start and finish at the disrubtion centre.
        day should be indentical to the row name i.e Monday = valid, monday = invalid.

    """
    durations = pd.read_csv('assignment_resources/WoolworthsTravelDurations.csv')
    # initalise
    cost_set =[]
    # All routes start and end here
    dc = ['Distribution Centre Auckland']
    # Loop through every route in the set
    duration_set = []
    for route in route_set:
        route = dc + route +dc
        duration= 0  # Initalise duration 
        cost =0     # Initalise cost to append to set
        # Loop through to get total duration of route
        for i in range(len(route)-1):
            currentStore = route[i]
            nextStore = route[i+1]
            row = durations.loc[durations['Store'] == currentStore]
            duration += row[nextStore].values[0]
            # add unpacking time for each pallet at each store (EXCLD. distribtion centre)
            if i >0:
                row = stores_df.loc[stores_df['Store'] == currentStore]
                nPallets = row[day].values[0]
                duration += 450* nPallets
        #return in duration in mins
        duration_set.append(duration/60)
    return duration_set
def getCosts(stores_df, route_set, day):
    """
    This function calculates the asocciated costs of a set of valid routes
    -------------------------------------------------------
    Inputs :
            stores_df: pandas df
            Dataframe containing onfromation on all the stores.
            route_set: list of tuples
            List of all the vild routes for a given day
            day: str
            corresponding day to set of routes 
    --------------------------------------------------------
    Returns: 
        cost_set: list
        Correspoding list of the cost of each route in routes sets
    ----------------------------------------------------
    NOTE: Indexes will match in the return so no further manipulation should be nesissary,
        Each route should start and finish at the disrubtion centre.
        day should be indentical to the row name i.e Saturday = valid, saturday = invalid.
    """
    print("generating cost of %s routes..."%day)
    durations = getDurations(stores_df, route_set, day)
    cost_set =[]
    for duration in durations:
        # less than four hours normal rates apply
        if duration <= 240:
            cost = duration*(225/60) #cost per min
        # greater than 4 hour trip then costs are extra
        if duration > 240:
            cost = 240*(225/60)
            extraTime = duration - 240
            cost += extraTime * (275/60)
        cost_set.append(cost)
    print("generating cost of %s routes complete."%day)
    return cost_set
if __name__ == "__main__":
    print("generating routes...")
        ## Read in store data
    stores_df = pd.read_csv('dataframe_csv/stores_df.csv')
    region_names = np.array(["Central Region","South Region","North Region","East Region","West Region","Southern Most Region"])
    # Number of stores visited per route min / max
    min = 1
    max = 4+1 #actually 4
    # intialise lists that store valid sub-routes per day
    weekday_routes = []
    wkday_regions = []
    saturday_routes = []
    sat_regions = []

    # generates sub-routes per day

    for regionType in region_names:
        region = stores_df[stores_df.loc[:]["Region"] == regionType]
        region_stores = np.array(region.Store) 
        ## Loop through number of stores per sub-route
        """ WEEK DAY ROUTES"""
        for i in range (min, max):
            # get every possible combintaion of stores based on how many are in each route.
            for subset in itertools.combinations(region_stores, i):
                # initalise counters, these count demands for routes.
                demandOfRoute =0
                # cycle through each store in the sub-route and sum their demands.
                for node in subset:
                    # add to counter based on day
                    store = region[region.Store == node]
                    demandOfRoute += store.Weekday.values[0] 
                # Check freasblity of specfic route
                if demandOfRoute <= 26:
                    weekday_routes.append(subset)
                    wkday_regions.append(regionType)

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
                demandOfSatRoute = 0 #Initalse demand counter
                for node in subset:
                    store = region[region.Store == node]
                    demandOfSatRoute += store.Saturday.values[0]
                #Check feaislblity
                if demandOfSatRoute <= 26:
                    saturday_routes.append(subset)
                    sat_regions.append(regionType)


    print("generating routes complete.")
    # Construct data frames for return.
    saturday_routes =[list(route) for  route in saturday_routes]
    weekday_routes =[list(route) for  route in weekday_routes]
    saturday_routes = pd.DataFrame({"Route":saturday_routes,"Cost":getCosts(stores_df,saturday_routes, "Saturday"), "Region":sat_regions})
    weekday_routes = pd.DataFrame({"Route":weekday_routes, "Cost":getCosts(stores_df, weekday_routes, "Weekday"),"Region":wkday_regions})
    print("generating cost of routes complete.")
    print("Saving data frames to csv..")
    saturday_routes.to_csv("dataframe_csv/saturday_df.csv", index=False)
    weekday_routes.to_csv("dataframe_csv/weekday_df.csv", index=False)
    print("Saved sucessfully")
 
