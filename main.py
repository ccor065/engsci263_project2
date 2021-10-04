from pulp import *
import pandas as pd
import numpy as np
from pulp.constants import LpMinimize


# This file generates an optimal solition based on routes generated



def generate_routes(stores_df, region_names):
    """
    This function generates sub-routes within a given region and checks the feasiblity of a sub-route based
    of truck capacities based on the mean daily demands of each store in the sub-route.
    ------------------------------------------------------------
    INPUTS:
        stores_df: pandas df
                Dataframe containing onfromation on all the stores.
        region_names:    array of strings
               contains the names of each region
    -------------------------------------------------------------
    RETURNS:
        Monday, Tuesday, Wednesday, Thursday, Friday and Saturday: list 
                are lists containing feasible routes on Monday, Tuesday, Wednesday, Thursday
                Friday and Saturday respectively.
    -------------------------------------------------------------
    NOTE: Region names should be exactly the same as the names in the 'region' column of df  
    """
    # Number of stores visited per route min / max
    min = 2
    max = 4+1 #actually 4
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
        day should be indentical to the row name i.e Monday = valid, monday = invalid.
    """
    durations = pd.read_csv('assignment_resources/WoolworthsTravelDurations.csv')
    # initalise
    cost_set =[]
    # Loop through every route in the set
    for route in route_set:
        
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


        # less than four hours normal rates apply
        if duration <= 14400:
            cost = duration*0.0625
        # greater than 4 hour trip then costs are extra
        if duration > 14400:
            cost = 14400*0.0625
            extraTime = duration - 14400
            cost += extraTime * (275/3600)
        cost_set.append(cost)
    return cost_set
## Forulate and solve linear prog
def solve(day_df, stores, day):
    """
    This function generates an optimal soltion for agiven day
    -----------------------------
    Inputs: 
        day_df: pandas data frame
                Stores the routes and the costs associated with each route
        stores: array-like (or pd series)
                Every store that must be visted on a given day
        day:    String
                name of the day that is being solved
    """
    # Create the 'prob' varibale to store all of the equations
    prob = LpProblem("%s Routing"%day, LpMinimize)
    # Create dictonary from the df
    routes = LpVariable.dicts("routes", day_df.index, lowBound =0, upBound=1)

    # Objective function
    prob += lpSum([routes[i] * day_df['Cost'][i] for i in range(len(day_df))]), "Cost_of_route"
    
    #S/T Constraints

    # Each node should be selected once
    for currentStore in stores:
        if currentStore == 'Distribution Centre Auckland':
                break
        indexes = []
        # loop through all routes and check if a route contains the store in the iteration
        for i in range(len(day_df)):
            route = day_df["Route"][i]
            for j in range(1,len(day_df["Route"][i]) -1):
                store = day_df["Route"][i][j]
                if  store == currentStore:
                    indexes.append(i)
        # add store constraint to problem
        prob += lpSum([routes[i]  for i in indexes]) == 1, "%s"% currentStore

    prob += lpSum([routes[i]  for i in range(len(day_df))]) <= 60, "Trucks_constraint"



    # Solving routines 
    prob.writeLP('lin_progs/%s.lp'%day)
    prob.solve()
    # The status of the solution is printed to the screen
    print("Status:", LpStatus[prob.status])

    # Each of the variables is printed with its resolved optimum value

    i=0
    for v in prob.variables():

        if v.varValue == 1.0:
            print(v.name, "=", v.varValue)
            print(day_df['Route'][i])
        i+=1
  
        
    # The optimised objective function valof Ingredients pue is printed to the screen    
    print("Total Cost = ", value(prob.objective))


    return  value(prob.objective)
def construct_matrix(stores, day_df):
    for currentStore in stores:
        for i in range(len(day_df)):
            route = day_df["Route"][i]
            print(type(route))
            for store in route:
                matrix =0


    return matrix
if __name__ == "__main__":


    ## Read in store data
    stores_df = pd.read_csv('stores_df.csv')
    weekday_stores = stores_df['Store']
    saturday_stores = stores_df.loc[stores_df["Saturday"] != 0, ["Store"]]
    # region names
    region_names = np.array(["Central Region","South Region","North Region","East Region","West Region","Southern Most Region"])
    monday, tuesday, wednesday, thursday, friday,saturday = generate_routes(stores_df, region_names)
    days = [monday, tuesday, wednesday, thursday, friday]

    #Check if routes generated are different, therefore solve seperately
    ele = days[0]
    chk = True
    # Comparing each element with first item 
    for item in days:
        if ele != item:
            chk = False
            break
              
    if (chk == True): 
        print("Equal")
    else: 
        print("Not equal")  
    # Generate dataframes
    monday_df = pd.DataFrame({'Route': monday, 'Cost': getCosts(stores_df, monday, 'Monday')})
    tuesday_df = pd.DataFrame({'Route': tuesday, 'Cost':getCosts(stores_df, tuesday, "Tuesday")})
    wednesday_df = pd.DataFrame({'Route': wednesday, 'Cost':getCosts(stores_df, wednesday, "Wednesday")})
    thursday_df = pd.DataFrame({'Route': thursday, 'Cost':getCosts(stores_df, thursday, "Thursday")})
    friday_df = pd.DataFrame({'Route': friday, 'Cost': getCosts(stores_df, friday, "Friday")})
    sat_df = pd.DataFrame({'Route': saturday, 'Cost':getCosts(stores_df, saturday, "Saturday")})
    
    monSoln = solve(monday_df, weekday_stores, "Monday")
    
    tuesSoln = solve(tuesday_df, weekday_stores, "Tuesday")
    
    wedSoln = solve(wednesday_df, weekday_stores, "Wednesday")
    
    thursSoln = solve(thursday_df, weekday_stores, "Thursday")
    
    friSoln = solve(friday_df, weekday_stores, "Friday")
    print(monSoln)
    print(tuesSoln)
    print(wedSoln)
    print(thursSoln)
    print(friSoln)
    # satSoln = solve(sat_df, saturday_stores, "Saturday")
    # print(satSoln)


