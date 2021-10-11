from pulp import *
import pandas as pd
import numpy as np
from pulp.constants import LpMinimize
from collections import Counter
import openrouteservice as ors
import folium
from random import randint

ORSkey = '5b3ce3597851110001cf6248324acec39fa94080ac19d056286d0ccb'


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
    min = 1
    max = 4+1 #actually 4
    # intialise lists that store valid sub-routes per day
    weekday_routes = []
    saturday_routes = []

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

    # return fealisble sub-routes for each day.
    return  saturday_routes, weekday_routes


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
    # All routes start and end here
    dc = 'Distribution Centre Auckland'
    # Loop through every route in the set
    for route in route_set:
        
        route = (dc,) + route + (dc,) 
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
    This function generates an optimal soltion for a given day
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
    # Create Vars
    # Each route is a varibale
    routes = LpVariable.dicts("Route", day_df.index, lowBound=0, upBound=1, cat = 'Binary')
    #integer var for extra trucks
    xt = LpVariable('xt', upBound= 5, lowBound=0, cat = 'Integer')

    # Objective function
    # = cost of route * route for all routes + 5000*number of extra trucks
    prob += lpSum([routes[i] * day_df['Cost'][i] for i in range(len(day_df))]+ 2000*xt), "Cost_of_route"
    
    #S/T onstraints
    # Each Store should be visited once and only once
    aMatrix = construct_matrix(day_df, stores)
    for i in range(len(aMatrix)):
        prob+= lpSum([routes[k] * aMatrix[i][k] for k in range(len(routes))]) == 1, "%s"% stores[i]

    # Trucks constraint
    prob += lpSum([routes[i]  for i in range(len(day_df))]- xt) <= 60, "Trucks_constraint"


    # Solving routines 
    #prob.writeLP('lin_progs/%s.lp'%day)
    prob.solve()
    # The status of the solution is printed to the screen
    print("Status:", LpStatus[prob.status])

    # Create df that stores th optimal routes, their costs and the region theyre in.
    optimalRoutes = []
    corRegion = []
    corCosts= []
    for v in prob.variables():
        if v.varValue == 1:
            index = (v.name).replace("Route_","")
            optimalRoutes.append(day_df['Route'][int(index)])
            corRegion.append(day_df["Region"][int(index)])
            corCosts.append(day_df["Cost"][int(index)])

    optimalRoutes = pd.DataFrame({"Route":optimalRoutes, "Region":corRegion, "Cost":corCosts})
    # The optimised objective function valof Ingredients pue is printed to the screen    
    print("Total Cost = ", value(prob.objective))
    # return objective value and the optimal routes
    return  value(prob.objective), optimalRoutes

def construct_matrix(day_df,storeSeries):
    """
    This function constructs a matrix of ones and zeros correlating to whether a store is visted in a specifc route
    ---------------------------------------------------------------
    INPUT:
        day_df: pandas data frame
                Stores the routes and the costs associated with each route
        stores: array-like (or pd series)
                Every store that must be visted on a given day
    --------------------------------------------------------------
    RETURNS:
        matrix: 2d numpy array
                matrix containing information on the which routes visit which stores.

    """
    stores = storeSeries.tolist()
    routes = (day_df["Route"]).tolist()

    matrix = np.zeros(( len(stores), len(routes)))
    # rows = stores, cols = routes

    for i in range(len(stores)):
        store = stores[i]
        if store == 'Distribution Centre Auckland':
            continue
        for j in range(len(routes)):
            route = routes[j]
            if store in route:
                matrix[i][j] = 1

    return matrix
# def mapSolutions(solutions, stores_df, index):
#     """ Maps solutions routes"""

#     # Set up open route client
#     client = ors.Client(key=ORSkey)

#     # read in files
#     durations = pd.read_csv('assignment_resources/WoolworthsTravelDurations.csv')
#     distances = pd.read_csv('assignment_resources/WoolworthsDistances.csv')

#     # Distrubtion centre attributes
#     dc = pd.read_csv('dc.csv')
#     dc_coords = (dc[['Long', 'Lat']]).to_numpy().tolist()

#     # region names
#     region_names = np.array(["Central Region","South Region","North Region","East Region","West Region","Southern Most Region"])

#     optimalRoutes = solutions['Optimal Route']
#     iconCol = "blue"
#     day = solutions.index[index]
#     routes = optimalRoutes[index]
#     i = 0


#     for route in routes:
#         route = (dc,) + route + (dc,)
#         locations = stores_df[stores_df['Store'].isin(route)]
#         coords = dc_coords + (locations[['Long', 'Lat']]).to_numpy().tolist() + dc_coords
#         map = folium.Map(location = list(reversed(coords[0])), zoom_start=12)
#         folium.Marker(list(reversed(coords[0])), popup= "DC", icon = folium.Icon(color = 'black')).add_to(map)
#         for i in range(1, len(route)-1):
#             folium.Marker(list(reversed(coords[i])), popup= str(route[i]), icon = folium.Icon(color = iconCol)).add_to(map)
#         line = client.directions(coordinates = [coord for coord in coords], profile ='driving-hgv', format ='geojson', validate = False)
#         folium.PolyLine(locations = [list(reversed(coord)) for coord in line['features'][0]['geometry']['coordinates']]).add_to(map)

#         map.save("route_maps/%s/%s_map.html"%(str(day), str(i)))
#         i+=1
                

#     return
def mapByRegion(solutions, stores_df, day):
    """ Maps solutions routes"""

    # Set up open route client
    client = ors.Client(key=ORSkey)

    # read in files
    durations = pd.read_csv('assignment_resources/WoolworthsTravelDurations.csv')
    distances = pd.read_csv('assignment_resources/WoolworthsDistances.csv')
    # list_colors = 
    # Distrubtion centre attributes
    dc = pd.read_csv('dc.csv')
    dc_coords = (dc[['Long', 'Lat']]).to_numpy().tolist()
    iconCol = "blue" #'beige', 'darkred', 'pink', 'white', 'cadetblue', 'darkblue', 'black', 'purple', 'green', 'red', 'darkpurple', 'lightred', 'gray', 'darkgreen', 'lightgray', 'blue', 'orange', 'lightgreen', 'lightblue'
    # region names

    region_names = np.array(["Central Region","South Region","North Region","East Region","West Region","Southern Most Region"])
    k=0
    for region in region_names:
        region_routes =(solutions.loc[solutions["Region"] == region, ["Route"]])
        region_routes = region_routes["Route"]
        map = folium.Map(location = list(reversed(dc_coords[0])), zoom_start=12)
        folium.Marker(list(reversed(dc_coords[0])), popup= "DC", icon = folium.Icon(color = 'black')).add_to(map)
        
        for route in region_routes:
            route = (dc,) + route + (dc,)
            locations = stores_df[stores_df['Store'].isin(route)]
            coords = dc_coords + (locations[['Long', 'Lat']]).to_numpy().tolist() + dc_coords
            colorA = ('#%06X' % randint(0, 0xFFFFFF))
            for i in range(1, len(route)-1):
                folium.Marker(list(reversed(coords[i])), popup= str(route[i]), icon = folium.Icon(color = iconCol)).add_to(map)
            line = client.directions(coordinates = [coord for coord in coords], profile ='driving-hgv', format ='geojson', validate = False)
            folium.PolyLine(locations = [list(reversed(coord)) for coord in line['features'][0]['geometry']['coordinates']], color =colorA).add_to(map)
            
            k+=1
            map.save("route_maps/%s/%s_map.html"%(str(day), region))

    return




if __name__ == "__main__":
    ## Read in store data
    stores_df = pd.read_csv('stores_df.csv')

    # Generate lists of stores for week-day deliveries
    weekday_stores = stores_df['Store']
    # List of stores that recive deliveries on saturdays
    satTemp = stores_df.loc[stores_df["Saturday"] != 0, ["Store"]]
    saturday_stores = satTemp['Store']
 

    # Intiallise region names array
    region_names = np.array(["Central Region","South Region","North Region","East Region","West Region","Southern Most Region"])
    
    # Generate sets of feaisble routes for saturday and weekdays
    saturday, weekdays = generate_routes(stores_df, region_names)


    
    orderedRegionSat = []
    orderedRegionWeek = []
    
    for route in saturday:
        store = route[0]
        orderedRegionSat.append((stores_df.loc[stores_df["Store"] == store, ["Region"]]).values[0])
    
    for route in weekdays:
        store = route[0]
        orderedRegionWeek.append((stores_df.loc[stores_df["Store"] == store, ["Region"]]).values[0])
    # Construct dataframes to store feasible routes and their corresponsing costs for sat and weekday
    sat_df = pd.DataFrame({'Route': saturday, 'Cost':getCosts(stores_df, saturday, "Saturday"), 'Region':orderedRegionSat})
    week_df = pd.DataFrame({'Route': weekdays, 'Cost':getCosts(stores_df, weekdays, "Weekday"), 'Region':orderedRegionWeek})

    # Solve each day!
    satCost, satRoutes = solve(sat_df, saturday_stores, "Saturday")
    weekCost, weekRoutes = solve(week_df, weekday_stores, "Weekdays")

    satRoutes.to_csv("sat_soln.csv")
    weekRoutes.to_csv("weekday_soln.csv")


    # Map the optimal routes per region per day :)
    mapByRegion(satRoutes, stores_df, "Saturday")
    mapByRegion(weekRoutes, stores_df, "Weekday")



    



