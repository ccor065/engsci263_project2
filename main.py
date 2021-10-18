from pandas.core.frame import DataFrame
from pulp import *
import pandas as pd
import numpy as np
from collections import Counter
import openrouteservice as ors
import folium
from scipy import stats
import matplotlib.pyplot as plt
from  random import randint
from multiprocessing import Pool
from time import time
ORSkey = '5b3ce3597851110001cf6248324acec39fa94080ac19d056286d0ccb'


# Map generation functions
def mapByRegion(solutions,stores_df,  day):
    """ Maps solutions routes"""
    print("generating maps...")
    # Set up open route client
    client = ors.Client(key=ORSkey)

    # read in files
    durations = pd.read_csv('assignment_resources/WoolworthsTravelDurations.csv')
    distances = pd.read_csv('assignment_resources/WoolworthsDistances.csv')
    # list_colors = 
    # Distrubtion centre attributes
    dc = pd.read_csv('dc.csv')
    dc_coords = (dc[['Long', 'Lat']]).to_numpy().tolist()
    iconCol = 'blue'

    
    # region names

    region_names = np.array(["Central Region","South Region","North Region","East Region","West Region","Southern Most Region"])
    
   
    j = 0
    for region in region_names:
        region_routes =(solutions.loc[solutions["Region"] == region, ["Route"]])
        region_routes = region_routes["Route"]
        map = folium.Map(location = list(reversed(dc_coords[0])), zoom_start=12)
        folium.Marker(list(reversed(dc_coords[0])), popup= "DC", icon = folium.Icon(color = 'black', icon = "glyphicon-asterisk")).add_to(map)
        k =0
        for route in region_routes:
            route = ['Distribution Centre Auckland'] + route + ['Distribution Centre Auckland']
            locations = stores_df[stores_df['Store'].isin(route)]
            coords = dc_coords + (locations[['Long', 'Lat']]).to_numpy().tolist() + dc_coords
            colorA =  ('#%06X' % randint(0, 0xFFFFFF))
            for i in range(1, len(route)-1):
                folium.Marker(list(reversed(coords[i])), popup= str(route[i]), icon = folium.Icon(color = iconCol, icon = "glyphicon-shopping-cart")).add_to(map)
            line = client.directions(coordinates = [coord for coord in coords], profile ='driving-hgv', format ='geojson', validate = False)
            folium.PolyLine(locations = [list(reversed(coord)) for coord in line['features'][0]['geometry']['coordinates']], color =colorA, weight = 5 , opacity = 1).add_to(map)
            map.save("route_maps/%s/%s_map.html"%(day, region))
            k+=1
        j+=1

    
    print("map generation complete")
    return
def mapStores():
    locations = stores_df
    coords = locations[['Long', 'Lat']]
    coords = coords.to_numpy().tolist()
    region_names = np.array(["Central Region","South Region","North Region","East Region","West Region","Southern Most Region"])

    map = folium.Map(location = list(reversed(coords[2])), zoom_start=10)

    for i in range(0, len(coords)):

        if locations.Region[i]==region_names[0]:
            iconCol="green"
        elif locations.Region[i]==region_names[1]:
            iconCol="blue"
        elif locations.Region[i]==region_names[2]:
            iconCol="red"
        elif locations.Region[i]==region_names[3]:
            iconCol="orange"
        elif locations.Region[i]==region_names[4]:
            iconCol= "black"
        elif locations.Region[i]==region_names[5]:
            iconCol= "pink"
        elif locations.Region[i]=='invalid':
            iconCol= "white"
        if locations.Type[i]=='Distribution Centre':
            iconCol= "white"
        folium.Marker(list(reversed(coords[i])), popup ="%s \n lg %.3f\n lat: %.3f" % (locations.Store[i],locations.Long[i], locations.Lat[i]), icon = folium.Icon(color = iconCol)).add_to(map)

    #display map
    map.save("maps/map_locations.html") ##Open html file to see output

    """ Get different maps per region."""
    for name in region_names:
        map = folium.Map(location = list(reversed(coords[2])), zoom_start=11)
        for i in range(0, len(coords)):
            if locations.Region[i] == name or locations.Type[i] == "Distribution Centre" :
                if locations.Type[i]=="Countdown":
                    iconCol="green"
                elif locations.Type[i]=="FreshChoice":
                    iconCol="blue"
                elif locations.Type[i]=="SuperValue":
                    iconCol="red"
                elif locations.Type[i]=="Countdown Metro":
                    iconCol="orange"
                elif locations.Type[i] == "Distribution Centre":
                    iconCol= "black"
                folium.Marker(list(reversed(coords[i])), popup ="%s\n lg %.3f\n lat%.3f" % (locations.Store[i],locations.Long[i], locations.Lat[i]), icon = folium.Icon(color = iconCol)).add_to(map)
        map.save("maps/%s_map.html"%name.split()[0])
        
    return
# File readinf functions
def readInSatDF():
    '''
    This function reads in a csv file containing the data for saturday feasible routes
    ---------------------------------------------------------------------------------
    Returns:
            df: pandas dataframe
                dataframe containing, feasible routes, their costs and the region the 
                route is in
    '''
    df = pd.read_csv("dataframe_csv/saturday_df.csv")
    routes = df["Route"]
    for i in range(len(df)):
        route = routes[i]
        route = route.replace('[','')
        route = route.replace(']','')
        route = route.replace("'",'')
        route = route.split(',')
        route = [ele.strip() for ele in route]
        
        df.at[i, "Route"] = route


    return df
def readInWeekDF():
    '''
    This function reads in a csv file containing the data for weekday feasible routes
    ---------------------------------------------------------------------------------
    Returns:
            df: pandas dataframe
                dataframe containing, feasible routes, their costs and the region the 
                route is in
    '''
    df = pd.read_csv("dataframe_csv/weekday_df.csv")
    routes = df["Route"]
    for i in range(len(df)):
        route = routes[i]
        route = route.replace('[','')
        route = route.replace(']','')
        route = route.replace("'",'')
        route = route.split(',')
        route = [ele.strip() for ele in route]
        
        df.at[i,"Route"] = route

    return df
def getSatOptimal():
    '''
    This function reads in a csv file containing the data for weekday feasible routes
    ---------------------------------------------------------------------------------
    Returns:
            df: pandas dataframe
                dataframe containing, feasible routes, their costs and the region the 
                route is in
    '''
    df = pd.read_csv("dataframe_csv/Saturday_optimal.csv")
    routes = df["Route"]
    for i in range(len(df)):
        route = routes[i]
        route = route.replace('[','')
        route = route.replace(']','')
        route = route.replace("'",'')
        route = route.split(',')
        route = [ele.strip() for ele in route]
        
        df.at[i,"Route"] = route
    f = open('SaturdayObjectiveValue.txt', 'r')
    objVal = float((f.readline()))

    return df, objVal
def getWeekOptimal():
    '''
    This function reads in a csv file containing the data for weekday feasible routes
    ---------------------------------------------------------------------------------
    Returns:
            df: pandas dataframe
                dataframe containing, feasible routes, their costs and the region the 
                route is in
    '''
    df = pd.read_csv("dataframe_csv/Weekdays_optimal.csv")
    routes = df["Route"]
    for i in range(len(df)):
        route = routes[i]
        route = route.replace('[','')
        route = route.replace(']','')
        route = route.replace("'",'')
        route = route.split(',')
        route = [ele.strip() for ele in route]
        
        df.at[i,"Route"] = route
    f = open('WeekdaysObjectiveValue.txt', 'r')
    objVal = float((f.readline()))
    return df, objVal
## Linear programme solver and helper function.
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
    -------------------------------
    Returns: NONE
    -----------------------------
    NOTE: the following are saved to files to be read in later.
        objectiveValue: float
                                optimal cost of solution
        optimalRoutes: pandas dataframe
                        contains optimal routes, their individual costs and 
                        the region that the route is in.
    """
    # Create the 'prob' varibale to store all of the equations
    print("solving %s...."%day)
    prob = LpProblem("%s_Routing"%day, LpMinimize)
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
    prob.writeLP('lin_progs/%s.lp'%day)
    prob.solve(PULP_CBC_CMD(msg=0))
    # The status of the solution is printed to the screen
    #print("Status:", LpStatus[prob.status])

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
    # return objective value and the optimal routes'
    optimalRoutes.to_csv("dataframe_csv/%s_optimal.csv"%day)
    optimalCost = value(prob.objective)
    with open('%sObjectiveValue.txt'%day, 'w') as f:
        f.write(str(optimalCost))
    return  
def construct_matrix(day_df,storeSeries): 

    """
    This function constructs an ajacency matrix of ones and zeros correlating to 
    whether a store is visted in a specifc route
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

## Simulate with uncertainty functions
## Bootstrapping distribution for demand
def bootstrap(mean, sd):
    """
    This function generates a demand of a store using bootstrapping.
    -----------------------------------------------
    Input:
        mean: float
            mean value of the demand for a given store
        sd: float
            that standard deviation of demand at a store
    -----------------------------------------------
    Returns:
        m: float
            mean value of the bootstrap 
    ----------------------------------------
    NOTE: Assuming normal distribution of demands to generate bootstrap
    """
    # generate normal distribution from the mean and standard deviation for a given day, simulates population distribution
    randDemands = np.random.normal(loc = mean, scale = sd, size = 100)
    # bootstrap 1000 samples 
    demandMeans = []
    for i in range(100):
        # takes sample of size 50 from population
        sampleTaken = np.random.choice(randDemands, replace = True, size = 50)
        # calculates mean of sample
        averageDemand = np.mean(sampleTaken)
        demandMeans.append(averageDemand)
    # estimate mean from list of sampleMeans after the boostrap
    demand = np.mean(demandMeans)
    return demand

def generateDemandsSat(saturday_stores, demand_data):
    """
    This function generates a set of demands for saturday stores.
    ---------------------------------------
    Inputs:
        saturday_stores: list
                        stores that recive deliveries on saturdays
    ----------------------------------------
    Returns:
        sat_stores: pandas dataframe
                    dataframe containing the stores and their generated demands
    """
    
    sat_demands = demand_data[demand_data["WeekendMu"] != 0]
    demands = []
    for i in range(len(saturday_stores)):
        mu = sat_demands.iloc[i]["WeekendMu"]
        sigma = sat_demands.iloc[i]["WeekendSd"]
        demands.append(round(bootstrap(mu,sigma)))
    sat_stores = pd.DataFrame({'Store':saturday_stores, 'Demand':demands})
    return sat_stores
def generateDemandsWeek(weekday_stores, demand_data):
    """
    This function generates a set of demands for weekday stores.
    ---------------------------------------
    Inputs:
        weekday_stores: list
                        stores that recive deliveries on weekdays (all)
    ----------------------------------------
    Returns:
        weekday_stores: pandas dataframe
                    dataframe containing the stores and their generated demands
    """


    demands = []
    for i in range(65):
        mu = demand_data.iloc[i]["WeekdayMu"]
        sigma =  demand_data.iloc[i]["WeekdaySd"]
        demand = bootstrap(mu, sigma)
        demands.append(round(demand)) 
    weekday_stores = pd.DataFrame({'Store':weekday_stores, 'Demand':demands})
    return weekday_stores

def getDurationsVariance(store_demands, route_set):
    """
    This function calculates the time taken of a set of valid routes and 
    accounts for variances in duration length due to traffic
    -------------------------------------------------------
    Inputs :
            store_demands: pandas df
            Dataframe containing stores and their demands for a given simulation

            route_set: list of tuples
            List of all the valid routes for a given day
    --------------------------------------------------------
    Returns: 
        duration_set: list
        Correspoding list of the duration of each route.
    ----------------------------------------------------
    NOTE: 1. Indexes will match in the return so no further manipulation should be nesissary,
        Each route should start and finish at the disrubtion centre.
        2. Duration times are normally distrobuted.
    """
    durations = pd.read_csv('assignment_resources/WoolworthsTravelDurations.csv')
    # initalise
    cost_set =[]
    # All routes start and end here
    dc = ['Distribution Centre Auckland']
    # Loop through every route in the set
    duration_set = []
    for route in route_set:
        route = dc + list(route) + dc
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
                row = store_demands.loc[store_demands['Store'] == currentStore]
                nPallets = row["Demand"].values[0]
                duration += 450* nPallets
        # inttroduce vairiance
        multiplcationFactor = np.random.normal(loc=1, scale= 0.2) # Normal distribution, small deviation
        if multiplcationFactor < 0.8:
            multiplcationFactor = 0.8 # Cut off factor values below lower range
        duration = (duration *multiplcationFactor)/60
        #return in duration in mins
        duration_set.append(duration)
    return duration_set
def getCostsVariance(store_demands, route_set):
    """
    This function calculates the asocciated costs of a set of valid routes
    -------------------------------------------------------
    Inputs :
            store_demands: pandas df
            Dataframe containing stores and their demands for a given simulation
            
            route_set: list of tuples
            List of all the valid routes for a given day
    --------------------------------------------------------
    Returns: 
        cost_set: list
        Correspoding list of the cost of each route in routes sets
    ----------------------------------------------------
    NOTE: Indexes will match in the return so no further manipulation should be nesissary,
        Each route should start and finish at the disrubtion centre.
    """
    durations = getDurationsVariance(store_demands, route_set)
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
    return cost_set
def solveExtra(day_df, stores):
    """
    This function generates an optimal soltion for re-routing 'overflow' stores
    -----------------------------
    Inputs: 
        day_df: pandas data frame
                Stores the routes and the costs associated with each route
        stores: array-like (or pd series)
                Every store that must be visted on a given day
    ------------------------------
    Returns:
        cost: float
            total cost of re routing the extra stores
        optimal routes: list
                lists of optimal routes generated from 'overlflow' stores.

    """
    # Create the 'prob' varibale to store all of the equations
    prob = LpProblem("Extra_Routing", LpMinimize)
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
  
    prob.solve(PULP_CBC_CMD(msg=0))
    #print("Status:", LpStatus[prob.status])
    # Create df that stores th optimal routes, their costs and the region theyre in.
    optimalRoutes = []

    for v in prob.variables():
        if v.varValue == 1:
            index = (v.name).replace("Route_","")
            optimalRoutes.append(day_df['Route'][int(index)])
    # The optimised objective function valof Ingredients pue is printed to the screen    
    # return objective value and the optimal routes
    return  value(prob.objective), optimalRoutes

def generateExtraRoutes(extraStores_df):
    """
    This function generates routes based on the 'overflow' stores where their simulated demands
    cause the total demand of a route to exceed a trucks capacity. These generated routes are
    then solved using a linear programme to optimise cost.
    ---------------------------------------------------
    Inputs: 
        extraStores_df: pandas df
                    dataframe containing the 'overflow' stores, and their simulated demands.
    -----------------------------------------------
    Returns:
        cost: float
            total cost of re routing the extra stores
        optimal routes: list
                lists of optimal routes generated from 'overlflow' stores.
    """
    routes = []
    stores = extraStores_df["Store"]
    for i in range (1, len(stores)):
        # get every possible combintaion of stores based on how many are in each route.
        for subset in itertools.combinations(stores, i):
            # initalise counters, these count demands for routes.
            demandOfRoute =0
            # cycle through each store in the sub-route and sum their demands.
            for node in subset:
                # add to counter based on day
                store = extraStores_df[extraStores_df.Store == node]
                demandOfRoute += store.Demand.values[0] 
            # Check freasblity of specfic route
            if demandOfRoute <= 26:
                routes.append(subset)
    routes =[list(ele) for ele in routes]
    routes_df = pd.DataFrame({'Route':routes, 'Cost':getCostsVariance(extraStores_df, routes)})
    cost, optimalRoutes = solveExtra(routes_df, stores)

    return cost, optimalRoutes

def simulateWeek(route_df, stores,demand_data, n):
    prior_routes = route_df["Route"]
    optimal_costs =np.zeros(n)
    extraStores_list = []
    for i in range(n):

        # Generate demands fo all stores
        week_demands = generateDemandsWeek(stores,demand_data)
        
        """Check validity of routes with the new demands"""
        extra_stores = []
        extra_storesDemand = []
        routes = []
        for route in prior_routes:
            demandOfRoute =0
            for node in route:
                store = week_demands[week_demands.Store == node]
                demandOfRoute += store.Demand.values[0] 
            if demandOfRoute <= 26:
                routes.append(route)
            else:
                extra_store = route[-1]
                route = route[0:-1]
                routes.append(route)
                extra_stores.append(extra_store)
                extraStores_list.append(extra_store)
                extra_storesDemand.append((week_demands[week_demands.Store == extra_store]).Demand.values[0])
        extraStores_cost =0
        if len(extra_stores)>1:

            extraStores_df = pd.DataFrame({"Store":extra_stores, "Demand":extra_storesDemand})
            extraStores_cost, extraRoutes = generateExtraRoutes(extraStores_df)

        if len(extra_stores) ==1:
            routes.append((extra_stores[0],))
        # generate other routes
        # get costs with the new demands, and incude variances in traffic.
        costs = getCostsVariance(week_demands, routes) 
        optimal_costs[i]= (sum(costs) + extraStores_cost)
        
    return optimal_costs, extraStores_list
def simulateSat(route_df, stores, demand_data, n):
    
    prior_routes = route_df["Route"]
    extraStores_list = []
    optimal_costs =np.zeros(n)
    for i in range(n):
        # Generate demand for stores on the pert-beta distrubution
        sat_demands = generateDemandsSat(stores, demand_data) 
        #Check validity of  routes with the new demands
        extra_stores = []
        extra_storesDemand = []
        routes = []
        for route in prior_routes:

            demandOfRoute =0
            for node in route:
                store = sat_demands[sat_demands.Store == node]
                demandOfRoute += store.Demand.values[0] 
            if demandOfRoute <= 26:
                routes.append(route)
            else:
                extra_store = route[-1]
                route = route[0:-1]
                routes.append(route)
                extra_stores.append(extra_store)
                extraStores_list.append(extra_store)
                extra_storesDemand.append((sat_demands[sat_demands.Store == extra_store]).Demand.values[0])

            # generate extra routes
        extraStores_cost =0
        if len(extra_stores)>1:
            extraStores_df = pd.DataFrame({"Store":extra_stores, "Demand":extra_storesDemand})
            extraStores_cost, extraRoutes = generateExtraRoutes(extraStores_df)


        if len(extra_stores) ==1:
            routes.append((extra_stores[0],))
        # generate other routes
        # get costs with the new demands, and incude variances in traffic.
        costs = getCostsVariance(sat_demands, routes) 
        optimal_costs[i] = (sum(costs) + extraStores_cost)
    return optimal_costs, extraStores_list

if __name__ == "__main__":



    # Read in store data
    stores_df = pd.read_csv('dataframe_csv/stores_df.csv')
    # Generate lists of stores for week-day deliveries
    weekday_stores = stores_df['Store']
    # List of stores that recive deliveries on saturdays
    saturday_stores = (stores_df.loc[stores_df["Saturday"] != 0, ["Store"]])["Store"]
   
    """  Unocmment saturday and sunday to obtain optimal cost and the optimal route set."""
    # # read in individual data frames containing feasible route info.
    # sat_df = readInSatDF()
    # week_df = readInWeekDF()
    # solve(sat_df, saturday_stores, "Saturday")
    # solve(week_df, weekday_stores, "Weekdays")

    # Obtain optimal routes data frames for each day and the objective value.
    satRoutes, satCost = getSatOptimal()
    weekRoutes, weekCost, = getWeekOptimal()
    
    """ Uncomment to run mapping for optimal routes"""
    # mapByRegion(satRoutes, stores_df, "Saturday")
    # mapByRegion(weekRoutes, stores_df, "Weekday")
    
    """ Simulation"""
    demand_data = pd.read_csv("assignment_resources/DemandMeanSD.csv")
    p = Pool(4)     #Simulate using parellel processing
    n = 1000        # number of simulations (>10)
    lwr = int(0.025 * n) #entral interval lower bound
    uper = int(0.975 * n) #central interval upper bound
    print("Simulating Weekdays (this takes about 2.5 mins)...")
    t0 = time()
    inputs = [(weekRoutes, weekday_stores, demand_data, int(n/8))]*8
    weekData = p.starmap(simulateWeek, inputs)
    t1 = time()
    weekSimulatedCosts = []
    weekExtraStores = []
    for tup in weekData:
        weekSimulatedCosts.extend([*tup[0]])
        weekExtraStores.extend([*tup[1]])
    weekSimulatedCosts.sort()
    print(weekExtraStores)

    print("")
    print("=========================================================")
    print("WEEKDAY RESULTS:")
    print("=========================================================")
    print("Central 95%% Interval: Lwr: %.3f - Upper: %.3f"%(weekSimulatedCosts[lwr],weekSimulatedCosts[uper]))
    print("Average Cost: $%f"%np.mean(weekSimulatedCosts))
    weekExtraStores = Counter(weekExtraStores)
    print("=========================================================")
    print("STORE                                   FREQUENCY DROPPED")
    print("=========================================================")
    for store in weekExtraStores:
        print("%s:"%store,+(45- len(store))*" " + "%d"% weekExtraStores[store])
    print("=========================================================")
    print("Simulation run time:                         %.2fs"%(t1-t0))
    print("")


    print("Simulating Saturdays (this takes about 2 mins)...")
    inputs = [(satRoutes,saturday_stores,demand_data,10)]*int(n/10)
    t0 = time()
    weekData = p.starmap(simulateSat, inputs)
    t1 = time()
    satSimulatedCosts = []
    satExtraStores = []
    for tup in weekData:
        satSimulatedCosts.extend([*tup[0]])
        satExtraStores.extend([*tup[1]])
    satSimulatedCosts.sort()
    print("")
    print("=========================================================")
    print("SATURDAY RESULTS:")
    print("=========================================================")
    print("Central 95%% Interval: Lwr: %.3f - Upper: %.3f"%(satSimulatedCosts[lwr],satSimulatedCosts[uper]))
    print("Average Cost: $%f" %np.mean(satSimulatedCosts))

    if len(satExtraStores) == 0:
        print("=========================================================")
        print("No extra routes were required in any simulation.")
        print("=========================================================")
    else:
        satExtraStores = Counter(satExtraStores)
        print("=========================================================")
        print("STORE                                  FREQUENCY DROPPED")
        print("=========================================================")
        for store in satExtraStores:
            print("%s:"%store,+(35- len(store))*" " + "%d"% satExtraStores[store])
        print("=========================================================")
    print("Simulation run time:                         %.2fs"%(t1-t0))



    plt.hist(weekSimulatedCosts, density=True, histtype='bar', alpha=0.2)
    plt.title("Cost distribution for Weekdays for 1000 Simulations")
    plt.axvline(weekSimulatedCosts[lwr], linestyle='dashed', color='red')
    plt.axvline( weekSimulatedCosts[uper], linestyle='dashed', color='red')
    

    plt.ylabel("Occurance")
    plt.xlabel("Cost")
    plt.savefig('weekday_sim_distribution ',dpi=300)
    plt.show()
    plt.hist(satSimulatedCosts, density=True, histtype='bar', alpha=0.2)
    plt.axvline(x = satSimulatedCosts[lwr], linestyle='dashed', color='red')
    plt.axvline(x = satSimulatedCosts[uper], linestyle='dashed', color='red')

    plt.title("Cost distribution for Saturdays for 1000 Simulations")
    plt.ylabel("Occurance")
    plt.xlabel("Cost")
    plt.savefig('saturday_sim_distribution ',dpi=300)
    plt.show()


    



