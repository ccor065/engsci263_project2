from pulp import *
import pandas as pd
import numpy as np
from pulp.constants import LpMinimize
from generate_routes import *

# This file generates an optimal solition based on routes generated

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
    routes = LpVariable.dicts("routes", day_df.Route)

    # Objective function
    prob += lpSum([routes[i] * day_df['Costs'][i] for i in range(len(day_df))]), "Cost_of_route"
    #S/T Constraints



    # Solving routines 
    prob.writeLP('%s_solved.lp'%day)

    prob.solve()
    return


if __name__ == "__main__":
    stores_df = pd.read_csv("stores_df.csv")
    #Generate data frames from file. These tore the routes and costs of each day
    monday = pd.read_csv("generated_routes/monday_routes.csv")
    # print(monday)
    tuesday = pd.read_csv("generated_routes/tuesday_routes.csv")
    # print(tuesday)
    wednesday = pd.read_csv("generated_routes/wednesday_routes.csv")
    # print(wednesday)
    thursday = pd.read_csv("generated_routes/thursday_routes.csv")
    # print(thursday)
    friday = pd.read_csv("generated_routes/friday_routes.csv")
    # print(friday)
    saturday = pd.read_csv("generated_routes/saturday_routes.csv")
    # print(saturday)
    weekday_stores = stores_df['Store']
    saturday_stores = stores_df.loc[stores_df["Saturday"] != 0, ["Store"]]
