# engsci263_project2

Project 2: Trent Wong, Charlotte Cordwell, Ashley Tolentino  & Jacob Kim.
- dataframe.py:
             Partition stores into different regions.
             creates a csv file containing the individual attributes per store.
             generates csv file containing the total daily demands per region.
- main.py
            - solves for optimal solution and saves the linear program.
            - runs simulations with vairance in demands and traffic and does 
               a statistical analysis on this.
            - maps regions and optimal routes and saves to html file (find these in \maps, \route_maps)
- generate_routes.py
            - generates feasible sub-routes within a given region. 
            - Generates dataframe for wkdays aand satdays, conatining the routes, costs and regions
            - Dataframes are saved to files to reduce run-time.
- /dataframe_csv
            -contains csv files which have data for useful dataframes

- /lin_progs
            - Weekday and Saturday Linear programs


- index.html 
            - displays maps, run file in browser to see maps (code for this in main.py).

- /assginment_resources
            constains data files (csv)
- /maps
            contains mapping files for each region (HTML)
- /maps_routes
            contains mapping files for optimal routes for each day (HTML)



