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

## Construct a3D array  where each value of z is a different region e.g z=0=central region
#                 z =    0         1       2       3       4       5
regions = np.array([[central], [south], [north], [east], [west], [southernMost]])
# access region through regions[:][:][z] -- eg get centeral though regions[:][:][0]
print(regions[:][:][0])



# CENTRAL STORES
central_stores = np.array(central.Store)
# intialise valid routes for central regions for each day
mc = []
tuc = []
wc = []
thc = []
fc = []
sc = []

# generates routes for weekdays in size 2 to 4
for i in range (2,5):
    for subset in itertools.combinations(central_stores, i):
        dm = 0
        dtu = 0
        dw = 0
        dth = 0
        df = 0
        for node in subset:
            z = central[central.Store == node]
            dm += z.Monday.values[0]
            dtu += z.Tuesday.values[0]
            dw += z.Wednesday.values[0]
            dth += z.Thursday.values[0]
            df += z.Friday.values[0]
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

for store in central_stores:
    z = central[central.Store == store]
    if z.Saturday.values[0] != 0:
        satC_stores.append(store)

for i in range (2,6):
    for subset in itertools.combinations(satC_stores, i):
        ds = 0
        for node in subset:
            z = central[central.Store == node]
            ds += z.Saturday.values[0]
        if ds <= 26:
            sc.append(subset)


# SOUTH
south_stores = np.array(south.Store)
# intialise valid routes for south regions for each day
mS = []
tuS = []
wS = []
thS = []
fS = []
sS = []

# generates routes for weekdays in size 2 to 4
for i in range (2,4):
    for subset in itertools.combinations(south_stores, i):
        dm = 0
        dtu = 0
        dw = 0
        dth = 0
        df = 0
        for node in subset:
            z = south[south.Store == node]
            dm += z.Monday.values[0]
            dtu += z.Tuesday.values[0]
            dw += z.Wednesday.values[0]
            dth += z.Thursday.values[0]
            df += z.Friday.values[0]
        if dm <= 26:
            mS.append(subset)
        if dtu <= 26:
            tuS.append(subset)
        if dw <= 26:
            wS.append(subset)
        if dth <= 26:
            thS.append(subset)
        if df <= 26:
            fS.append(subset)

# generates routes on saturdays, array below stores that get stock on saturday 
satS_stores = []

for store in south_stores:
    z = south[south.Store == store]
    if z.Saturday.values[0] != 0:
        satS_stores.append(store)

for i in range (2,6):
    for subset in itertools.combinations(satS_stores, i):
        ds = 0
        for node in subset:
            z = south[south.Store == node]
            ds += z.Saturday.values[0]
        if ds <= 26:
            sS.append(subset)


# WEST STORES
west_stores = np.array(west.Store)
# intialise valid routes for west regions for each day
mW = []
tuW = []
wW = []
thW = []
fW = []
sW = []

# generates routes for weekdays in size 2 to 4
for i in range (2,4):
    for subset in itertools.combinations(west_stores, i):
        dm = 0
        dtu = 0
        dw = 0
        dth = 0
        df = 0
        for node in subset:
            z = west[west.Store == node]
            dm += z.Monday.values[0]
            dtu += z.Tuesday.values[0]
            dw += z.Wednesday.values[0]
            dth += z.Thursday.values[0]
            df += z.Friday.values[0]
        if dm <= 26:
            mW.append(subset)
        if dtu <= 26:
            tuW.append(subset)
        if dw <= 26:
            wW.append(subset)
        if dth <= 26:
            thW.append(subset)
        if df <= 26:
            fW.append(subset)

# generates routes on saturdays, array below stores that get stock on saturday 
satW_stores = []

for store in west_stores:
    z = west[west.Store == store]
    if z.Saturday.values[0] != 0:
        satW_stores.append(store)

for i in range (2,6):
    for subset in itertools.combinations(satW_stores, i):
        ds = 0
        for node in subset:
            z = west[west.Store == node]
            ds += z.Saturday.values[0]
        if ds <= 26:
            sW.append(subset)



# EAST STORES
east_stores = np.array(east.Store)
# intialise valid routes for east regions for each day
mE = []
tuE = []
wE = []
thE = []
fE = []
sE = []

# generates routes for weekdays in size 2 to 4
for i in range (2,4):
    for subset in itertools.combinations(east_stores, i):
        dm = 0
        dtu = 0
        dw = 0
        dth = 0
        df = 0
        for node in subset:
            z = east[east.Store == node]
            dm += z.Monday.values[0]
            dtu += z.Tuesday.values[0]
            dw += z.Wednesday.values[0]
            dth += z.Thursday.values[0]
            df += z.Friday.values[0]
        if dm <= 26:
            mE.append(subset)
        if dtu <= 26:
            tuE.append(subset)
        if dw <= 26:
            wE.append(subset)
        if dth <= 26:
            thE.append(subset)
        if df <= 26:
            fE.append(subset)

# generates routes on saturdays, array below stores that get stock on saturday 
satE_stores = []

for store in east_stores:
    z = east[east.Store == store]
    if z.Saturday.values[0] != 0:
        satE_stores.append(store)

for i in range (2,6):
    for subset in itertools.combinations(satE_stores, i):
        ds = 0
        for node in subset:
            z = east[east.Store == node]
            ds += z.Saturday.values[0]
        if ds <= 26:
            sE.append(subset)


# NORTH STORES
north_stores = np.array(north.Store)
# intialise valid routes for north regions for each day
mN = []
tuN = []
wN = []
thN = []
fN = []
sN = []

# generates routes for weekdays in size 2 to 4
for i in range (2,4):
    for subset in itertools.combinations(north_stores, i):
        dm = 0
        dtu = 0
        dw = 0
        dth = 0
        df = 0
        for node in subset:
            z = north[north.Store == node]
            dm += z.Monday.values[0]
            dtu += z.Tuesday.values[0]
            dw += z.Wednesday.values[0]
            dth += z.Thursday.values[0]
            df += z.Friday.values[0]
        if dm <= 26:
            mN.append(subset)
        if dtu <= 26:
            tuN.append(subset)
        if dw <= 26:
            wN.append(subset)
        if dth <= 26:
            thN.append(subset)
        if df <= 26:
            fN.append(subset)

# generates routes on saturdays, array below stores that get stock on saturday 
satN_stores = []

for store in north_stores:
    z = north[north.Store == store]
    if z.Saturday.values[0] != 0:
        satN_stores.append(store)

for i in range (2,6):
    for subset in itertools.combinations(satN_stores, i):
        ds = 0
        for node in subset:
            z = north[north.Store == node]
            ds += z.Saturday.values[0]
        if ds <= 26:
            sN.append(subset)


# SOUTHERNMOST STORES
southernmost_stores = np.array(southernMost.Store)
# intialise valid routes for southermost regions for each day
mSM = []
tuSM = []
wSM = []
thSM = []
fSM = []
sSM = []

# generates routes for weekdays in size 2 to 4
for i in range (2,4):
    for subset in itertools.combinations(southernmost_stores, i):
        dm = 0
        dtu = 0
        dw = 0
        dth = 0
        df = 0
        for node in subset:
            z = southernMost[southernMost.Store == node]
            dm += z.Monday.values[0]
            dtu += z.Tuesday.values[0]
            dw += z.Wednesday.values[0]
            dth += z.Thursday.values[0]
            df += z.Friday.values[0]
        if dm <= 26:
            mSM.append(subset)
        if dtu <= 26:
            tuSM.append(subset)
        if dw <= 26:
            wSM.append(subset)
        if dth <= 26:
            thSM.append(subset)
        if df <= 26:
            fSM.append(subset)

# generates routes on saturdays, array below stores that get stock on saturday 
satSM_stores = []

for store in southernmost_stores:
    z = southernMost[southernMost.Store == store]
    if z.Saturday.values[0] != 0:
        satSM_stores.append(store)

for i in range (2,6):
    for subset in itertools.combinations(satSM_stores, i):
        ds = 0
        for node in subset:
            z = southernMost[southernMost.Store == node]
            ds += z.Saturday.values[0]
        if ds <= 26:
            sSM.append(subset)
