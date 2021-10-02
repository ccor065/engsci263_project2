import numpy as np
import pandas as pd
import folium

ORSkey = '88'

locations = pd.read_csv('stores_df.csv')
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
    folium.Marker(list(reversed(coords[i])), popup ="%s\n lng %s\n lat%s" % (locations.Store[i],locations.Long[i], locations.Lat[i]), icon = folium.Icon(color = iconCol)).add_to(map)

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
            folium.Marker(list(reversed(coords[i])), popup ="%s\n lng %s\n lat%s" % (locations.Store[i],locations.Long[i], locations.Lat[i]), icon = folium.Icon(color = iconCol)).add_to(map)
    map.save("maps/%s_map.html"%name.split()[0])
    


