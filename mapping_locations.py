import numpy as np
import pandas as pd
import folium

ORSkey = '88'

locations = pd.read_csv('assignment_resources/WoolworthsLocations.csv')
coords = locations[['Long', 'Lat']]
coords = coords.to_numpy().tolist()

map = folium.Map(location = list(reversed(coords[2])), zoom_start=10)

for i in range(0, len(coords)):

    if locations.Type[i]=="Countdown":
        iconCol="green"
    elif locations.Type[i]=="FreshChoice":
        iconCol="blue"
    elif locations.Type[i]=="SuperValue":
        iconCol="red"
    elif locations.Type[i]=="Countdown Metro":
        iconCol="orange"
    elif locations.Type[i]=="Distribution  Centre":
        iconCol= "black"
    folium.Marker(list(reversed(coords[i])), popup = locations.Store[i], icon = folium.Icon(color = iconCol)).add_to(map)

#display map
map.save("index.html") ##Open html file to see output