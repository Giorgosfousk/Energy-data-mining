import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('sources_united.csv')
def figplot(col):
    fig, ax = plt.subplots(figsize=(12,6))
    ax.scatter(df['Time'], df[col])
    ax.set_xlabel('Time')
    ax.set_ylabel(col)
    #plt.show()
    title = col + " Scatter"
    plt.savefig(title+'.png')

for x in ("Solar","Wind","Geothermal","Biomass","Biogas","Small hydro","Coal","Nuclear","Batteries","Imports","Other","Natural Gas merged","Large Hydro merged"):
    figplot(x)