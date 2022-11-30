import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('sources_united.csv')
def figplot(x):
    df[x].plot()
    title = x + " Plot"
    #plt.show
    plt.savefig(title + '.png')

figplot("Large Hydro merged")
#Biomass,Biogas,Small hydro,Coal,Nuclear,Batteries,Imports,Other,Natural Gas merged,Large Hydro merged