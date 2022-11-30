import pandas as pd
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from pandas.errors import EmptyDataError  # for ignoring empty files
from sklearn.datasets import make_classification
from sklearn.cluster import DBSCAN

sources = "C:\\Users\\fousk\\OneDrive\\Desktop\\ceid\\project_mining_2022\\pythonProject"
demand = "C:\\Users\\fousk\\OneDrive\\Desktop\\ceid\\project_mining_2022\\pythonProject"
sources_exported_file = "C:\\Users\\fousk\\OneDrive\\Desktop\\ceid\\project_mining_2022\\pythonProject\\sources_united.csv"
demand_exported_file = "C:\\Users\\fousk\\OneDrive\\Desktop\\ceid\\project_mining_2022\\pythonProject\\demand_united.csv"

able_to_be_negative_sources = ["Time","Batteries","Large Hydro merged","Imports","Other"]


# Basic metrics Section
def show_statistic_metrics(df):
    for x in df.columns:
       if x != "Time":
          print(x) 
          print("mean:"+"\t"+str(df[x].mean()))
          print("deviation:"+"\t"+str(df[x].std()))
          print("median:"+"\t"+str(df[x].median()))
          print("mode:"+"\t"+str(df[x].mode()))
    

# Analyze dataset Section
# 1.Fill the missing values - interpolation
def fillin_the_missing_values(df):
    # drop_the_duplicates(df)
    df.drop_duplicates(inplace = True)
    #forward interpolation instead of backward.This way,when all the samples from the start up to a certain point are null, they all shall be filled with the nect following value.
    df.interpolate(method='linear', limit_direction='forward', inplace=True)
    print(df.isnull().sum())


def normalize_outliers(df):
    print("okay")

def dbscan_classification(df):
 dbscan = DBSCAN(eps=10, min_samples=5).fit(df)
 labels = dbscan.labels_
 plt.xlabel("Time")
 plt.ylabel("Spending Score")
 plt.show()        

# unify Csv files Section
def clean_csv(path, exported, type):
    
    files = os.path.join(path, "20*.csv")
    files = glob.glob(files) 
    files_list = []
 
    #put all the files into a list
    for i in range(0, len(files)):
        try:
            pd.read_csv(files[i])
            files_list.append(files[i])
        except EmptyDataError:
            continue
    #unify the list into one
    df = pd.concat(map(pd.read_csv, files_list), ignore_index=True)
    df.to_csv(exported, index=False)  #export to csv
     
    #if we're cleaning the sources, merge the four broken columns and the negative values- wrong values 
    if type == 0: 
     #merge seperated columns
     df['Natural Gas merged'] = df['Natural Gas'].fillna(0) + df['Natural gas'].fillna(0)
     df['Large Hydro merged'] = df['Large Hydro'].fillna(0) + df['Large hydro'].fillna(0)

     #drop function which is used in removing or deleting rows or columns from the CSV files
     df.drop('Natural gas', inplace=True, axis=1)
     df.drop('Natural Gas', inplace=True, axis=1)
     df.drop('Large Hydro', inplace=True, axis=1)
     df.drop('Large hydro', inplace=True, axis=1)  
    
     #Clean the negative values out of specified sources. Fill with none so that the interpolation will fix it later on    
     for x in df.columns:
        if x not in able_to_be_negative_sources:
          for y in df.index:
            if df.loc[y,x] < 0: df.loc[y,x]= None
    
    #correct wrong time format
    df['Time'] = pd.to_datetime(df['Time']).dt.strftime("%H:%M") 
    
    
    #fill in the mising values and drop the duplicates        
    fillin_the_missing_values(df)
    
    dbscan_classification(df)
    
    
    #df.plot(x="Time", y=["Solar", "Wind", "Geothermal","Biomass","Biogas","Small hydro","Coal","Nuclear","Batteries","Large Hydro merged","Natural Gas merged","Imports","Other"])
    #plt.show()

    
    #df.plot(x="Time", y=["Day ahead forecast", "Hour ahead forecast", "Current demand"])
    #plt.show()
    
    #show_statistic_metrics(df)
    
    normalize_outliers(df)
    
    
    df.to_csv(exported, index=False)  # to eniaio csv file

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.scatter(df['Time'], df['Solar'])
    ax.set_xlabel('Time')
    ax.set_ylabel('Solar')
    plt.show()

    return df


clean_csv(sources, sources_exported_file,0) 

#clean_csv(demand,demand_exported_file,1)


# Graphic Representation Section


