import os
import pandas as pd
import numpy as np
def merge_submissions(submit_dir_path):
     # Get all .csv files in the folder
    csv_files = [file for file in os.listdir(submit_dir_path) if file.endswith('.csv')]

    # Initialize an empty list to store dataframes
    dfs = []

    # Iterate over each csv file and load it into a dataframe
    for csv_file in csv_files:
        file_path = os.path.join(submit_dir_path, csv_file)
        df = pd.read_csv(file_path)
        dfs.append(df)

    to_merge = [[] for i in range(len(dfs[0]))]
    
    for df in dfs:
        array = df.values
        for i, row in enumerate(array):
            for j, elem in enumerate(row):
                # i is the row index, j is the column index
                if j==1:
                    for q,id_string in enumerate(elem.split(' ')):
                        to_merge[i].append([id_string,60-q])
    s_pred = []
    for i,row in enumerate(to_merge):
        saw_ids = {}   
        for elem in row:
            id_geo, score = elem
            if id_geo in saw_ids:
                saw_ids[id_geo] += score
            else:
                saw_ids[id_geo] = score
        a = sorted(saw_ids.items(), key=lambda x: x[1], reverse=True)
        b=[elem[0] for elem in a]
        s_pred.append(" ".join(b[:30])) 
        sdsdsd=3
    df = pd.DataFrame(
        {
            "Id": dfs[0]["Id"],
            "Predicted": s_pred,
        }
    )
    df.to_csv(os.path.join(submit_dir_path,"submission_merged.csv"), index=False)
    
         
if __name__ =="__main__":
    merge_submissions("/usr/users/cei2023_2024_sondra_cself/coscoy_rem/Documents/deepchallenge4-team4/submits_to_merge")
