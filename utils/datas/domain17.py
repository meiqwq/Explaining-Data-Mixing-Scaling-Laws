import numpy as np
import pandas as pd
import json
mixture = pd.read_csv("data/test_mixture_1B.csv")
loss_val = pd.read_csv("data/test_pile_loss_1B.csv")
dic={
    "train_the_pile_nih_exporter":"NIH ExPorter",
    "train_the_pile_philpapers":"PhilPapers",
    "train_the_pile_enron_emails":"Enron Emails",
    "train_the_pile_europarl":"EuroParl",
}   

H_data=[]
L_data=[]
h_=0
missing={}
with open("data/eval.jsonl","r", encoding="utf-8") as f:
    for line in f:
        obj=json.loads(line)
        missing[(obj["index"]-1,obj["domain"])]=obj["avg_loss"]

for (i1,row),(i2,row2) in zip(mixture.iterrows(), loss_val.iterrows()):
    h=[]
    l=[]
    for domain in mixture.columns[1:]:
        h.append(row[domain])
        #sprint(row)
        if domain in dic:
            l.append(missing[(int(row["index"]),dic[domain])])
        else:
            #print(int(row["index"]),f"metric/{domain[6:]}_val_loss")
            l.append(row2[f"metric/{domain[6:]}_val_loss"])
    H_data.append(h)
    L_data.append(l)

h_data=np.array(H_data)
L_data=np.array(L_data)

if __name__ == "__main__":
    print("h_data:",h_data)
    print("L_data:",L_data)