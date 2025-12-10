import pandas as pd

df = pd.read_csv("data/metadata/UrbanSound8K.csv")
id_to_name = df.groupby("classID")["class"].first().to_dict()
print(id_to_name)