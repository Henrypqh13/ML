
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, mean_absolute_error, precision_score


def convert_ftr(ftr):
    if ftr == "H":
        return 1
    if ftr == "A":
        return -1
    else:
        return 0
    

def cleandata(df):
    df["FTR"] = df["FTR"].apply(convert_ftr)
    df = df.dropna()
    return df

def get_dis(df):
   df["XGD"] = df["HxG"] - df["AxG"]
   return df

def team_x_points(df):
    df["XPD"] = df["HxPTS"] - df["AxPTS"]
    return df

model = KNeighborsClassifier(9)


train_set = ["2019-2020.csv","2018-2019.csv"]

for file in train_set:
    df_train = pd.read_csv(file)
    df_train = cleandata(df_train)
    df_train = get_dis(df_train)
    df_train = team_x_points(df_train)
    y_train = df_train["FTR"]
    #x_train = df_train.drop(["Date", "Div", "HomeTeam", "AwayTeam", "HTR", ], axis = 1)
    x_train = df_train[["XGD","XPD"]]
    model.fit(x_train,y_train)

df_test = pd.read_csv("2021-2022.csv")
df_test = cleandata(df_test)
df_test = get_dis(df_test)
df_test= team_x_points(df_test)
x_test = df_test[["XGD", "XPD"]]
y_test = df_test["FTR"]

y_pred = model.predict(x_test)
accu_score = accuracy_score(y_pred, y_test)
print(accu_score)


        