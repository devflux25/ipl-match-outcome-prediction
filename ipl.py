import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


path = r""  #enter the path of the downloded file 
df = pd.read_csv(path)




df = df[['team1', 'team2', 'winner', 'venue', 'toss_winner', 'toss_decision','date']]


df = df.dropna(subset=['winner'])


X = df[['team1', 'team2', 'venue', 'toss_winner', 'toss_decision']]
y = (df['winner'] == df['team1']).astype(int)

print("Class distribution:")
print(y.value_counts(), "\n")
class_counts = y.value_counts().sort_index()
list1 = ['Team2 Wins','Team1 Wins']
values = class_counts.values
plt.figure(figsize=(6,4))
plt.bar(list1,values,edgecolor='black')
plt.xlabel("Match Outcome")
plt.ylabel("Number of Matches")
plt.title("Class Distribution: Team1 vs Team2 Wins")
plt.savefig("class_distribution.png", dpi=300, bbox_inches='tight')
plt.close()



encoder = OneHotEncoder(
    sparse_output=False,
    handle_unknown='ignore'
)

X_encoded = encoder.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(
    X_encoded,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Model accuracy: {accuracy:.4f}")

baseline_accuracy = y.value_counts(normalize=True).max()
print(f"Baseline accuracy: {baseline_accuracy:.4f}")

list2 = [accuracy,baseline_accuracy]

labels = ['Baseline', 'Logistic Regression']
values = [baseline_accuracy, accuracy]
 
plt.figure(figsize=(6,4))
plt.bar(labels, values, edgecolor='black')
plt.ylim(0, 1)
plt.ylabel('Accuracy')
plt.title('Baseline vs Logistic Regression Accuracy')

plt.savefig("accuracy_comparison.png", dpi=300, bbox_inches='tight')
plt.close()

#random forest and adding more features to improve accuracy 

df = df.sort_values(by ="date",ascending=True)
df = df.reset_index(drop=True)


print(sorted(df["team1"].dropna().unique()))
print(sorted(df["team2"].dropna().unique()))
print(sorted(df["winner"].dropna().unique()))



team_mapping = {
    "Chennai Super Kings": "CSK",
    "CSK": "CSK",

    "Mumbai Indians": "MI",

    "Royal Challengers Bangalore": "RCB",

    "Kolkata Knight Riders": "KKR",

    "Kings XI Punjab": "PBKS",
    "Punjab Kings": "PBKS",

    "Delhi Daredevils": "DC",
    "Delhi Capitals": "DC",

    "Sunrisers Hyderabad": "SRH",

    "Rajasthan Royals": "RR",

    "Gujarat Lions": "GL",
    "Rising Pune Supergiant": "RPS",
    "Rising Pune Supergiants": "RPS",
    "Pune Warriors": "PW",
    "Deccan Chargers": "DCG",
    "Kochi Tuskers Kerala": "KTK"
}
valid_teams = ["CSK", "MI", "RCB", "PBKS", "SRH", "DC", "RR", "KKR"]


df["team1"] = df["team1"].replace(team_mapping)
df["team2"] = df["team2"].replace(team_mapping)
df["winner"] = df["winner"].replace(team_mapping)

df = df[df["team1"].isin(valid_teams)]
df = df[df["team2"].isin(valid_teams)]

team_stats = {
    
}
team1_win_rate = []
team2_win_rate = []


for i, row in df.iterrows():
    team1 = row["team1"]
    team2 = row["team2"]
    winner = row["winner"]

    if team1 not in team_stats:
        team_stats[team1] = {"wins": 0, "matches": 0}

    if team2 not in team_stats:
        team_stats[team2] = {"wins": 0, "matches": 0}


    if(team_stats[team1]['matches']>0):
        team1_rate = team_stats[team1]["wins"]/team_stats[team1]["matches"]
    else:
        team1_rate=0
    if(team_stats[team2]['matches']>0):
        team2_rate = team_stats[team2]["wins"]/team_stats[team2]["matches"]
    else:
        team2_rate=0
    team1_win_rate.append(team1_rate)
    team2_win_rate.append(team2_rate)

    team_stats[team1]["matches"] +=1
    team_stats[team2]["matches"] +=1

    if(winner==team1):
        team_stats[team1]["wins"] +=1
    elif(winner==team2):
        team_stats[team2]["wins"] +=1

print(team_stats)

df["team1_win_rate"] = team1_win_rate
df["team2_win_rate"] = team2_win_rate

recent_results = {

}
team1_last5 = [ 

]
team2_last5 =[

]


for i, row in df.iterrows():
    t1 = row["team1"]
    t2 = row["team2"]
    winner = row["winner"]

    if t1 not in recent_results:
        recent_results[t1] = []

    if t2 not in recent_results:
        recent_results[t2] = []
    t1_form = sum(recent_results[t1][-5:])
    t2_form = sum(recent_results[t2][-5:])
    team1_last5.append(t1_form)
    team2_last5.append(t2_form)

    if(winner == t1):
        recent_results[t1].append(1)
        recent_results[t2].append(0)
    elif(winner == t2):
        recent_results[t1].append(0)
        recent_results[t2].append(1)


df['team1_recent_form'] = team1_last5
df['team2_recent_form'] = team2_last5

h2h = {}

team1_h2h_wins = []
team2_h2h_wins = []

team1_h2h_win_rate =[]
team2_h2h_win_rate = []

for i, row in df.iterrows():
    t1 = row["team1"]
    t2 = row["team2"]
    winner = row["winner"]

    pair = tuple(sorted([t1, t2]))

    if pair not in h2h:
        h2h[pair] = {
            pair[0]: 0,
            pair[1]: 0
        }

    t1_h2h = h2h[pair][t1]
    t2_h2h = h2h[pair][t2]

    team1_h2h_wins.append(t1_h2h)
    team2_h2h_wins.append(t2_h2h)

    # update AFTER match
    if winner == t1:
        h2h[pair][t1] += 1
    elif winner == t2:
        h2h[pair][t2] += 1

total_matches = h2h[pair][t1] + h2h[pair][t2]
team1_h2h_win_rate = h2h[pair][t1]/total_matches
team2_h2h_win_rate = h2h[pair][t2]/total_matches

df['team1_h2h_win_rate'] = team1_h2h_win_rate
df['team2_h2h_win_rate'] = team2_h2h_win_rate


model1 = RandomForestClassifier(
    n_estimators=200,
    #max_features=
    max_depth= 10,
    #max_leaf_nodes=
    min_samples_split=5,
    random_state= 42 ,
)

features = [
    "team1_win_rate",
    "team2_win_rate",
    'team1_recent_form',
    "team2_recent_form",
    'team1_h2h_win_rate',
    'team2_h2h_win_rate'
    'venue',
    'toss_winner',
    'toss_decision'
]

X1 = df[['team1', 'team2',  'venue', 'toss_winner', 'toss_decision',"team1_win_rate","team2_win_rate",
         'team1_recent_form','team2_recent_form','team1_h2h_win_rate','team2_h2h_win_rate']]

Y1 = (df['winner'] == df['team1']).astype(int)

X1_encoded = encoder.fit_transform(X1)

X1_train,X1_test,Y1_train,Y1_test = train_test_split(
    X1_encoded,
    Y1,
    train_size=0.8
)

model1.fit(X1_train,Y1_train)
Y1_pred = model1.predict(X1_test)

accuracy1 = accuracy_score(Y1_test, Y1_pred)

print(f"Model accuracy: {accuracy1:.4f}")

list3 = [accuracy,accuracy1]

labels = ['Baseline','Logistic Regression', 'RandomForestClassifier']
values = [baseline_accuracy,accuracy, accuracy1]
 
plt.figure(figsize=(6,4))
plt.bar(labels, values, edgecolor='black')
plt.ylim(0, 1)
plt.ylabel('Accuracy')
plt.title('Model Accuracy Comparison')
plt.savefig("accuracy_comparison.png", dpi=300, bbox_inches='tight')
plt.show()
