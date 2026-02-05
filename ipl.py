import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


path = r"C:\Users\Kairav\OneDrive\Desktop\ipl dataset\matches.csv"
df = pd.read_csv(path)


df = df[['team1', 'team2', 'winner', 'venue', 'toss_winner', 'toss_decision']]


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