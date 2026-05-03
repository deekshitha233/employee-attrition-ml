import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("data/employee.csv")
df = pd.get_dummies(df, drop_first=True)

X = df.drop("PerformanceRating", axis=1)
y = df["PerformanceRating"]

X_train,X_test,y_train,y_test=train_test_split(
X,y,test_size=0.2,random_state=42
)

model = GradientBoostingClassifier()
model.fit(X_train,y_train)

pred = model.predict(X_test)

print("Accuracy:",accuracy_score(y_test,pred))

joblib.dump(model,"models/performance_model.pkl")