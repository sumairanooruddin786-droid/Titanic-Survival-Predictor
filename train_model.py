import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# 1. Dataset Load Karein
df = pd.read_csv('train.csv')

# 2. Missing Values Fill Karein
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Fare'].fillna(df['Fare'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# 3. Categorical Data ko Numbers mein Convert Karein
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

# 4. Features aur Target Selection
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
X = df[features]
y = df['Survived']

# 5. Model Train Karein
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# 6. Model ko Save Karein
joblib.dump(model, 'titanic_model.pkl')
print("Model successfully trained and saved as 'titanic_model.pkl'!")