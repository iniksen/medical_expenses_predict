import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


#mikahymbedonim chand nafar zende mondan?

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

print(train.head())
print(train.info())
print(train.describe())

#soton haye adadi? panssange survivied pclass age sibsp parch fare
#soton haye matni?name sex ticket cabin embarked
#soton hayee ke missing value daran?  age cabin  embarked

#zan ha bishtar zende mondan ya mard ha?

sns.countplot(x="Sex", hue="Survived", data=train)
plt.title("Survival by Sex")
plt.show()
#zan ha ishtar zende mondan chon ye ganon naneveshte vojod dasht women and children first
#kodom kelas bishtar zende mon?
#class1 yani tabageye morafahe keshti bedalil nazdiki be gayeg haye nejat va...


#alan mirirm soragh missing value ha
"""
| Age      | پر کردن با **Median** |
| Embarked | پر کردن با **Mode**   |
| Cabin    | حذف کامل ستون         |

"""
#age median
train["Age"].fillna(train["Age"].median(), inplace=True)
test["Age"].fillna(test["Age"].median(), inplace=True)
#inplace=True yani train ro be in shekl avaz kon

# Embarked → Mode
train["Embarked"].fillna(train["Embarked"].mode()[0], inplace=True)

# Drop Cabin
train.drop("Cabin", axis=1, inplace=True)
test.drop("Cabin", axis=1, inplace=True)
print(train.isnull().sum())
"""
alan dige missing value nadarim
PassengerId    0
Survived       0
Pclass         0
Name           0
Sex            0
Age            0
SibSp          0
Parch          0
Ticket         0
Fare           0
Embarked       0
"""


#tabdil sex be adad
train["Sex"]=train["Sex"].map({"male":0,"female":1})
test["Sex"]=test["Sex"].map({"male":0,"female":1})


train = pd.get_dummies(train, columns=["Embarked"], drop_first=True)
test = pd.get_dummies(test, columns=["Embarked"], drop_first=True)

drop_cols = ["PassengerId", "Name", "Ticket"]
train = train.drop(drop_cols, axis=1)
test = test.drop(drop_cols, axis=1)
print(train.head())
print(train.info())

X = train.drop("Survived", axis=1)
y = train["Survived"]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_val)

print("Accuracy:", accuracy_score(y_val, y_pred))
print(confusion_matrix(y_val, y_pred))
print(classification_report(y_val, y_pred))

# ====== KAGGLE SUBMISSION (FIXED) ======

# دوباره test.csv رو بخون
test = pd.read_csv("test.csv")

# PassengerId رو نگه دار
passenger_ids = test["PassengerId"]

# Age
test["Age"].fillna(test["Age"].median(), inplace=True)
test["Fare"].fillna(test["Fare"].median(), inplace=True)


# Sex
test["Sex"] = test["Sex"].map({"male": 0, "female": 1})

# Embarked  (خیلی مهم: قبل از get_dummies)
test["Embarked"].fillna("S", inplace=True)

# One-hot encoding
test = pd.get_dummies(test, columns=["Embarked"], drop_first=True)

# حذف ستون‌های اضافی
test.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1, inplace=True)

# پیش‌بینی
predictions = model.predict(test)

# ساخت فایل submission
submission = pd.DataFrame({
    "PassengerId": passenger_ids,
    "Survived": predictions
})

submission.to_csv("titanic_submission.csv", index=False)

sns.heatmap(confusion_matrix(y_val, y_pred), annot=True, fmt="d")
plt.title("Confusion Matrix")
plt.show()








