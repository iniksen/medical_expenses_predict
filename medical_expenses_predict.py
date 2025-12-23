#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

#load the dataset
df = pd.read_csv("insurance.csv")
print(df.head())

#initial exploration of the data
print(df.info())
print(df.describe())

#check for missing values
print(df.isnull().sum())

#convert categorical variables to numeric using LabelEncoder
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['sex'] = le.fit_transform(df['sex'])       # male/female
df['smoker'] = le.fit_transform(df['smoker']) # yes/no
df['region'] = le.fit_transform(df['region']) # northeast, northwest, etc.

print(df.head()) #now the dataset is fully numeric

#scatter plot: BMI vs Charges, colored by smoker status
plt.figure(figsize=(8,6))
sns.scatterplot(
    data=df,
    x='bmi',
    y='charges',
    hue=df['smoker'].map({0: 'No', 1: 'Yes'}),
    palette={"No":"orange","Yes":"blue"},
)
plt.title("BMI vs Charges (Smoker Status)")
plt.xlabel("BMI")
plt.ylabel("Charges")
plt.show()

#linear relationship between BMI and charges is not strong, so we'll try Polynomial Regression

#split data into features and target
X = df.drop('charges', axis=1)
y = df['charges']

#train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

#Linear Regression model
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

#predictions
y_train_pred_lin = linear_model.predict(X_train)
y_test_pred_lin = linear_model.predict(X_test)

#calculate RMSE and R2
rmse_train_lin = np.sqrt(mean_squared_error(y_train, y_train_pred_lin))
rmse_test_lin = np.sqrt(mean_squared_error(y_test, y_test_pred_lin))
r2_train_lin = r2_score(y_train, y_train_pred_lin)
r2_test_lin = r2_score(y_test, y_test_pred_lin)

#Polynomial Regression (degree 2)
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train)

y_train_pred_poly = poly_model.predict(X_train_poly)
y_test_pred_poly = poly_model.predict(X_test_poly)

rmse_train_poly = np.sqrt(mean_squared_error(y_train, y_train_pred_poly))
rmse_test_poly = np.sqrt(mean_squared_error(y_test, y_test_pred_poly))
r2_train_poly = r2_score(y_train, y_train_pred_poly)
r2_test_poly = r2_score(y_test, y_test_pred_poly)

#display results
results = pd.DataFrame({
    'Model': ['Linear Regression (Degree 1)', 'Polynomial Regression (Degree 2)'],
    'RMSE_Train': [rmse_train_lin, rmse_train_poly],
    'RMSE_Test': [rmse_test_lin, rmse_test_poly],
    'R2_Train': [r2_train_lin, r2_train_poly],
    'R2_Test': [r2_test_lin, r2_test_poly]
})

results

#bar chart to compare RMSE of models
models = ['Degree 1', 'Degree 2']
rmse_train = [rmse_train_lin, rmse_train_poly]
rmse_test = [rmse_test_lin, rmse_test_poly]

x = np.arange(len(models))
width = 0.35

plt.figure(figsize=(8,6))
plt.bar(x - width/2, rmse_train, width, label='Train RMSE')
plt.bar(x + width/2, rmse_test, width, label='Test RMSE')
plt.xticks(x, models)
plt.ylabel('RMSE')
plt.title('RMSE Comparison Between Models')
plt.legend()
plt.show()

#scatter plot: actual vs predicted charges (Polynomial Regression degree 2)
plt.figure(figsize=(7,7))
plt.scatter(y_test, y_test_pred_poly, alpha=0.6, color="purple")
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    linestyle='--', color="black"
)
plt.xlabel('Actual Charges')
plt.ylabel('Predicted Charges')
plt.title('Actual vs Predicted Charges (Polynomial Regression Degree 2)')
plt.show()

#analyze model complexity by varying polynomial degree
degrees = [1, 2, 3, 4]
train_errors = []
test_errors = []

for d in degrees:
    poly = PolynomialFeatures(degree=d, include_bias=False)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    
    model = LinearRegression()
    model.fit(X_train_poly, y_train)
    
    y_train_pred = model.predict(X_train_poly)
    y_test_pred = model.predict(X_test_poly)
    
    train_errors.append(np.sqrt(mean_squared_error(y_train, y_train_pred)))
    test_errors.append(np.sqrt(mean_squared_error(y_test, y_test_pred)))



plt.figure(figsize=(8,6))

# plot train and test RMSE
plt.plot(degrees, train_errors, 
         marker='o', linewidth=2, color='blue', label='Train RMSE')

plt.plot(degrees, test_errors, 
         marker='o', linestyle='--', linewidth=2, color='red', label='Test RMSE')

# highlight optimal degree (degree = 2)
optimal_degree = 2
optimal_rmse = test_errors[degrees.index(optimal_degree)]

plt.scatter(optimal_degree, optimal_rmse, color='green', s=100, zorder=5)

# arrow pointing to optimal degree
plt.annotate(
    'Optimal (Deg 2)',
    xy=(optimal_degree, optimal_rmse),
    xytext=(optimal_degree, optimal_rmse - 600),
    arrowprops=dict(facecolor='green', arrowstyle='->', linewidth=2),
    ha='center',
    fontsize=11
)

# labels and title
plt.xlabel('Polynomial Degree')
plt.ylabel('RMSE Error')
plt.title('Concept: Model Complexity vs Error (Overfitting Analysis)')

plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.show()









