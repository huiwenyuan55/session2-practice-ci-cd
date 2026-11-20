import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# Load training data
data = pd.read_csv('training_data.csv')
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Train a simple linear regression model
model = LinearRegression()
model.fit(X, y)

# Save the trained model object
joblib.dump(model, 'linear_model.pkl')

# Save coefficients and intercept as the release artifact
with open('linear_model.txt', 'w', encoding='utf-8') as f:
    f.write(f'Coefficients: {model.coef_}\n')
    f.write(f'Intercept: {model.intercept_}\n')

print('Training complete.')
print(f'Coefficients: {model.coef_}')
print(f'Intercept: {model.intercept_}')
