import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import joblib

data = pd.read_csv("housing.csv")
data = data.dropna(axis=0)

x_feature = ['longitude', 'latitude', 'housing_median_age', 'total_bedrooms', 'population', 'households', 'median_income', 'ocean_proximity']
y_feature = ['median_house_value']

x = data[x_feature]
y = data[y_feature]

x = pd.get_dummies(x, columns=['ocean_proximity'])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

model = RandomForestRegressor()

param_grid = {
    'n_estimators': [300, 400, 500, 600],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(x_train, y_train)
best_model = grid_search.best_estimator_
joblib.dump(model.fit(x_train, y_train), 'best_random_forest_model.pkl')

print(best_model, '\n')
print(best_model.score(x_test, y_test), '\n')

pred = best_model.predict(x_test)
print(mean_absolute_error(y_test, pred))

seaborn.scatterplot(data = data, x = 'latitude', y = 'longitude', hue='median_house_value')
plt.show()

model.fit(x_train, y_train)

pred = model.predict(x_test)

print(r2_score(y_test, pred))

plt.scatter(x['latitude'], y, color='blue', label='Original Data (Feature 1)')
plt.scatter(x['longitude'], y, color='blue', label='Original Data (Feature 2)')
plt.scatter(x['housing_median_age'], y, color='blue', label='Original Data (Feature 3)')
plt.scatter(x['total_bedrooms'], y, color='blue', label='Original Data (Feature 4)')
plt.scatter(x['population'], y, color='blue', label='Original Data (Feature 5)')
plt.scatter(x['households'], y, color='blue', label='Original Data (Feature 6)')
plt.scatter(x['median_income'], y, color='blue', label='Original Data (Feature 7)')

plt.plot(x, model.predict(x), color='red', label='RandomForestRegressor')
plt.title('RandomForestRegressor')
# plt.legend()
plt.show()
