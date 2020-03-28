import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
import seaborn as sns


def to_date(x):
    x = x.split('/')  #
    return dt.datetime(year=int("20" + x[2]), month=int(x[0]), day=int(x[1]))


sp = pd.read_csv('sp.csv', delimiter=r",")
sp = sp[['Date', 'Open', 'Close', 'Volume']]

cd = pd.read_csv('cd.csv', delimiter=r",")
cd = cd.rename(columns={
    "DateRep": "Date",
    "Countries and territories": "Country",
    "Cases": "Daily_Cases"}, errors="raise")
cd['Date'] = cd['Date'].apply(to_date)
sp['Date'] = pd.to_datetime(sp['Date'])

cd_us = cd[['Date', 'Daily_Cases', 'Deaths', 'Country', 'GeoId']].query("GeoId == 'US'")
cd_world = cd[['Date', 'Daily_Cases', 'Deaths', 'Country', 'GeoId']].groupby(['Date'])['Daily_Cases'].sum()

# print(cd_world.dtypes)

result_us = pd.merge(sp, cd_us, on='Date')
result_world = pd.merge(sp, cd_world, on='Date')
# print(result_world.dtypes)

result_us = result_us[['Date', 'Close', 'Daily_Cases']]
result_world = result_world[['Date', 'Close', 'Daily_Cases']]
# print(result_world)

sns.lineplot(x="Date", y="Close", data=sp)
plt.ylabel('S&P Close Price')
plt.xlabel('Date')
plt.show()
plt.clf()
#
# sns.lineplot(x="Date", y="Daily_Cases", data=result_us)
# plt.ylabel('Daily Corona Cases in the U.S.')
# plt.xlabel('Date')
# plt.show()
# plt.clf()
#
# sns.lineplot(x="Date", y="Daily_Cases", data=result_world)
# plt.ylabel('Daily Corona Cases Worldwide')
# plt.xlabel('Date')
# plt.show()
# plt.clf()
#
# result_us_sortedbycase = result_us.sort_values(['Daily_Cases']).reset_index(drop=True)
# sns.scatterplot('Daily_Cases', 'Close', data=result_us)
# plt.show()
# plt.clf()
#
# result_world_sortedbycase = result_world.sort_values(['Daily_Cases']).reset_index(drop=True)
# sns.scatterplot('Daily_Cases', 'Close', data=result_world)
# plt.show()
# plt.clf()

# Cumulative sum

result_world['Total_Cases'] = result_world['Daily_Cases'].cumsum(skipna=False)
result_world50k = result_world[result_world['Total_Cases'] > 58000]
print(result_world50k)
result_us['Total_Cases'] = result_us['Daily_Cases'].cumsum(skipna=False)
# print(result_us)

# sns.lineplot(x="Date", y="Total_Cases", data=result_us)
# plt.ylabel('Total Corona Cases in the U.S.')
# plt.xlabel('Date')
# plt.show()
# plt.clf()
#
sns.lineplot(x="Date", y="Total_Cases", data=result_world)
plt.ylabel('Total Corona Cases Worldwide')
plt.xlabel('Date')
plt.show()
plt.clf()
#
# result_us_sortedbycase = result_us.sort_values(['Total_Cases']).reset_index(drop=True)
# sns.scatterplot('Total_Cases', 'Close', data=result_us)
# plt.show()
# plt.clf()
#

# result_world_sortedbycase = result_world.sort_values(['Total_Cases']).reset_index(drop=True)
# sns.scatterplot('Total_Cases', 'Close', data=result_world50k)
# plt.show()
# plt.clf()

# a very ugly plot to delete which is the list of daily cases by country
# sns.lineplot(x="Date", y="Daily_Cases", hue="Country", data=cd)
# plt.ylabel('Total Corona Cases in the U.S.')
# plt.xlabel('Date')
# plt.show()
# plt.clf()

msk = np.random.rand(len(result_world50k)) < 0.65
train = result_world50k[msk]
test = result_world50k[~msk]

x_train = np.asanyarray(train[['Total_Cases']])
y_train = np.asanyarray(train[['Close']])

x_test = np.asanyarray(test[['Total_Cases']])
y_test = np.asanyarray(test[['Close']])

# Polynomial Features
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from sklearn import metrics

# Applying Polynomial features with degree 2 and calculate prediction values.
# Then evaluate through MSE, RMSE and R2 score
poly_1 = PolynomialFeatures(degree=2)
x_train_poly = poly_1.fit_transform(x_train)

clf_1 = linear_model.LinearRegression()
train_y_ = clf_1.fit(x_train_poly, y_train)
predicted_value_clf_1 = clf_1.predict(poly_1.fit_transform(x_test))
error_mse_deg_2 = metrics.mean_squared_error(y_test, predicted_value_clf_1)
error_rmse_deg_2 = np.sqrt(metrics.mean_squared_error(y_test, predicted_value_clf_1))
r2_score_deg_2 = metrics.r2_score(y_test, predicted_value_clf_1)
print("Linear 2 degree")
print(clf_1.intercept_)
print(clf_1.coef_)
print("Mean squared Error for Polynomial with 2 degree:", error_mse_deg_2)
print("Root mean squared error for poly with 2 degree:", error_rmse_deg_2)
print("R2 Score for Poly with 2 degree", r2_score_deg_2)

# Applying Polynomial features with degree 3 and calculate prediction values.
# Then evaluate through MSE, RMSE and R2 score
poly_2 = PolynomialFeatures(degree=3)
x_train_poly = poly_2.fit_transform(x_train)

clf_2 = linear_model.LinearRegression()
train_y_ = clf_2.fit(x_train_poly, y_train)
predicted_value_clf_2 = clf_2.predict(poly_2.fit_transform(x_test))
error_mse_deg_3 = metrics.mean_squared_error(y_test, predicted_value_clf_2)
error_rmse_deg_3 = np.sqrt(metrics.mean_squared_error(y_test, predicted_value_clf_2))
r2_score_deg_3 = metrics.r2_score(y_test, predicted_value_clf_2)
print("Linear 3 degree")
print(clf_2.intercept_)
print(clf_2.coef_)
print("Mean squared Error for Polynomial with 3 degree:", error_mse_deg_3)
print("Root mean squared error for poly with 2 degree:", error_rmse_deg_3)
print("R2 Score for Poly with 3 degree", r2_score_deg_3)


def f_1(x):
    return clf_1.intercept_[0] + clf_1.coef_[0][1] * x + clf_1.coef_[0][2] * x ** 2


def f_2(x):
    return clf_2.intercept_[0] + clf_2.coef_[0][1] * x + clf_2.coef_[0][2] * x ** 2 + clf_2.coef_[0][3] * x ** 3


# Plot the final graph of applied model with legend.
plt.scatter(result_world50k.Total_Cases, result_world50k.Close, color='blue', label='Train values')
plt.scatter(x_test, predicted_value_clf_1, color="yellow", label='Predicted val 2 degree')
plt.scatter(x_test, predicted_value_clf_2, color='purple', label='Predicted val 3 degree')
xx = np.arange(58 * 10 ** 3, 4 * 10 ** 5)
plt.plot(xx, f_1(xx), '-r', label='degree 2')
plt.plot(xx, f_2(xx), '-g', label='degree 3')
plt.xlabel('Total Cases')
plt.ylabel('Close')
plt.legend(loc='lower left')
plt.show()

