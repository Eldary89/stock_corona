import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt


def to_date(x):
    x = x.split('/')  #
    return dt.datetime(year=int("20" + x[2]), month=int(x[0]), day=int(x[1]))


sp = pd.read_csv('S_P500.csv', delimiter=r",")
sp = sp[['Date', 'Open', 'Close', 'Volume']]

cd = pd.read_csv('COVID19.csv', delimiter=r",")
cd = cd.rename(columns={"DateRep": "Date", "Countries and territories": "Country"}, errors="raise")
cd['Date'] = cd['Date'].apply(to_date)
sp['Date'] = pd.to_datetime(sp['Date'])

cd_us = cd[['Date', 'Cases', 'Deaths', 'Country', 'GeoId']].query("GeoId == 'US'")
cd_world = cd[['Date', 'Cases', 'Deaths', 'Country', 'GeoId']].groupby(['Date'])['Cases'].sum().reset_index(
    name='Total_cases')

print(cd_world)

result_us = pd.merge(sp, cd_us, on='Date')
result_world = pd.merge(sp, cd_world, on='Date')

result_us = result_us[['Date', 'Close', 'Cases']]
result_world = result_world[['Date', 'Close', 'Total_cases']]

print(result_world)

# plt.scatter(result.Date, result.Close, color='blue')
# plt.xlabel('Date')
# plt.ylabel('Close value')
# plt.show()
#
# plt.scatter(result.Date, result.Cases, color='red')
# plt.xlabel('Date')
# plt.ylabel('Covid cases')
# plt.show()
#
plt.scatter(result_world.Total_cases, result_world.Close, color='green')
plt.xlabel('Cases')
plt.ylabel('Close value')
plt.show()
