from pandas import read_csv
from pandas import datetime
from matplotlib import pyplot as plt

def parser(x):
	return datetime.strptime('190'+x, '%Y-%m')

series = read_csv('./data/sales-of-shampoo-over-a-three-ye.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
upsampled  = series.resample('D')
interpolated = upsampled.interpolate(method='spline', order=2)
resampled1 = series.resample('Q')
quarterly_mean_sales = resampled1.mean()
resampled2 = series.resample('A')
annual_mean_sales = resampled2.mean()
print(series.head(32))
print('---')
print(upsampled.head(32))
print('---')
print(interpolated.head(32))
print('---')
print(quarterly_mean_sales.head())
print('---')
print(annual_mean_sales.head())
# interpolated.plot()
# plt.show()