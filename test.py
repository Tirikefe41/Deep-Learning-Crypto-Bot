from pandas_datareader import data as pdr
import fix_yahoo_finance as yf
import datetime
yf.pdr_override()


start = datetime.datetime(2018,7,16)
end = datetime.datetime(2018,9,18)

data_original = pdr.get_data_yahoo('BTC-USD', start=start, end=end)
print(data_original)