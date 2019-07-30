#This modules retrieves accurate OHLCV data from binances and converts to requisite pandas dataframe for use by the CNN


import ccxt
import pandas as pd
import numpy

binance = ccxt.binance()

def binance_to_df(pair, startdate):
	since = binance.parse8601(startdate)
	OHLCV=binance.fetch_ohlcv(pair, '1d', since)
	# for day in OHLCV
	# 	pass
	df = pd.DataFrame(OHLCV, columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
	#pd.to_datetime(df.Date, format='%Y%m%d')
	df['Date'] = pd.to_datetime(df['Date'],unit='ms')
	df.set_index('Date', inplace=True, drop=True)
	df['Adj Close'] = df['Close']
	return df

#data = binance_to_df('BTC/USDT', '2018-07-24T00:00:00Z')
#print(data)
# def TrackUpdateTrades(signal):
# 	 return 1 if associatewith(signal) else 

def trade_binance(signal, pair):
	if signal == 'buy':
		balance = binance.fetch_balance()['free']['USDT']
		price = binance.fetch_ticker(pair)['bid']
		amount = (balance *0.5)/price
		#order = binance.create_order()
		TrackUpdateTrades(order['id'])

def create_Xt_Yt(X, y, percentage=0.9): 

	p = int(len(X) * percentage) 
	X_train = X[0:p] 
	Y_train = y[0:p] 
	X_train, Y_train = shuffle_in_unison(X_train, Y_train) 
	X_test = X[p:] 
	Y_test = y[p:] 	

	return X_train, X_test, Y_train, Y_test

def shuffle_in_unison(a, b):
    assert len(a) == len(b)
    shuffled_a = numpy.empty(a.shape, dtype=a.dtype)
    shuffled_b = numpy.empty(b.shape, dtype=b.dtype)
    permutation = numpy.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b


