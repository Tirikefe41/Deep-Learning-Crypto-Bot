



# In general, deep learning refers to neural networks with multiple hidden layers that can learn increasingly abstract representations of the input data.

# Convolutional Neural Networks (CNN’s) are multi-layer neural networks (sometimes up to 17 or more layers) that assume the input data to be images.

# For CNN structure, you generally have:

# Inputs ======(Convolutions)=====> Feature Maps ======== (Subsampling) =========> (feature maps) ========== (Convolutions)=======>(feature maps) ===========(subsampling) =========> (Fully Connected)=======> (Output Layer)

# CNNS have typically been used for image recognition. Such as:

# 1. The first hidden layers might only learn local edge patterns.
# 2. Then, each subsequent layer (or filter) learns more complex representations.
# 3. Finally, the last layer can classify the image as a cat or kangaroo.



# Using Keras CNN (convolutional neural network) as a classifier to predict multivariate time series data (multiple x features, or variables)


# Training on 80% , Testing on 20%


from utils import *

import pandas as pd
import matplotlib.pylab as plt

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.recurrent import LSTM, GRU
from keras.layers import Convolution1D, MaxPooling1D, AtrousConvolution1D, RepeatVector
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from keras.layers.wrappers import Bidirectional
from keras import regularizers
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import *
from keras.optimizers import RMSprop, Adam, SGD, Nadam
from keras.initializers import *
import ffn
import seaborn as sns
sns.despine()
from math import sqrt
import datetime
#yf.pdr_override()


from get_Binance_data import binance_to_df, create_Xt_Yt

start = datetime.datetime(2010,7,16)
end = datetime.datetime(2018,9,18)

data_original = binance_to_df('BTC/USDT', '2018-07-24T00:00:00Z')
print(data_original)

openp = data_original.ix[:, 'Open'].tolist()
highp = data_original.ix[:, 'High'].tolist()
lowp = data_original.ix[:, 'Low'].tolist()
closep = data_original.ix[:, 'Adj Close'].tolist()
volumep = data_original.ix[:, 'Volume'].tolist()


WINDOW = 30 # 30 day windows
EMB_SIZE = 5 # number of features (open, high, low, close, volume)
STEP = 1
FORECAST = 1 # forecasting 1 day out

X, Y = [], []
for i in range(0, len(data_original), STEP):
    try:
        o = openp[i:i+WINDOW]
        h = highp[i:i+WINDOW]
        l = lowp[i:i+WINDOW]
        c = closep[i:i+WINDOW]
        v = volumep[i:i+WINDOW]

  
        o = (np.array(o) - np.mean(o)) / np.std(o)
        h = (np.array(h) - np.mean(h)) / np.std(h)
        l = (np.array(l) - np.mean(l)) / np.std(l)
        c = (np.array(c) - np.mean(c)) / np.std(c)
        v = (np.array(v) - np.mean(v)) / np.std(v)


   
        x_i = closep[i:i+WINDOW] # closing price of x values for the window of 30 days
        y_i = closep[i+WINDOW+FORECAST] # closing price of y value (close) for the window of 30 days, + a forecast of 1 day out

        last_close = x_i[-1] # previous day's close
        next_close = y_i # predicted future close 1 day out

        if last_close < next_close: # if next (or future close price 1 day out) was greater than prior day's close price.....
            y_i = [1, 0] # price went up with 100% probability
        else:
            y_i = [0, 1] # price went down with 100% probability


        x_i = np.column_stack((o, h, l, c, v)) # stack columns for open, high, low, close, and volume

    except Exception as e:
        print("Encountered Exception ")
        break

    X.append(x_i)
    Y.append(y_i)


X, Y = np.array(X), np.array(Y)
X_train, X_test, Y_train, Y_test = create_Xt_Yt(X, Y) # utilize function from utils.py to create x train, x test, y train, y test , see method 'def create_Xt_Yt' in utils.py: we are training on 90% , and testing on 10%

# scale data
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], EMB_SIZE))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], EMB_SIZE))



# TECHNICAL INDICATORS


# SIGNALS
data_original['Stdev'] = data_original['Adj Close'].rolling(window=90).std()  # calculate rolling std
data_original['SMA'] = data_original['Adj Close'].rolling(100).mean()  # calculate n tick SMA

# Bollinger Bands
data_original['Upper Band'] = data_original['SMA'] + (data_original['Stdev'] * 1)  # 1 standard deviations above
data_original['Lower Band'] = data_original['SMA'] - (data_original['Stdev'] * 1)  # 1 standard deviations below



# DEMA
window = 20
data_original['ema'] = pd.ewma(data_original['Adj Close'], span=window, min_periods=window-1)
data_original['ema_of_ema'] = pd.ewma(data_original['ema'], span=window, min_periods=window-1)

data_original['dema'] = (2 * data_original['ema']) - data_original['ema_of_ema']



# MACD 12,26,9
data_original['stock_df_12_ema'] = pd.ewma(data_original['Close'], span=12)
data_original['stock_df_26_ema'] = pd.ewma(data_original['Close'], span=26)
data_original['stock_df_macd_12_26'] = data_original['stock_df_12_ema'] - data_original['stock_df_26_ema']
data_original['stock_df_signal_12_26'] = pd.ewma(data_original['stock_df_macd_12_26'], span=9)
data_original['stock_df_crossover_12_26'] = data_original['stock_df_macd_12_26'] - data_original['stock_df_signal_12_26'] # means, if this is > 0, or stock_df['Crossover'] =  stock_df['MACD'] - stock_df['Signal'] > 0, there is a buy signal
                                                                               # means, if this is < 0, or stock_df['Crossover'] =  stock_df['MACD'] - stock_df['Signal'] < 0, there is a sell signal



# RSI
def RSI(series, period):
    delta = series.diff().dropna()
    u = delta * 0
    d = u.copy()
    u[delta > 0] = delta[delta > 0]
    d[delta < 0] = -delta[delta < 0]
    u[u.index[period - 1]] = np.mean(u[:period])  # first value is sum of avg gains
    u = u.drop(u.index[:(period - 1)])
    d[d.index[period - 1]] = np.mean(d[:period])  # first value is sum of avg losses
    d = d.drop(d.index[:(period - 1)])
    rs = pd.stats.moments.ewma(u, com=period - 1, adjust=False) / \
            pd.stats.moments.ewma(d, com=period - 1, adjust=False)
    return 100 - 100 / (1 + rs)


data_original['RSI'] = RSI(data_original['Adj Close'],14)  # RSI function of series defined by the close price, and period of choosing (defaulted to 14)

# calculate daily returns
data_original['returns'] = np.log(data_original['Adj Close'] / data_original['Adj Close'].shift(1))
data_original['returns'].fillna(0)
data_original['returns_1'] = data_original['returns'].fillna(0)
data_original['returns_2'] = data_original['returns_1'].replace([np.inf, -np.inf], np.nan)
data_original['returns_final'] = data_original['returns_2'].fillna(0)
# Momentum setup for parameters over a rolling mean time window of 2 ( i.e average of past two day returns)
data_original['mom'] = np.sign(data_original['returns_final'].rolling(5).mean())





# 1. DEFINE NEURAL NETWORK ARCHITECTURE: Using Keras CNN (convolutional neural network) as a classifier.

# CNN chosen for its flexibility and interpretability of hyperparameters (convolutional kernal, downsampling size, etc.)
# Performance is similar to RNNs, and known to be overall better than MLP (multi layer perception) with faster training.


# Models in Keras can come in two forms – Sequential and via the Functional API.  For most deep learning networks that you build, the Sequential model is likely what you will use.
# t allows you to easily stack sequential layers (and even recurrent layers) of the network in order from input to output.  The functional API allows you to build more complicated architectures.
model = Sequential() # configures the model for training

# 1st layer
# 1D Convolution, i.e. temporal convolution. This layer creates a convolution kernel that is convolved with the layer input over a single spatial dimension to produce a tensor of outputs.
# Also notice that we don’t have to declare any weights or bias variables like we do in TensorFlow, Keras sorts that out for us.
model.add(Convolution1D(input_shape = (WINDOW, EMB_SIZE), # 3D tensor with shape: (samples, steps, input_dim). For us, windows = 30 is steps and EMB_SIZE = 5 is input_dim.
                        nb_filter=16, # nb_filter: Number of convolution kernels to use (dimensionality of the output)
                        filter_length=4, # the extension (spatial or temporal) of each filter.
                        border_mode='same')) # : 'valid', 'same', or 'full' ('full' requires the Theano backend).
model.add(BatchNormalization()) # normalize the activations of the previous layer at each batch, i.e. applies a transformation that maintains the mean activation close to 0 and the activation standard deviation close to 1
model.add(LeakyReLU()) # The rectifier is an activation function (what defines the output of a node given an input or set of inputs). Leaky version of a Rectified Linear Unit. Leaky ReLUs allow a small, non-zero gradient when the unit is not active.
#  Leaky ReLUs are one attempt to fix the "dying ReLU" problem. Instead of the function being zero when x < 0, a leaky ReLU will instead have a small negative slope (of 0.01, or so). That is, the function
# computes f(x)=𝟙(x<0)(αx)+𝟙(x>=0)(x) where α is a small constant. Some people report success with this form of activation function, but the results are not always consistent.

model.add(Dropout(0.5)) # fraction of neurons (inputs) to drop. goal for dropout is to help neural network not overfit the data. We randomly deactivate 50% of certain units (neurons) in each
# layer by setting some of the dimensions in our input vector to be zero with probability keep_prob.; thus the neural network will continue to learn different. Training thus will be faster.


# 2nd Layer, no input shape given (only needed to be defined in 1st layer above)
model.add(Convolution1D(nb_filter=8,
                        filter_length=4,
                        border_mode='same'))
model.add(BatchNormalization())
model.add(LeakyReLU())
model.add(Dropout(0.5))



model.add(Flatten()) # flattens the input (or rows of a 3D matrix). does not affect the batch size. Flatten() operator unrolls the values beginning at the last dimension.


#INPUT LAYER
model.add(Dense(64)) # Dense(64) is a fully-connected layer with 64 hidden units. A dense layer represents a matrix vector multiplication. (assuming your batch size is 1) The values in the matrix are the
# trainable parameters which get updated during backpropagation. A dense layer thus is used to change the dimensions of your vector. Mathematically speaking, it applies a rotation, scaling, translation
# transform to your vector. In simple terms, A dense layer is just a regular layer of neurons in a neural network. Each neuron recieves input from all the neurons in the previous layer, thus densely connected.
# The layer has a weight matrix W, a bias vector b, and the activations of previous layer a.
model.add(BatchNormalization())
model.add(LeakyReLU())


# OUTPUT LAYER
model.add(Dense(2)) # Dense(64) is a fully-connected layer with 2 hidden units.
model.add(Activation('softmax')) # apply softmax activation function; typically used for models with probabilistic approach, such as this.




# 2. COMPILE THE NEURAL NETWORK MODEL

opt = Nadam(lr=0.002)  # Define optimizer. use Nadam optimizer. https://keras.io/optimizers/. Stands for Nesterov Adam optimizer. Much like Adam is essentially RMSprop with momentum, Nadam is Adam RMSprop with Nesterov momentum.
# Adapting the learning rate for your stochastic gradient descent optimization procedure can increase performance and reduce training time.
# The simplest and perhaps most used adaptation of learning rate during training are techniques that reduce the learning rate over time. These have the benefit of making large changes at the beginning of the training procedure when larger learning rate values are used, and decreasing the learning rate such that a smaller rate and therefore smaller training updates are made to weights later in the training procedure.

#This has the effect of quickly learning good weights early and fine tuning them later.

#Two popular and easy to use learning rate schedules are as follows:

#A. Decrease the learning rate gradually based on the epoch.
#B. Decrease the learning rate using punctuated large drops at specific epochs.

# Arguments for optimizers:
# lr: learning rate, float >= 0.
# beta_1, beta_2: floats, 0 < beta < 1. Generally close to 1.
# epsilon: float > = 0. Fuzz factor. If None, defaults to K.epsilon().

reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.9, patience=30, min_lr=0.000001, verbose=1) # https://keras.io/callbacks/#reducelronplateau. Reduce learning rate  by a pre-defined factor when a metric has stopped improving.
# Models often benefit from reducing the learning rate by a factor of 2-10 once learning stagnates. This callback monitors a quantity and if no improvement is seen for a 'patience' number of epochs,
# the learning rate is reduced.

# arguments:
# monitor: quantity to be monitored
# factor: factor by which the learning rate will be reduced. new_lr = lr * factor
# patience: number of epochs with no improvement after which learning rate will be reduced.
# verbose: int. 0: quiet, 1: update messages.
# min_lr: lower bound on the learning rate.

checkpointer = ModelCheckpoint(filepath="lolkek.hdf5", verbose=1, save_best_only=True) # save the model after every epoch. create a custom callback by extending the base class keras.callbacks.Callback. A callback has access to its associated model through the class property self.model.
# Arguments:
# filepath: string, path to save the model file.
# monitor: quantity to monitor.
# verbose: verbosity mode, 0 or 1.
# save_best_only: if save_best_only=True, the latest best model according to the quantity monitored will not be overwritten.

model.compile(optimizer=opt, # optimization is whatever we set optimizer to
              loss='categorical_crossentropy', # https://keras.io/losses/ . Decide which loss function will be applied. The actual optimized objective is th mean of the output array across all datapoints.
              metrics=['accuracy']) # https://keras.io/metrics/ . Goal is to return a single tensor value representing the mean of the output array across all datapoints.

history = model.fit(X_train, Y_train, # fit the model using X_train, and Y_train data. https://keras.io/models/sequential/
          nb_epoch = 100, #  Integer. Number of epochs to train the model. An epoch is an iteration over the entire x and y data provided.
          batch_size = 128, # number of samples per gradient update. If unspecified, it will default to 32.
          verbose=1, # show updated messages when set to 1.  0 = silent, 1 = progress bar.
          validation_data=(X_test, Y_test), # tuple (x_val, y_val) or tuple (x_val, y_val, val_sample_weights) on which to evaluate the loss and any model metrics at the end of each epoch. The model will not be trained on this data. This will override validation_split.
          callbacks=[reduce_lr, checkpointer], # List of keras.callbacks.Callback instances. List of callbacks to apply during training
          shuffle=True) # shuffle: Boolean (whether to shuffle the training data before each epoch) or str (for 'batch'). 'batch' is a special option for dealing with the limitations of HDF5 data; it shuffles in batch-sized chunks. Has no effect when steps_per_epoch is not  None.




# 3. PREDICT MODEL OUTPUT FOR OUT OF SAMPLE Y_TEST

model.load_weights("lolkek.hdf5") # loads the weights of the model from a HDF5 file (created by  save_weights). By default, the architecture is expected to be unchanged. https://keras.io/models/about-keras-models/
pred = model.predict(np.array(X_test)) # predict output for out of sample Y_test, setting X_test (our out of sample features) to a numpy array





# USE CONFUSION MATRIX TO HELP CHECK FOR OVERFITTING:

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
C = confusion_matrix([np.argmax(y) for y in Y_test], [np.argmax(y) for y in pred])

print(C / C.astype(np.float).sum(axis=1))

# Classification
# [[ 0.75510204  0.24489796]
#  [ 0.46938776  0.53061224]] # forecasted up movements with 75%, and down movements with 53% , off diagonal elements are those mislabeled by the classifier






# PLOT MODEL LOSS & ACCURACY

plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='best')
plt.show()

plt.figure()
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='best')
plt.show()






#****************************************************** BACKTESTING PREPERATION & SETUP ***************************************************************************#


print("PREDICTION LENGTH")
print(len(pred))
print(pred)
pred = pd.DataFrame(pred)
predictions_dataframe = pd.DataFrame(pred)  # convert array to dataframe.

frames = []

df_close = data_original['Adj Close'][-590:].reset_index(drop=True)
df_SMA = data_original['SMA'][-590:].reset_index(drop=True)
df_mom = data_original['mom'][-590:].reset_index(drop=True)
df_MACD = data_original['stock_df_crossover_12_26'][-590:].reset_index(drop=True)
df_RSI = data_original['RSI'][-590:].reset_index(drop=True)
df_upper_band = data_original['Upper Band'][-590:].reset_index(drop=True)
df_lower_band = data_original['Lower Band'][-590:].reset_index(drop=True)
dema = data_original['dema'][-590:].reset_index(drop=True)


df_buy1 = df_close > df_SMA

df_buy2 = df_mom > 0

df_buy3 = df_RSI < 55 #55

df_buy4 = df_MACD > 0

df_buy5 = df_close > dema

#df_buy4 = df_close < df_lower_band

df_BUY = df_buy1 + df_buy2 + df_buy3 + df_buy4 + df_buy5
df_BUY_FINAL = df_BUY.reset_index(drop=True)


df_sell1 = df_close < df_SMA

df_sell2 = df_mom < 0

df_sell3 = df_RSI > 50 #50

df_sell4 = df_MACD < 0

df_sell5 = df_close < dema

#df_sell4 = df_close > df_upper_band

df_SELL = df_sell1 + df_sell2 + df_sell3 + df_sell4 + df_sell5
df_SELL_FINAL = df_SELL.reset_index(drop=True)



#print("LENGTH DF BUY FINAL")
#print(len(df_BUY_FINAL))

#print("LENGTH DF SELL FINAL")
#print(len(df_SELL_FINAL))


buy_technical_signal = df_BUY_FINAL # extract last n daily signals
sell_technical_signal = df_SELL_FINAL # extract last n daily signals
print("buy technical signal length", len(buy_technical_signal))
print("sell technical signal length", len(sell_technical_signal))
print("buy technical signal", buy_technical_signal)
print("sell technical signal", sell_technical_signal)
buy_technical_signal = pd.DataFrame(buy_technical_signal)
sell_technical_signal = pd.DataFrame(sell_technical_signal)


pred = pd.DataFrame(predictions_dataframe)
print("prediction", pred)
print("PREDICTION LENGTH", len(predictions_dataframe))

pred['Signal'] = 0

# reindex data for later concatenation into 1 DataFrame - will all need same index to concat properly
buy_technical_signal.set_index(pred.index, inplace=True)
sell_technical_signal.set_index(pred.index, inplace=True)




#set up filters
filter1 = (pred[0] > 0.57) # buy filter, 1st column [0] of pred dataframe (recall 2nd column [1] is simply % that stock will drop)
filter2 = (pred[0] < 0.41) # sell filter, 1st column [0] of pred dataframe (recall 2nd column [1] is simply % that stock will drop#)
filter3 = buy_technical_signal > 0  # second buy filter (technical one), > 0 implies argument is true (1)
filter4 = sell_technical_signal > 0  # second sell filter (technical one), > 0 implies argument is true (1)

#concatenate all data into 1 DataFrame for easy viewing/confirmationpip
pred2 = pd.concat([pred,filter1,filter2, filter3, filter4],axis=1)
print(pred2)
pred2.columns = ['Pred 1st Column','Pred 2nd Column','Signal','Filter1','Filter2', 'Filter3', 'Filter4']


#use np.where function to set Signal according to whether above filters are satisfied
pred2['Signal'] = np.where(pred2['Filter1'] & pred2['Filter3'],1,0)
pred2['Signal'] = np.where(pred2['Filter2'] & pred2['Filter4'],-1,pred2['Signal'])



buys = pred2.loc[pred2['Signal'] == 1]
sells = pred2.loc[pred2['Signal'] == -1]


# need to reindex the buys and sells DataFrames to match the index of 'data_original[Close]'
if not buys.empty:
    buy_index_new = buys.index[-1] - buys.index
    buy_index_new_2 = len(data_original.index) - buy_index_new
    buys.set_index(buy_index_new_2,inplace=True)

if not sells.empty:
    sell_index_new = sells.index[-1] - sells.index
    sell_index_new_2 = len(data_original.index) - sell_index_new
    sells.set_index(sell_index_new_2,inplace=True)




# iloc[row slicing, column slicing] Real Stock Price set to last 788 days of stock's close price
real_stock_price = data_original.iloc[-590:,4] # length of predictions_dataframe is the last 1,607 days when training at 80%, testing 20% of total data . 4th column is close prices
real_stock_price = real_stock_price.values
real_stock_price = pd.DataFrame(real_stock_price) # convert to pandas data frame
print(real_stock_price)
print("REAL STOCK PRICE",real_stock_price)

#print(data_original.iloc[-98:,4])


# Visualize strategy with a chart

# the buys and sells have integer range indices (i.e. 1,2,3....) whereas the df index has datetime values. Reindex df using:
data_original.set_index(pd.RangeIndex(0,len(data_original)),inplace=True)

# plot price
plt.plot(data_original.index,data_original['Close'], label='BTC/USD')

# Plot the buy and sell signals on the same plot
plt.plot(sells.index, data_original.loc[sells.index]['Close'], 'v', markersize=10, color='r')
plt.plot(buys.index, data_original.loc[buys.index]['Close'], '^', markersize=10, color='g')
plt.ylabel('Price')
plt.xlabel('Date')
plt.legend(loc=0)
# Display everything
plt.show()


#print("PRED 2")
#print(pred2)



# BACKTESTING *********************************************************************************************************


# define backtest method
def backtest(data):
    cash = 100000 # set starting cash to $100,000
    position = 0 # set position to 0 for current number of shares
    total = 0
    equity_curve_df = []

    data['Total'] = 100000 # start with 100k for our strategy
    # To compute the Buy and Hold value, I invest all of my cash in X asset on the first day of the backtest
    increment = 100 # number of shares

    for row in data.iterrows():
        price = float(row[1][0]) # Remember that "iterrows" returns an indexer(i.e. 0,1,2,3,4....)  and the row of the DataFrame in a row vector - so you need to also reference the column you want in the row vector, hence the [1][3] - the 1 being the row (rather than the indexer), and the column within that row.
        signal = pred2.iloc[row[0]][2]  # signal for our strategy, 3rd column in the pred2 dataframe is signals of 1 and -1


        if (signal > 0 and cash - increment * price > 0): # ensure signal is 1 (or > 0), and there is enough cash to place trade (100,000 - (1,000 * price of asset) > 0 )
            # Buy
            cash = cash - increment * price # deduct how many shares we bought and update cash remaining value
            position = position + increment # position is 0 + 1,000 shares (this keeps on going and looping as long as cash is available for another buy, assuming signal is there)
            #print(row[0].strftime('%d %b %Y') + " Position = " + str(position) + " Cash = " + str(cash) + " // Total = {:,}".format(int(position * price + cash)))

        elif (signal < 0 and abs(position * price) < cash): # ensure signal is -1 (or < 0), and absolute value of (position or shares sold * price of stock is less than cash value to allow trade)
            # Sell
            cash = cash + increment * price # add cash value of portfolio to how many shares we sold, and update cash remaining value
            position = position - increment # position is new position (number of shares) - increment (or how many shares were sold)
            #print(row[0].strftime('%d %b %Y') + " Position = " + str(position) + " Cash = " + str(cash) + " // Total = {:,}".format(int(position * price + cash)))

        equity_curve_df.append(float(position * price + cash))

    # equity_curve_df = pd.DataFrame(equity_curve_df,index=range(len(equity_curve_df)),columns=["Total"])
    index = pd.date_range('01/01/2017', periods=len(equity_curve_df), freq='D')
    equity_curve_df = pd.DataFrame(equity_curve_df, index=index, columns=['Total'])
    return equity_curve_df  # return number of shares multiplied by price of asset + cash left in balance


# Now, implement backtest method defined above:

# Backtest for our strategy to create equity curve df by running backtest function defined above
equity_curve_df  = backtest(real_stock_price) # for our strategy, backtest will be equal to the backtest(data) method we define above, utilizing our dataframe as the data
print(equity_curve_df) # prints out cash value of backtest result in USD




# *******************************************************************************************************************************************************************************************************************************

# Apply Financial Functions for Python (FFN) to calculate statistics of algorithm
equity_curve_df['Equity'] = equity_curve_df
perf = equity_curve_df['Equity'].calc_stats()


# plot equity curve
perf.plot()
plt.show()

# show overall metrics
perf.display()


# display monthly returns
perf.display_monthly_returns()


# plotting visual representation of strategy drawdown series:
ffn.to_drawdown_series(equity_curve_df['Equity']).plot(figsize=(15,7),grid=True)
plt.show()


# plot histogram of returns
perf.plot_histogram()
plt.show()


# extract lookback returns
perf.display_lookback_returns()


# *************************************************************************************************************************************************************************************************
# PRINT COLUMN OF SIGNALS FOR THE STRATEGY
print("CURRENT SIGNAL FOR ALGORITHM: 1: BUY, 0: HOLD, -1:SELL")
print(pred2.iloc[:,2])
# **************************************************************************************************************************************************************************************************
