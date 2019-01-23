# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

dataset_train = pd.read_csv("/Users/benstruhl/Documents/stockpredictor/train.csv")
training_set = dataset_train.iloc[:, 4:5].values
TIMESTEP = 7

sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

X_train = []
y_train = []
for i in range(TIMESTEP, 446):
    X_train.append(training_set_scaled[i-TIMESTEP:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)


X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

regressor = Sequential()

regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

regressor.add(Dense(units = 1))

regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

regressor.fit(X_train, y_train, epochs = 150, batch_size = 32)

dataset_test = pd.read_csv('/Users/benstruhl/Documents/stockpredictor/test.csv')
real_stock_price =  sc.fit_transform(dataset_test.iloc[:, 4:5].values)
real_stock_price = sc.inverse_transform(real_stock_price)


dataset_total = pd.concat((dataset_train['Close'], dataset_test['Close']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - TIMESTEP:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(TIMESTEP, 470):
    X_test.append(inputs[i-TIMESTEP:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

shares = []
max_ob = []
profit = 0.0
for i in range(0, len(real_stock_price)):
    if i > TIMESTEP:
    #    print(sum(shares))
        max_ob.append(sum(shares))  
        if predicted_stock_price[i-TIMESTEP][0] > real_stock_price[i][0]:
            shares.append(real_stock_price[i][0])
           # print("Buy " + " Real Price Today " + str(real_stock_price[i][0]) )
            #+ " Predicted Price tomorrow " + str(predicted_stock_price[i-TIMESTEP][0]))
        elif predicted_stock_price[i-TIMESTEP][0] < real_stock_price[i][0] and len(shares) > 0:
            num_shares = len(shares)
            score = (real_stock_price[i][0] * num_shares) -  sum(shares)  
            profit += ((real_stock_price[i][0] * num_shares) -  sum(shares))  
            shares = []
            #print("Sell Profit Gained: " + str(score) + " | Real Price Today " + str(real_stock_price[i][0])) 
            #+ " Predicted Price tomorrow " + str(predicted_stock_price[i+1][0]))
        else:
            continue
           # print("Hold Real Price Today " + str(real_stock_price[i][0])) 
            #+ " Predicted Price tomorrow " + str(predicted_stock_price[i-TIMESTEP][0]))

diff_at_each_step = np.zeros((len(real_stock_price) - TIMESTEP,1)) 
print(diff_at_each_step)
for i in range(TIMESTEP, len(real_stock_price)):
    diff_at_each_step[i-TIMESTEP][0] = predicted_stock_price[i-TIMESTEP] - real_stock_price[i][0]

#print(diff_at_each_step)
mean = np.sum(diff_at_each_step) / len(diff_at_each_step)

square_minus_mean_diff_at_each_step = np.zeros((len(diff_at_each_step),1)) 
for i in range(0, len(diff_at_each_step)):
    square_minus_mean_diff_at_each_step[i][0] = ((diff_at_each_step[i][0] - mean) ** 2)

square_minus_mean = np.sum(square_minus_mean_diff_at_each_step)  / len(square_minus_mean_diff_at_each_step)

for i in range(0, len(diff_at_each_step)):
    diff_at_each_step[i][0] = (diff_at_each_step[i][0] ** 2)

net_good_trades = [[0]]
t = 0
for i in range(0, len(real_stock_price)-1):
    if i > TIMESTEP:
        if predicted_stock_price[i-TIMESTEP][0] > real_stock_price[i][0]:
            if predicted_stock_price[i-TIMESTEP][0] > real_stock_price[i+1][0]: 
                net_good_trades.append([net_good_trades[t][0] + 1])
            else:
                net_good_trades.append([net_good_trades[t][0] + 1])
        elif predicted_stock_price[i-TIMESTEP][0] < real_stock_price[i][0] and len(shares) > 0:
            if predicted_stock_price[i-TIMESTEP][0] < real_stock_price[i+1][0]: 
                net_good_trades.append([net_good_trades[t][0] + 1])
            else:
                net_good_trades.append([net_good_trades[t][0] + 1])
        else:
            net_good_trades.append([net_good_trades[t][0]])
        t += 1

print("Standard dev: " + str(math.sqrt(square_minus_mean)))

print(len(predicted_stock_price))
print(len(real_stock_price))
print("Total Proft: " + str(profit))
print(max(max_ob))
real_stock_price = real_stock_price[TIMESTEP:len(real_stock_price)]

plt.plot(real_stock_price, color = 'black', label = 'GOOGL Stock Price')
plt.plot(predicted_stock_price, color = 'red', label = 'Predicted GOOGL Stock Price')
plt.title('GOOGL Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('GOOGL Stock Price')
plt.legend()
plt.show()

plt.plot(net_good_trades, color = 'red', label = 'Net good trades')
plt.title('Net good trades overtime')
plt.xlabel('Time')
plt.ylabel('Net good trades')
plt.legend()
plt.show()


plt.plot(diff_at_each_step, color = 'red', label = 'Difference')
plt.title('Squared Differences of predicted prices to actual prices at each time step')
plt.xlabel('Time')
plt.ylabel('Difference between prices')
plt.legend()
plt.show()