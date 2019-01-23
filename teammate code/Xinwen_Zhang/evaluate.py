import matplotlib.pyplot as plt
from keras.models import load_model

from agent import Agent
from functions import *

try:
    stock_name = "test"
    model_name = "netwl_as_reward_ep150"

    model = load_model("models/" + model_name)
    window_size = model.layers[0].input.shape.as_list()[1]

    # evaluation agent
    agent = Agent(window_size, True, model_name)
    data = getStockDataVec(stock_name)

    l = len(data) - 1

    state = getState(data, 0, window_size + 1)

    agent.inventory = []

    total_profit = 0
    total_reward = 0

    total_reward_change= []

    total_losing = 0

    total_winning =0

    for t in range(l):
        action = agent.act(state)

        # sit
        next_state = getState(data, t + 1, window_size + 1)
        reward = 0

        if action == 1:  # buy
            agent.inventory.append(data[t])
            print ("Buy: " + formatPrice(data[t]))

        elif action == 2 and len(agent.inventory) > 0:  # sell
            temp_profit = 0
            while len(agent.inventory) != 0:
                bought_price = agent.inventory.pop(0)
                temp_profit += data[t] - bought_price
                # total_profit += data[t] - bought_price

            # profit as reward
            # reward = temp_profit

            if temp_profit > 0:
                reward = 1
                total_winning +=1
            if temp_profit < 0:
                reward = -1
                total_losing+=1

            total_profit += temp_profit

            total_reward_change.append(total_profit)

            print ("Sell: " + formatPrice(data[t]) + " | Profit: " + formatPrice(temp_profit))

        total_reward_change.append(total_profit)

        done = True if t == l - 1 else False
        state = next_state

        total_reward += reward
        if done:
            print ("--------------------------------")
            print (stock_name + " Total Profit: " + formatPrice(total_profit) +
                   " total reward: "+str(total_reward)+
                   " total winning: "+str(total_winning)+
                   " total losing: "+str(total_losing))
            print ("--------------------------------")

            print (total_reward_change)



except Exception as e:
    print("Error is: " + e)
finally:
    exit()
