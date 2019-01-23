from agent import Agent
from functions import *

try:
    stock_name = "train"
    window_size = 2
    episode_count = 150

    agent = Agent(window_size)
    data = getStockDataVec(stock_name)

    # data = load_data()
    l = len(data) - 1
    batch_size = 32

    reward_list = []

    # one episode
    for e in range(episode_count + 1):
        # print ("Episode " + str(e) + "/" + str(episode_count))

        state = getState(data, 0, window_size + 1)

        # set inventory and profit
        agent.inventory = []
        total_profit = 0

        total_reward = 0

        for t in range(l):
            reward = 0
            action = agent.act(state)

            # sit
            next_state = getState(data, t + 1, window_size + 1)

            if action == 1:  # buy
                agent.inventory.append(data[t])
            elif action == 2 and len(agent.inventory) > 0:  # sell
                temp_profit = 0
                while len(agent.inventory) != 0:
                    bought_price = agent.inventory.pop(0)
                    temp_profit += data[t] - bought_price
                    # total_profit += data[t] - bought_price

                total_profit += temp_profit

                reward = temp_profit
                # if temp_profit > 0:
                #     reward = 1
                #
                # if temp_profit < 0:
                #     reward = -1

            done = True if t == l - 1 else False

            agent.memory.append((state, action, reward, next_state, done))

            # update new state
            state = next_state

            if len(agent.memory) > batch_size:
                agent.expReplay(batch_size)

            total_reward += reward

        # if e % 5 == 0:
        reward_list.append(total_reward)

        print ("total reward " + str(total_reward) + " running " + str(e) + "/" + str(episode_count))

    agent.model.save("models/test" + str(episode_count))
    print reward_list

except Exception as e:
    print("Error occured: {0}".format(e))
finally:
    exit()
