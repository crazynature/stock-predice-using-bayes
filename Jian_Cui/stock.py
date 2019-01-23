import nltk
import random
import pandas
import numpy as np
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
class Stock:
    def __init__(self):
        self.label = []

    def labelPlot(self,label,index):

        x = []
        y = []
        for i in range(50):
            x.append(i)
            if label[i] == self.label[i]:
                y.append(1)
            else:
                y.append(0)
        # fig = plt.figure()
        plt.scatter(x, y, alpha=0.6)
        plt.savefig('label50 '+str(index)+'.png')
        plt.gcf().clear()
        x = []
        y = []
        for i in range(100):
            x.append(i)
            if label[i] == self.label[i]:
                y.append(1)
            else:
                y.append(0)
        # fig = plt.figure()
        plt.scatter(x, y, alpha=0.6)
        plt.savefig('label100 '+str(index)+'.png')
        plt.gcf().clear()
        x = []
        y = []
        for i in range(len(label)):
            x.append(i)
            if label[i] == self.label[i]:
                y.append(1)
            else:
                y.append(0)
        # fig = plt.figure()
        plt.scatter(x, y, alpha=0.6)
        plt.savefig('labelall '+str(index)+'.png')
        plt.gcf().clear()
        # plt.show()

    def profitPlot(self,profit,index):
        f = open('profit'+str(index)+'.txt', 'w')

        x = []
        for i in range(len(profit)):
            x.append(i)
            f.writelines(str(profit[i]))
            f.write('\n')
        f.close()
        # fig = plt.figure()
        plt.plot(x, profit)
        plt.savefig('profit '+str(index)+'.png')
        plt.gcf().clear()

    def stock_load(self,file_name):
        """
        """
        stock_list = []
        file = open(file_name)

        is_first_line = True
        for line in file:
            if is_first_line:
                is_first_line = False
                continue
            prop_list = line.split(',')

            open_price = float(prop_list[1])
            high_price = float(prop_list[2])
            low_price = float(prop_list[3])
            close_price = float(prop_list[4])
            price_change = float(open_price - close_price)
            p_change = float(price_change/open_price)
            stock_list.append([open_price, high_price, low_price, close_price,price_change, p_change])
        file.close()
        return stock_list
    def nGram_data(self,stock_data,n=3,featureAmount = 6,featureStart =0):
        # openprice,highprice,lowprice,closeprice,pricediffer,pricediff_slope
        data=[]
        self.label = []

        for i in range(0,len(stock_data)-n):
            temp = []
            open_diff = 0
            high_diff = 0
            low_diff = 0
            close_diff = 0
            price_diff = 0
            p_diff = 0
            open = stock_data[i][0]
            close = stock_data[i+n][2]
            for j in range(1,n):
                open_diff = open_diff+ stock_data[i+j][0]- stock_data[i+j-1][0]
                high_diff = open_diff + stock_data[i + j][1] - stock_data[i + j - 1][1]
                low_diff = open_diff + stock_data[i + j][2] - stock_data[i + j - 1][2]
                close_diff = open_diff + stock_data[i + j][3] - stock_data[i + j - 1][3]
                price_diff = open_diff + stock_data[i + j][4] - stock_data[i + j - 1][4]
                p_diff = open_diff + stock_data[i + j][5] - stock_data[i + j - 1][5]

            temp.append(open_diff)

            temp.append(high_diff)

            temp.append(low_diff)

            temp.append(close_diff)

            temp.append(price_diff)

            temp.append(p_diff)

            temp = temp[featureStart:featureAmount]
            if i == 1:
                print(temp)
            data.append(temp)
            change = close - open
            if change > 0:
                self.label.append(1)
            else:
                self.label.append(0)

        return data

    def newFeature_nGram_data(self,stock_data,n=2,featureStart = 0,featureEnd = 6):
        # openprice,highprice,lowprice,closeprice,pricediffer,pricediff_slope
        data=[]
        self.label = []

        for i in range(0,len(stock_data)-n):
            temp = []
            open_diff = 0
            high_diff = 0
            low_diff = 0
            close_diff = 0
            price_diff = 0
            p_diff = 0
            open = stock_data[i][0]
            close = stock_data[i+n][2]
            for j in range(1,n-1):
                open_diff = open_diff+ stock_data[i+j][0]- stock_data[i+j-1][0]
                high_diff = open_diff + stock_data[i + j][1] - stock_data[i + j-1 ][1]
                low_diff = open_diff + stock_data[i + j][2] - stock_data[i + j-1][2]
                close_diff = open_diff + stock_data[i + j][3] - stock_data[i + j-1 ][3]
                price_diff = open_diff + stock_data[i + j][4] - stock_data[i + j-1][4]
                p_diff = open_diff + stock_data[i + j][5] - stock_data[i + j-1][5]
            open_slope = (stock_data[i+n-1][0]-stock_data[i][0])/stock_data[i][0]
            high_slope = (stock_data[i+n-1][1]-stock_data[i][1])/stock_data[i][1]
            low_slope = (stock_data[i+n-1][2]-stock_data[i][2])/stock_data[i][2]
            close_slope = (stock_data[i+n-1][3]-stock_data[i][3])/stock_data[i][3]
            price_slope = (stock_data[i+n-1][3]-stock_data[i][0])/stock_data[i][4]
            temp.append(open_diff)
            temp.append(high_diff)
            temp.append(low_diff)
            temp.append(close_diff)
            temp.append(price_diff)
            temp.append(p_diff)
            temp.append(open_slope)
            temp.append(high_slope)
            temp.append(low_slope)
            temp.append(close_slope)
            temp.append(price_slope)
            temp = temp[featureStart:featureEnd]

            data.append(temp)
            change = close - open
            if change > 0:
                self.label.append(1)
            else:
                self.label.append(0)

        return data

    def single_feature_train_model(self,index, featureStart=0,):
        """
        # """
        stock_data = self.stock_load('GOOG.csv')
        nGram_data = self.nGram_data(stock_data, featureAmount=featureStart+1,featureStart=featureStart)
        print(len(stock_data))
        print(len(nGram_data))
        print(len(self.label))
        # print(self.label)
        X = np.array(nGram_data)
        Y = np.array(self.label)
        clf = GaussianNB()
        clf.fit(X, Y)

        print("==Predict result by predict==" + str(featureStart))
        predictedData = clf.predict(nGram_data)
        # print((predictedData))
        print("==Predict accuracy==" + str(featureStart))
        accuracy = self.calculateAccuracy(predictedData)
        print(accuracy)
        print("==Profit==" + str(featureStart))
        profit = self.tradePolicy(predictedData,stock_data,index)
        print(profit)
        # print("==Predict result by predict_proba==")
        # print(clf.predict_proba(nGram_data))
        # print("==Predict result by predict_log_proba==")
        # print(clf.predict_log_proba(nGram_data))
        test_stock_data = self.stock_load('GOOG_test.csv')
        test_nGram_data = self.nGram_data(test_stock_data,featureAmount=featureStart+1,featureStart=featureStart,)
        print(len(test_stock_data))
        print(len(test_nGram_data))
        print(len(self.label))
        print("==Predict test result by predict==" + str(featureStart))
        test_predictedData = clf.predict(test_nGram_data)
        # print((test_predictedData))
        self.labelPlot(test_predictedData, index)
        print("==Predict test accuracy==" + str(featureStart))
        test_accuracy = self.calculateAccuracy(test_predictedData, )
        print(test_accuracy)
        print("==test Profit==" + str(featureStart))
        test_profit = self.tradePolicy(test_predictedData,test_stock_data,index)
        self.profitPlot(self.profitGraph, index)
        print(test_profit)

        f = open('result.txt', 'a')

        f.writelines("single "+str(featureStart) + " " + "sample " + str(profit)+' ' + str(accuracy) + ' test ' + str(
            test_profit) + ' ' + str(test_accuracy) + '\n')
        f.close()
        return  self.profitGraph

    def train_model(self,index,featureAmount=6):
            """
            # """
            stock_data = self.stock_load('GOOG.csv')
            nGram_data = self.nGram_data(stock_data,featureAmount=featureAmount)
            print(len(stock_data))
            print(len(nGram_data))
            print(len(self.label))
            # print(self.label)
            X = np.array(nGram_data)
            Y = np.array(self.label)
            clf = GaussianNB()
            clf.fit(X, Y)

            print("==Predict result by predict=="+str(featureAmount))
            predictedData = clf.predict(nGram_data)
            # print((predictedData))
            print("==Predict accuracy=="+str(featureAmount))
            accuracy = self.calculateAccuracy(predictedData)
            print(accuracy)
            print("==Profit=="+str(featureAmount))
            profit = self.tradePolicy(predictedData,stock_data,index)
            print(profit)
            # print("==Predict result by predict_proba==")
            # print(clf.predict_proba(nGram_data))
            # print("==Predict result by predict_log_proba==")
            # print(clf.predict_log_proba(nGram_data))
            test_stock_data =self.stock_load('GOOG_test.csv')
            test_nGram_data = self.nGram_data(test_stock_data,featureAmount=featureAmount)
            print(len(test_stock_data))
            print(len(test_nGram_data))
            print(len(self.label))
            print("==Predict test result by predict=="+str(featureAmount))
            test_predictedData = clf.predict(test_nGram_data)
            self.labelPlot(test_predictedData,index)
            # print((test_predictedData))
            print("==Predict test accuracy=="+str(featureAmount))
            test_accuracy = self.calculateAccuracy(test_predictedData,)
            print(test_accuracy)
            print("==test Profit=="+str(featureAmount))
            test_profit = self.tradePolicy(test_predictedData, test_stock_data,index)
            self.profitPlot(self.profitGraph, index)
            print(test_profit)

            f = open('result.txt', 'a')
            f.writelines(str(featureAmount)+" "+"sample "+ str(profit)+' '+str(accuracy)+' test '+str(test_profit)+' '+str(test_accuracy)+'\n')
            f.close()
            return self.profitGraph
    # def newFeature_train_model(self):
    #         """
    #         # """
    #         stock_data = self.stock_load('GOOG.csv')
    #         nGram_data = self.newFeature_nGram_data(stock_data)
    #         print(len(stock_data))
    #         print(len(nGram_data))
    #         print(len(self.label))
    #         # print(self.label)
    #         X = np.array(nGram_data)
    #         Y = np.array(self.label)
    #         clf = GaussianNB()
    #         clf.fit(X, Y)
    #
    #         print("==Predict result by predict==")
    #         predictedData = clf.predict(nGram_data)
    #         # print((predictedData))
    #         print("==Predict accuracy==")
    #         accuracy = self.calculateAccuracy(predictedData)
    #         print(accuracy)
    #         print("==Profit==")
    #         profit = self.tradePolicy(predictedData,stock_data)
    #         print(profit)
    #         # print("==Predict result by predict_proba==")
    #         # print(clf.predict_proba(nGram_data))
    #         # print("==Predict result by predict_log_proba==")
    #         # print(clf.predict_log_proba(nGram_data))
    #         test_stock_data =self.stock_load('GOOG_test.csv')
    #         test_nGram_data = self.newFeature_nGram_data(test_stock_data)
    #         print(len(test_stock_data))
    #         print(len(test_nGram_data))
    #         print(len(self.label))
    #         print("==Predict test result by predict==")
    #         test_predictedData = clf.predict(test_nGram_data)
    #         # print((test_predictedData))
    #         print("==Predict test accuracy==")
    #         test_accuracy = self.calculateAccuracy(test_predictedData)
    #         print(test_accuracy)
    #         print("==test Profit==")
    #         test_profit = self.tradePolicy(test_predictedData, test_stock_data)
    #         print(test_profit)


    def calculateAccuracy(self,predictedData,n=3):
        count =0
        # print(predictedData[0])
        # print(self.label[0])
        for i in range(len(predictedData)):
            if self.label[i] == predictedData[i]:
                count = count + 1
        result = count/len(predictedData)

        # print(result)

        return result

    def tradePolicy(self,predictedData,stock_data,index,n=3):
        # openprice,highprice,lowprice,closeprice,pricediffer,pricediff_slope
        currentStock = 0
        self.profitGraph =[]
        profit = 0
        lastBuy = 0
        action = []
        print(stock_data[3][0])
        print(stock_data[len(stock_data)-1][3])
        for i in range(len(predictedData)):
            # print(stock_data[i][0])
            if predictedData[i] == 1:
                currentStock += 1
                lastBuy = lastBuy + stock_data[i+n][0]
                action.append(('buy',str(stock_data[i+n][0])))

            else:
                if currentStock > 0:
                    profit = profit + (stock_data[i+n-1][1]*currentStock) - lastBuy
                    self.profitGraph.append(profit)
                    currentStock = 0
                    lastBuy = 0
                    action.append(('sell '+str(stock_data[i+n-1][3])+' | profit ', str(profit)))
                    # action.append(('sell '+' | profit ', str(profit)))
        # if currentStock>0:
        #     profit = profit + (stock_data[i + n][1] * currentStock) - lastBuy
        #     currentStock = 0
        #     lastBuy = 0
        #     action.append(('sell ' + str(stock_data[i + n][1]) + ' | profit ', str(profit)))
        f = open('action'+str(index)+'.txt', 'w')
        for a in action:
            f.writelines(a)
            f.write('\n')
        f.close()
        return profit




if __name__ == '__main__':
    test = Stock()
    profit = []
    f= open('result.txt','w')
    f.write("")
    f.close()
    for i in range(0,6):
        profit.append(test.single_feature_train_model(i,i))
    for i in range(1,7):
        profit.append(test.train_model(5+i,i))

    ax = plt.subplot()
    name = ["open","close","higest","lowest","diff","slope","1 feature","2 features","3 features","4 features","5 features","6 features"]
    for i in range(len(profit)):
        x = []
        for j in range(len(profit[i])):
            x.append(j)
        ax.plot(x,profit[i],label="$'+name[i]+'$")
    plt.savefig('profitAll ' + '.png')
    plt.gcf().clear()