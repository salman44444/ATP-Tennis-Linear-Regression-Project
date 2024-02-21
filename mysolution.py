# %%
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# %%
players=pd.read_csv('tennis_stats.csv')
players.head()

# %%

# exploratory analysis
plt.scatter(players['FirstServeReturnPointsWon'],players['Winnings'])
plt.title('FirstServeReturnPointsWon vs Winnings')
plt.xlabel('FirstServeReturnPointsWon')
plt.ylabel('Winnings')
plt.show()
plt.clf()

plt.scatter(players['BreakPointsOpportunities'],players['Winnings'])
plt.title('BreakPointsOpportunities vs Winnings')
plt.xlabel('BreakPointsOpportunities')
plt.ylabel('Winnings')
plt.show()
plt.clf()

plt.scatter(players['BreakPointsSaved'],players['Winnings'])
plt.title('BreakPointsSaved vs Winnings')
plt.xlabel('BreakPointsSaved')
plt.ylabel('Winnings')
plt.show()
plt.clf()

plt.scatter(players['TotalPointsWon'],players['Ranking'])
plt.title('TotalPointsWon vs Ranking')
plt.xlabel('TotalPointsWon')
plt.ylabel('Ranking')
plt.show()
plt.clf()

plt.scatter(players['TotalServicePointsWon'],players['Wins'])
plt.title('TotalServicePointsWon vs Wins')
plt.xlabel('TotalServicePointsWon')
plt.ylabel('Wins')
plt.show()
plt.clf()


# %%
list = data[]

players[[list[0]]].dtypes
players.info()

# %%
#Printing each feature of players
for feature_name in players.columns[:-1]:  
    feature = players[[feature_name]]  
    if (feature.dtypes=='object').any():
        continue
    winnings=players.Winnings
    features_train,features_test,winnings_train,winnings_test=train_test_split(feature,winnings,train_size=0.8)
    features_train_reshaped=features_train.values.reshape(-1,1)
    model=LinearRegression()
    model.fit(features_train_reshaped,winnings_train)
    print(f'Predicting Winnings with {feature_name} Test Score:', model.score(features_test.values.reshape(-1,1),winnings_test))

    # Predicting Winnings with Year Test Score: -0.003121636894414026
    # Predicting Winnings with FirstServe Test Score: -0.003116710746064122
    # Predicting Winnings with FirstServePointsWon Test Score: 0.08912037107660054
    # Predicting Winnings with FirstServeReturnPointsWon Test Score: 0.05750797669517238
    # Predicting Winnings with SecondServePointsWon Test Score: 0.080793086260735
    # Predicting Winnings with SecondServeReturnPointsWon Test Score: 0.05198611949383869
    # Predicting Winnings with Aces Test Score: 0.555504627582022
    # Predicting Winnings with BreakPointsConverted Test Score: 0.011057886232584746
    # Predicting Winnings with BreakPointsFaced Test Score: 0.7773388679291842
    # Predicting Winnings with BreakPointsOpportunities Test Score: 0.7454820659187528
    # Predicting Winnings with BreakPointsSaved Test Score: 0.06932496451561254
    # Predicting Winnings with DoubleFaults Test Score: 0.7549353721177651
    # Predicting Winnings with ReturnGamesPlayed Test Score: 0.8271811764260122
    # Predicting Winnings with ReturnGamesWon Test Score: 0.04227074843513634
    # Predicting Winnings with ReturnPointsWon Test Score: 0.054450298112083906
    # Predicting Winnings with ServiceGamesPlayed Test Score: 0.8409758529387522
    # Predicting Winnings with ServiceGamesWon Test Score: 0.1512540802972442
    # Predicting Winnings with TotalPointsWon Test Score: 0.23515590598242553
    # Predicting Winnings with TotalServicePointsWon Test Score: 0.1716025379543692
    # Predicting Winnings with Wins Test Score: 0.8236402911732517
    # Predicting Winnings with Losses Test Score: 0.7400101604666354
    # Predicting Winnings with Winnings Test Score: 1.0
    
# %%
# Features with high scores used in this multiple Linear Regession
feature = players[[
'Aces',
'BreakPointsFaced','BreakPointsOpportunities',
'DoubleFaults','ReturnGamesPlayed','ReturnGamesWon',
'ServiceGamesPlayed','Wins','Losses']]
winnings=players.Winnings
features_train,features_test,winnings_train,winnings_test=train_test_split(feature,winnings,train_size=0.8)
#features_train_reshaped=features_train.values.reshape(-1,1)
model=LinearRegression()
model.fit(features_train,winnings_train)
print(f'Predicting Winnings with {feature_name} Test Score:', model.score(features_test,winnings_test))

#Predicting Winnings with Winnings Test Score: 0.8749662942756673

# %%
winning_predict=model.predict(features_test)

# %%
plt.scatter(winnings_test,winning_predict, alpha=0.4)
plt.title('Predicted Winnings vs. Actual Winnings - 1 Feature')
plt.xlabel('Actual Winnings')
plt.ylabel('Predicted Winnings')
plt.show()
plt.clf()
