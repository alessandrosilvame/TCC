# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 16:39:31 2020

@author: amanoels
"""

import pandas as pd
import seaborn as sns
sns.set(color_codes=True)
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

data = pd.read_csv('brainhead.csv', sep=';', header=0)

data.describe()

data.info()

data.head()

data.isnull().sum()

data.isna().sum()

sns.scatterplot(x=data['head_size'], y=data['brain_weight'])

correlacao = data.corr()

correlacao

ax = sns.heatmap(correlacao, annot=True)
ax.set_yticklabels(ax.get_yticklabels(),rotation=0)
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
ax.set_title('Correlation Head Size & Brain Weight')


sns.boxplot(x=data['head_size'], orient="v")

sns.boxplot(x=data['brain_weight'], orient="v")

data.info()

data[(data['brain_weight'] > 1570)]
df = data.loc[data['brain_weight'] <= 1565]
brainhead = df[(df['brain_weight'] >= 990)]

brainhead.describe()

sns.boxplot(x=brainhead['head_size'], orient="v")

sns.boxplot(x=brainhead['brain_weight'], orient="v")

sns.distplot(brainhead['brain_weight'])


reg = sm.ols(formula='brain_weight~head_size', data=brainhead).fit()

print(reg.summary())

bx = sns.regplot(x=brainhead['head_size'], y=brainhead['brain_weight'], data=brainhead)
bx.set_title('Regresão Linear')

cx = sns.residplot('head_size', 'brain_weight', data=brainhead)
cx.set_title('Residuos')

y_hat = reg.predict()

res = brainhead['brain_weight'] - y_hat

sns.distplot(res)


# Para comparação com 
coefs = pd.DataFrame(reg.params)
coefs.columns = ['Coeficientes']
coefs

x = brainhead['head_size'].values.reshape(-1,1)
y = brainhead['brain_weight'].values.reshape(-1,1)

# Random_State 12 é melhor por enquanto
x_train, x_test, y_train, y_test = train_test_split(x,y,
    test_size=0.2, random_state=12)

regressor = LinearRegression()
regressor.fit(x_train, y_train)

regressor.intercept_
regressor.coef_
coefs

y_pred = regressor.predict(x_test)

df = pd.DataFrame({'Brain Weight atual': y_test.flatten(), 
                   'Brain Weight Preditos': y_pred.flatten()})
dfhead = df.head(21)
dfhead

dfhead.plot(kind='bar',figsize=(16,10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()

plt.scatter(x_test, y_test,  color='blue')
plt.plot(x_test, y_pred, color='red', linewidth=2)
plt.show()

print("Acuracia na base de teste: {:.2f}".format(regressor.score(x_test, y_test)))
print("Acuracia na base de treino: {:.2f}".format(regressor.score(x_train, y_train)))

# Proximo de 1 é bom
print(f"R2 Score: {r2_score(y_test, y_pred)}")

# Proximo de 0 é bom
print(f"MSE Score: {mean_squared_error(y_test, y_pred)}")
















