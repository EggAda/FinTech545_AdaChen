from missing_data import *
from EWMA import *
from nearpsd_higham import *
from chol import *
from simulation import *
from returns_fit_var_es import *

portfolio = pd.read_csv('Portfolio_thisweek.csv')
df_32 = pd.read_csv('DailyPrices_thisweek.csv')
dailyreturn = return_calculate(df_32, "DISCRETE", "Date")
# dailyreturn -= dailyreturn.mean(numeric_only=True)
returns = dailyreturn.copy()
returns = returns.drop('Date', axis=1)
print(returns)
for stock in portfolio["Stock"]:
    portfolio.loc[portfolio['Stock'] == stock, 'Starting Price'] = df_32.iloc[-1][stock]
portfolio.loc[portfolio['Portfolio'].isin(['A', 'B']), 'Distribution'] = 'T'
portfolio.loc[portfolio['Portfolio'] == 'C', 'Distribution'] = 'Normal'
# print(portfolio)

portfolio_A = portfolio[portfolio['Portfolio'] == 'A']
portfolio_B = portfolio[portfolio['Portfolio'] == 'B']
portfolio_C = portfolio[portfolio['Portfolio'] == 'C']
portfolio_A = portfolio_A.drop('Portfolio', axis=1)
portfolio_B = portfolio_B.drop('Portfolio', axis=1)
portfolio_C = portfolio_C.drop('Portfolio', axis=1)

stocks_A = portfolio_A['Stock'].tolist()
returns_A = returns[stocks_A]
stocks_B = portfolio_B['Stock'].tolist()
returns_B = returns[stocks_B]
stocks_C = portfolio_C['Stock'].tolist()
returns_C = returns[stocks_C]
var_es_A = simulateCopula(portfolio_A, returns_A)
print('Portfolio A:')
print(var_es_A)
var_es_B = simulateCopula(portfolio_B, returns_B)
print('Portfolio B:')
print(var_es_B)
var_es_C = simulateCopula(portfolio_C, returns_C)
print('Portfolio C:')
print(var_es_C)

