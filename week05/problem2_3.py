from missing_data import *
from EWMA import *
from nearpsd_higham import *
from chol import *
from simulation import *
from returns_fit_var_es import *


# problem 2
# a. Normal distribution with an exponentially weighted variance (lambda=0.97)
def calculate_ewma_variance(data, lambda_EW):
    ewma_variance = np.zeros(len(data))
    ewma_variance[0] = data.var()  
    for t in range(1, len(data)):
        ewma_variance[t] = lambda_EW * ewma_variance[t-1] + (1 - lambda_EW) * (data[t-1] - data[:t].mean())**2
    return ewma_variance

# Assuming a 95% confidence level
df = pd.read_csv("problem1.csv")
data = df['x']
lambda_EW = 0.97
variance_EW = calculate_ewma_variance(data, lambda_EW)
std_dev_EW = np.sqrt(variance_EW)
z_score = norm.ppf(0.95)
var_normal_EW = -z_score * std_dev_EW[-1]
alpha = 0.05  # Assuming a 95% confidence level, so alpha is 0.05
z_alpha = norm.ppf(alpha)  # This is the same as Φ^{-1}(α)
pdf_at_z_alpha = norm.pdf(z_alpha)  # This is the same as φ(Φ^{-1}(α))
mu = data.mean()
# Calculate ES for a standard normal distribution
es_normal_EW = mu - pdf_at_z_alpha / alpha * std_dev_EW[-1]

print(var_normal_EW, es_normal_EW)

# b. MLE fitted T distribution
var_t = calculate_t_VaR(data)
es_t = calculate_t_ES(data)
print(var_t, es_t)

# c. Historical Simulation
sorted_returns = data.sort_values()
var_historic = -sorted_returns.quantile(0.05)
es_historic = np.mean(data[data<-var_historic])
print(var_historic, es_historic)

# problem 3
portfolio = pd.read_csv('Portfolio.csv')
df_32 = pd.read_csv('DailyPrices.csv')
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

var_es_A.to_csv('myoutput_risk_A.csv')
var_es_B.to_csv('myoutput_risk_B.csv')
var_es_C.to_csv('myoutput_risk_C.csv')
