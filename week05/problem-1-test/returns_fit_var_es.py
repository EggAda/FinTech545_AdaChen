import numpy as np
import pandas as pd
from scipy.stats import norm, t
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm 
from scipy.optimize import minimize
from simulation import simulate_pca

# def calculate_arithmetic_returns(prices):
#     return (prices[1:] - prices[:-1]) / prices[:-1]
# 6 calculate arithmetic / log returns
def return_calculate(prices, method="DISCRETE", date_column="date"):
    if date_column not in prices.columns:
        raise ValueError(f"dateColumn: {date_column} not in DataFrame: {prices.columns}")
    
    # Extract prices and compute returns
    p = prices.drop(columns=[date_column]).values
    n, m = p.shape
    p2 = np.empty((n - 1, m))
    
    for i in range(n - 1):
        for j in range(m):
            p2[i, j] = p[i + 1, j] / p[i, j]
    
    if method.upper() == "DISCRETE":
        p2 = p2 - 1.0
    elif method.upper() == "LOG":
        p2 = np.log(p2)
    else:
        raise ValueError(f"method: {method} must be in (\"LOG\", \"DISCRETE\")")
    
    # Prepare the output DataFrame
    dates = prices[date_column].iloc[1:]
    out = pd.DataFrame(data=p2, columns=prices.columns.drop(date_column))
    out.insert(0, date_column, dates.values)
    
    return out


def fit_normal_distribution(data):
    mu, std = norm.fit(data)
    return mu, std

def fit_t_distribution(data):
    # t值的绝对值较大意味着样本均值与假设均值之间的差异较大
    params = t.fit(data)
    df, loc, scale = params
    # result_df = pd.DataFrame({
    #     'mu': [loc],        # loc parameter is the mean for the t-distribution
    #     'sigma': [scale],   # scale parameter is the standard deviation for the t-distribution 尺度参数越大，分布就越宽，尾部就越厚。
    #     'nu': [df]          # df parameter is the degrees of freedom for the t-distribution自由度
    # })
    return df, loc, scale

def MLE_t(X, Y):
    X = sm.add_constant(X)
    def ll_t(params):
        df = params[0]
        s = params[1]
        b = params[2:]
        e = Y - np.dot(X, b)
        ll = np.sum(t.logpdf(e, df=df, loc=0, scale=s))
        return -ll
    beta = np.zeros(X.shape[1])
    s = np.std(Y - np.dot(X, beta))
    df = 1
    params = [df, s]
    for i in beta:
        params.append(i)
    bnds = ((0, None), (1e-9, None), (None, None), (None, None), (None, None), (None, None))
    res = minimize(ll_t, params, bounds=bnds, options={"disp": False})
    beta_mle = res.x[2:]
    return beta_mle

def t_regression(data):
    X = data.iloc[:,:-1]
    Y = data.iloc[:,-1]
    betas = MLE_t(X, Y)
    X = sm.add_constant(X)
    e = Y - np.dot(X, betas)
    df, loc, scale = t.fit(e)
    return loc, scale, df, betas[0], betas[1], betas[2], betas[3]

# 8-VaR calculation methods Given data and alpha, return the VaR
def calculateVar(data, u=0, alpha=0.05):
    return u-np.quantile(data, alpha)

def calculate_normal_VaR(data, alpha=0.05):
    mu, sigma = fit_normal_distribution(data)
    Z_alpha = norm.ppf(alpha)
    var_norm = mu - Z_alpha * sigma
    return abs(var_norm) # +

def calculate_t_VaR(data, alpha=0.05):
    params = t.fit(data)
    VaR_t = t.ppf(alpha, *params)
    return abs(VaR_t) # +

# VaR for t Distribution simulation
def calculate_simulation_VaR(data, alpha=0.05, size = 10000):
    nu, mu, sigma = fit_t_distribution(data) 
    # Generate given size random numbers, t-distribution
    random_numbers = t.rvs(df=nu, loc=mu, scale=sigma, size=size)
    return calculate_t_VaR(random_numbers, alpha)

# 8-ES calculation
def calculate_normal_ES(data, u=0, alpha=0.05):
    # print(calculate_normal_VaR(data)) 0.12258619228053136
    # print(abs(np.mean(data[data<-calculate_normal_VaR(data)]))) nan
    return abs(np.mean(data[data<=calculate_normal_VaR(data)]))

def calculate_t_ES(data, u=0, alpha=0.05):
    return abs(np.mean(data[data<-calculate_t_VaR(data)]))

def calculate_simulation_ES(data, u=0, alpha=0.05):
    return abs(np.mean(data[data<-calculate_simulation_VaR(data)]))


# 9-VaR/ES on 2 levels from simulated values - Copula

def simulateCopula(portfolio, returns):
    portfolio['CurrentValue'] = portfolio['Holding'] * portfolio['Starting Price']
    models = {}
    uniform = pd.DataFrame()
    standard_normal = pd.DataFrame()
    
    for stock in returns.columns:
        # If the distribution for the model is normal, fit the data with normal distribution.
        if portfolio.loc[portfolio['Stock'] == stock, 'Distribution'].iloc[0] == 'Normal':
            models[stock] = norm.fit(returns[stock])
            mu, sigma = norm.fit(returns[stock])
            
            # Transform the observation vector into a uniform vector using CDF.
            uniform[stock] = norm.cdf(returns[stock], loc=mu, scale=sigma)
            
            # T将统一向量转为标准正态向量ransform the uniform vector into a Standard Normal vector usig the normal quantile function.
            standard_normal[stock] = norm.ppf(uniform[stock])
            
        # the distribution for the model is t
        elif portfolio.loc[portfolio['Stock'] == stock, 'Distribution'].iloc[0] == 'T':
            models[stock] = t.fit(returns[stock])
            nu, mu, sigma = t.fit(returns[stock])
            
            # Transform the observation vector into a uniform vector using CDF.
            uniform[stock] = t.cdf(returns[stock], df=nu, loc=mu, scale=sigma)
            
            # Transform the uniform vector into a Standard Normal vector usig the normal quantile function.
            standard_normal[stock] = norm.ppf(uniform[stock])
            
    # Calculate Spearman's correlation matrix 这种相关性度量捕获非线性和非正态分布之间的关联。
    spearman_corr_matrix = standard_normal.corr(method='spearman')
    
    nSim = 10000
    
    # Use the PCA to simulate the multivariate normal.
    simulations = simulate_pca(nSim, spearman_corr_matrix)
    simulations = pd.DataFrame(simulations.T, columns=[stock for stock in returns.columns])
    
    # 均匀分布变量Transform the simulations into uniform variables using standard normal CDF. uni = norm.cdf(simulations)
    uni = norm.cdf(simulations)
    uni = pd.DataFrame(uni, columns=[stock for stock in returns.columns])
    
    simulatedReturns = pd.DataFrame()
    # Transform the uniform variables into the desired data using quantile.
    for stock in returns.columns:
        # If the distribution for the model is normal, use the quantile of the normal distribution.
        if portfolio.loc[portfolio['Stock'] == stock, 'Distribution'].iloc[0] == 'Normal':
            mu, sigma = models[stock]
            simulatedReturns[stock] = norm.ppf(uni[stock], loc=mu, scale=sigma)
            
        # If the distribution for the model is t, use the quantile of the t distribution.
        elif portfolio.loc[portfolio['Stock'] == stock, 'Distribution'].iloc[0] == 'T':
            nu, mu, sigma = models[stock]
            simulatedReturns[stock] = t.ppf(uni[stock], df=nu, loc=mu, scale=sigma)
    
    simulatedValue = pd.DataFrame()
    pnl = pd.DataFrame()
    # Calculate the daily prices for each stock
    for stock in returns.columns:
        currentValue = portfolio.loc[portfolio['Stock'] == stock, 'CurrentValue'].iloc[0]
        simulatedValue[stock] = currentValue * (1 + simulatedReturns[stock])
        pnl[stock] = simulatedValue[stock] - currentValue # 每日盈亏
        
    risk = pd.DataFrame(columns = ["Stock", "VaR95", "ES95", "VaR95_Pct", "ES95_Pct"])
    w = pd.DataFrame()

    for stock in pnl.columns:
        i = risk.shape[0]
        risk.loc[i, "Stock"] = stock
        risk.loc[i, "VaR95"] = -np.percentile(pnl[stock], 5)
        risk.loc[i, "VaR95_Pct"] = risk.loc[i, "VaR95"] / portfolio.loc[portfolio['Stock'] == stock, 'CurrentValue'].iloc[0]
        risk.loc[i, "ES95"] = -pnl[stock][pnl[stock] <= -risk.loc[i, "VaR95"]].mean()
        risk.loc[i, "ES95_Pct"] = risk.loc[i, "ES95"] / portfolio.loc[portfolio['Stock'] == stock, 'CurrentValue'].iloc[0]
        
        # Determine the weights for the two stock 权重
        w.at['Weight', stock] = portfolio.loc[portfolio['Stock'] == stock, 'CurrentValue'].iloc[0] / portfolio['CurrentValue'].sum()
        
    # Calculate the total pnl. 总盈亏
    pnl['Total'] = 0
    for stock in returns.columns:
        pnl['Total'] += pnl[stock]
    
    i = risk.shape[0]
    risk.loc[i, "Stock"] = 'Total'
    risk.loc[i, "VaR95"] = -np.percentile(pnl['Total'], 5)
    risk.loc[i, "VaR95_Pct"] = risk.loc[i, "VaR95"] / portfolio['CurrentValue'].sum()
    risk.loc[i, "ES95"] = -pnl['Total'][pnl['Total'] <= -risk.loc[i, "VaR95"]].mean()
    risk.loc[i, "ES95_Pct"] = risk.loc[i, "ES95"] / portfolio['CurrentValue'].sum()
    
    return risk

