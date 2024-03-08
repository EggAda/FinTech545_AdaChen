import numpy as np
import pandas as pd

class EWMAStatistics:
    def __init__(self, input_file):
        self.data = pd.read_csv(input_file)
    
    def calculate_ewma_covariance(self, lambda_EW):
        n_obs = self.data.shape[0]
        n_stocks = self.data.shape[1] 
        ewma_covariance = np.zeros((n_stocks, n_stocks))  
        weights = np.array([(lambda_EW)**(n_obs-1-i) for i in range(n_obs)])
        for i in range(n_stocks):
            for j in range(i, n_stocks):
                if i == j:
                    ewma_covariance[i, j] = np.average((self.data.iloc[:, i] - self.data.iloc[:, i].mean())**2, weights=weights)
                else:
                    ewma_covariance[i, j] = np.average((self.data.iloc[:, i] - self.data.iloc[:, i].mean()) * (self.data.iloc[:, j] - self.data.iloc[:, j].mean()), weights=weights)
                ewma_covariance[j, i] = ewma_covariance[i, j]
        return ewma_covariance
    
    def calculate_ewma_correlation(self, lambda_EW):
        ewma_covariance = self.calculate_ewma_covariance(lambda_EW)
        n_stocks = ewma_covariance.shape[0]
        ewma_correlation = np.zeros((n_stocks, n_stocks))
        ewma_volatility = np.sqrt(np.diag(ewma_covariance))
        for i in range(n_stocks):
            for j in range(i, n_stocks):
                ewma_correlation[i, j] = ewma_covariance[i, j] / (ewma_volatility[i] * ewma_volatility[j])
                ewma_correlation[j, i] = ewma_correlation[i, j]
        return ewma_correlation

    def calculate_ewma_variance(self, lambda_EW):
        n_obs = self.data.shape[0]  
        n_stocks = self.data.shape[1]  
        ewma_variance_matrix = np.zeros((n_stocks, n_stocks)) 
        for i in range(n_stocks):
            stock_data = self.data.iloc[:, i]
            ewma_variance = np.zeros(n_obs)
            ewma_variance[0] = stock_data.var() 
            for t in range(1, n_obs):
                ewma_variance[t] = lambda_EW * ewma_variance[t-1] + (1 - lambda_EW) * (stock_data[t-1] - stock_data[:t].mean())**2
            ewma_variance_matrix[i, i] = ewma_variance[-1]  
        return ewma_variance_matrix
    
