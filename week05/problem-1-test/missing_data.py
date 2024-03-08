import numpy as np
import pandas as pd

class MissingData:
    def __init__(self, input_file):
        self.input_file = input_file
        self.data = pd.read_csv(input_file)

    def SkipMissingRows_cov(self):
        cleaned_data = self.data[~np.isnan(self.data).any(axis=1)]
        return cleaned_data.cov()
    
    def SkipMissingRows_corr(self):
        cleaned_data = self.data[~np.isnan(self.data).any(axis=1)]
        return cleaned_data.corr()

    def Pairwise_cov(self):
        n, m = self.data.shape
        matrix = np.empty((m, m))
        matrix[:] = np.nan

        for i in range(m):
            for j in range(m):
                if i <= j:
                    # Select rows where both columns i and j have non-missing data
                    valid_rows = ~self.data.iloc[:, i].isna() & ~self.data.iloc[:, j].isna()
                    if valid_rows.sum() > 1:
                        # Calculate covariance between columns i and j for valid rows
                        matrix[i, j] = np.cov(self.data.loc[valid_rows, self.data.columns[i]], self.data.loc[valid_rows, self.data.columns[j]])[0, 1]
                    if i != j:
                        matrix[j, i] = matrix[i, j]  # Mirror the value for the lower triangle
        return pd.DataFrame(self.data.cov())
    
    def Pairwise_corr(self):
        n, m = self.data.shape
        matrix = np.empty((m, m))
        matrix[:] = np.nan

        for i in range(m):
            for j in range(m):
                if i <= j:
                    # Select rows where both columns i and j have non-missing data
                    valid_rows = ~self.data.iloc[:, i].isna() & ~self.data.iloc[:, j].isna()
                    if valid_rows.sum() > 1:
                        # Calculate covariance between columns i and j for valid rows
                        matrix[i, j] = np.cov(self.data.loc[valid_rows, self.data.columns[i]], self.data.loc[valid_rows, self.data.columns[j]])[0, 1]
                    if i != j:
                        matrix[j, i] = matrix[i, j]  # Mirror the value for the lower triangle
        return pd.DataFrame(self.data.corr())

    
