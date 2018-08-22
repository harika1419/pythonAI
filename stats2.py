import numpy as np
import pandas as pd
from scipy import stats

df = pd.read_csv("Stats.csv")
Experience = df.iloc[:,1]
Salary = df.iloc[:,2]

print("----------- Standard Deviation and Varience of Experience------------")
print("Standard Deviation of Exprience",np.std(Experience))
print("Varience of Exprience",np.var(Experience))

print("----------- Standard Deviation and Varience of Salary------------")
print("Standard Deviation of Exprience",np.std(Salary))
print("Varience of Exprience",np.var(Salary))