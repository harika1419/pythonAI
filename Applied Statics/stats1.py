import numpy as np
import pandas as pd
from scipy import stats

df = pd.read_csv("Stats.csv")
Experience = df.iloc[:,1]
Salary = df.iloc[:,2]

print("----------- Mean,Median & Mode of Experience------------")
print("Mean of Exprience",np.mean(Experience))
print("Median of Exprience",np.median(Experience))
print("Mode of Exprience",stats.mode(Experience))

print("----------- Mean,Median & Mode of Salary------------")
print("Mean of Salary",np.mean(Salary))
print("Median of Salary",np.median(Salary))
print("Mode of Salary",stats.mode(Salary))