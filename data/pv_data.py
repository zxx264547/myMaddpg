import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Microsoft YaHei'
df = pd.read_excel('PV1.xlsx')
# 将所有数据变为正数
df = df.abs()  # 取绝对值，确保所有数据为正数

# 这里使用均值填补，可以根据需求选择其他方法（如中位数、常数等）
df.fillna(df.mean(), inplace=True)  # 使用均值填补缺失值
plt.figure(figsize=(10,5))
plt.plot(df.iloc[:,0], marker='.', linestyle='-', label='PV1')
plt.plot(df.iloc[:,1], marker='.', linestyle='-', label='PV2')
plt.plot(df.iloc[:,2], marker='.', linestyle='-', label='PV3')
plt.plot(df.iloc[:,3], marker='.', linestyle='-', label='PV4')
plt.plot(df.iloc[:,4], marker='.', linestyle='-', label='PV5')
plt.plot(df.iloc[:,5], marker='.', linestyle='-', label='PV6')
plt.title("PV1数据")
plt.xlabel("采样点")
plt.ylabel("功率(MW)")
plt.show()
# PV1.to_csv('PV1.csv')


