import nsfg

df = nsfg.ReadFemPreg()

# pandasの基本的な操作

print(df)

print(df.columns)

pregordr = df['pregordr']
print(type(pregordr))

print(pregordr)

print(pregordr[0])

print(pregordr[2:5])

# 検証

print(df.columns.value_counts().sort_index())

print(df.birthwgt_lb.value_counts().sort_index())

# 解釈

caseid = 10229
preg_map = nsfg.MakePregMap(df)
indices = preg_map[caseid]

print(df.outcome[indices].values)