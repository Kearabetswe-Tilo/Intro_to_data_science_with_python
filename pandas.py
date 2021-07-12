
import numpy as np
import pandas as pd

s = pd.Series(np.random.randn(5), index = ['a', 'b', 'c', 'd', 'e'])
print(s)
print(s.index)
print(pd.Series(np.random.randn(5)))


d = {'b': 1, 'a': 0, 'c': 2}
print(pd.Series(d))


d = {'a': 0., 'b': 1., 'c': 2.}
print(pd.Series(d))
print(pd.Series(d, index=['b', 'c', 'd', 'a']))


print(pd.Series(5., index=['a', 'b', 'c', 'd', 'e']))


s = pd.Series(np.random.randn(5), index = ['a', 'b', 'c', 'd', 'e'])
print(s)
print(s.index)
print(s[0])
print(s[:3])
print(s[s > s.median()])
print(s[[4, 3, 1]])
print(np.exp(s))


s = pd.Series(np.random.randn(5), index = ['a', 'b', 'c', 'd', 'e'])
print(s)
print(s['a'])
s['e'] = 12.

print(s)

print('e' in s)
print('f' in s)


s = pd.Series(np.random.randint(1, 10, 5), index = ['a', 'b', 'c', 'd', 'e'])
print(s)

print(s + s)
print(s * 3)
print(np.exp(s))

print(s[1:] + s[:-1])


d = {'one': pd.Series([1., 2., 3.], index=['a', 'b', 'c']), 
     'two': pd.Series([1., 2., 3., 4.], index=['a', 'b', 'c', 'd'])}

df = pd.DataFrame(d)

print(df)
print(pd.DataFrame(d, index=['d', 'b', 'a']))
print(pd.DataFrame(d, index=['d', 'b', 'a'], columns=['two', 'three']))


print(df.index)
print(df.columns)


d = {'one': [1., 2., 3., 4.], 'two': [4., 3., 2., 1.]}

print(pd.DataFrame(d))
print(pd.DataFrame(d, index=['a', 'b', 'c', 'd']))


data = np.zeros((2, ), dtype=[('A', 'i4'), ('B', 'f4'), ('C', 'a10')])
print(data)

data[:] = [(1, 2., 'Hello'), (2, 3., "World")]
print(pd.DataFrame(data))
print(pd.DataFrame(data, index=['first', 'second']))
print(pd.DataFrame(data, columns=['C', 'A', 'B']))


data2 = [{'a': 1, 'b': 2}, {'a': 5, 'b': 10, 'c': 20}]

print(pd.DataFrame(data2))
print(pd.DataFrame(data2, index=['first', 'second']))
print(pd.DataFrame(data2, columns=['a', 'b']))


s = pd.Series([1, 3, 5, np.nan, 6, 8])
print(s)



dates = pd.date_range('20130101', periods = 6)
print(dates)

df = pd.DataFrame(np.random.randn(6, 4), index = dates, columns = list('ABCD'))
print(df)



df2 = pd.DataFrame({'A': 1., 'B': pd.Timestamp('20130102'), 
   'C': pd.Series(1, index  = list(range(4)), dtype = 'float32'), 
   'D': np.array([3] * 4, dtype='int32'), 
   'E': pd.Categorical(["test", "train", "test", "train"]), 'F': 'foo'})

print(df2)

print(df2.dtypes)
print(df2.head())
print(df2.tail())
print(df2.index)
print(df2.columns)

print(df2.to_numpy)



d = {'one': pd.Series([1., 2., 3.], index=['a', 'b', 'c']), 
     'two': pd.Series([1., 2., 3., 4.], index=['a', 'b', 'c', 'd'])}

df = pd.DataFrame(d)
print(df)
print(df.describe())

print(df.T)

print(df.sort_index(axis = 0, ascending = False))


df2 = pd.DataFrame({'A': 1., 'B': pd.Timestamp('20130102'), 
   'C': pd.Series(1, index  = list(range(4)), dtype = 'float32'), 
   'D': np.array([3] * 4, dtype='int32'), 
   'E': pd.Categorical(["test", "train", "test", "train"]), 'F': 'foo'})

print(df2)


print(df2.sort_values(by = 'E'))

print(df2['A'])
print(df2[0:3])

print(df2.loc[2])
print(df2.iloc[2])

df2.index = ['R1', 'R2', 'R3', 'R4'] 
print(df2.loc['R2'])
print(df2.iloc[2])

print(df2[df2['E'] == "train"])

s1 = pd.Series([1, 2, 3, 4, 5, 6], index=pd.date_range('20130102', periods=6))
df['F'] = s1
df.at[dates[0], 'A'] = 0
df.iat[0, 1] = 0
df.loc[:, 'D'] = np.array([5] * len(df))
df2 = df.copy()
df2[df2 > 0] = -df2










