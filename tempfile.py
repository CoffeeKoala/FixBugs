print("{str1:.<20}{count:.>5}".format(str1 = 'Fetch Row Count'   ,count = 120))
print("{str1:.<20}{count:.>5}".format(str1 = 'Total Result Count',count = 20))

output

Fetch Row Count.......120
Total Result Count.....20



N = 4
x = pd.qcut(df[var],N)
bins = x.dtype.categories.values.to_tuples()
labels = [ str(i) + "_"+ str(bins[i]) for i in range(0,N)]
df[var+ '_bin'] =  pd.qcut(df[var],N,labels = labels)
df[var+'_bin'].value_counts().sort_index()
