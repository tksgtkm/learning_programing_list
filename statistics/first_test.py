import stats
import plot

hist = stats.Hist([1, 2, 2, 3, 5])
print(hist)

print(hist.Freq(2))

print(hist[2])

print(hist.Freq(4))

print(hist.Values)

for val in sorted(hist.Values()):
    print(val, hist.Freq(val))

for val, freq in hist.Items():
    print(val, freq)

plot.Hist(hist)
plot.Show(xlabel='value', ylabel='frequency')