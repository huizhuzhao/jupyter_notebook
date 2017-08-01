### unbiased variance
* suppose we want to estimate the *mean* and *variance* of variable $x$ according to the observed samples $\{x_1, x_2, ..., x_N\}$. If we have enough samples, i.e $N \rightarrow \infty$, thus

$$
\begin{array}{lll}
\mu &= & \lim_{N \rightarrow \infty}\frac{1}{N}\sum_{i} x_i \\
\sigma^2 &= & \lim_{N \rightarrow \infty}\frac{1}{N}\sum_{i}(x_i - \mu)^2
\end{array}
$$

So, $\hat{\mu} =  \frac{1}{N}\sum_{i} x_i$ approxiates $\mu$; but 
$$\hat{\sigma}^2 =  \frac{1}{N}\sum_{i}(x_i - \hat{\mu})^2$$ could be further reformulated as following

$$
\begin{array}{lll}
\hat{\sigma}^2 &= & \frac{1}{N}\sum_{i}(x_i - \hat{\mu})^2 \\
& = & (\frac{N-1}{N})^2 \frac{1}{N} \sum_{i} (x_i - \hat{\mu}_i)^2  \\
& = & (\frac{N-1}{N})^2 \bar{\sigma}^2
\end{array}
$$
where $\hat{\mu}_i \equiv \frac{1}{N-1}\sum_{j\neq i} x_j$

In the above, we can see that $\bar{\sigma}^2$
will be much closer to $\sigma^2$ than $\hat{\sigma}^2$


### joblib vs numpy vs cPickle
我们使用矩阵 **x_dense** 和 **x_sparse**　对`joblib`, `numpy` 和 `cPickle`存储/读取性能进行测试, 其中

`x_dense = np.random.binomial(2, 0.01, size=(1000, 1000))` 

```
indices = np.where(x_dense != 0)
data = x_dense[indices]
x_sparse = csr_matrix((data, indices), shape=x_dense.shape)
```

测试存储的文件字节大小以及读取文件时间分别为:

|        |dense |          |sparse|       |
|--------|------|----------|------|----------|
|        | bytes| load time|bytes | load time|
|joblib  |100k  | 7.21ms   |40k   | 0.495ms  |
|numpy   |7.7M  | 1.18ms   |240k  | 0.181ms  |
|cPickle  |31M| 1.02s    |928k  | 30.3ms   |


因此可以看出, joblib 和 numpy 分别在文件大小和读取时间上各自有优势，而 cPickle 在两方面都远远落后。