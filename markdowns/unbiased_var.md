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