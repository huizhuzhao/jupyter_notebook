## softmax

Now, we have vector $x = [x_1, x_2, .., x_n]$, which is the output of one classification model; however values $x_i$ are unnormalized, i.e. $\sum_{k=1}^n x_k != 1$.

We will use **softmax** function to normalize $x$, and the formular is

$
\begin{array}{ll}
x^{normal} & = \left [e^{-x_1}, e^{-x_2}, ..., e^{-x_n}\right ]/\sum_{k=1}^n e^{-x_k} \\ \\
&  = \left [ g_1, g_2, ..., g_n \right ]  \\ \end{array}
$

where $g_i =  1 / \sum_{k=1}^n e^{-(x_k - x_i)} \\ $


In case, X is a 2D tensor, i.e


$
\begin{align}
X = \left( \begin{array}{ccc}
x_{11} & x_{12} & ... & x_{1n} \\
x_{21} & x_{22} & ... & x_{2n} \\
... \\
x_{m1} & x_{m2} & ... & x_{mn} \end{array} \right)
\end{align}
$

after normalization, we have

$
X^{normal} = \left( \begin{array}{ccc}
g_{11} & g_{12} & .. & g_{1n} \\
g_{21} & g_{22} & .. & g_{2n} \\
... \\
g_{m1} & g_{m2} & .. & g_{mn} \end{array} \right)
$

where $g_{ij} =  1 / \sum_{k=1}^n e^{-(x_{ik} - x_{ij})}$


$
\begin{array}{ccc}
\bar{x}_{n+1} & = &\frac{1}{n+1}\sum_{i=1}^{n+1} x_i \\
& = &\bar{x}_{n} + \frac{x_{n+1}-\bar{x}_{n}}{n+1} \\
\end{array}
$

$
\begin{array}{ccc}
\bar{x}_{n+1} & = &\frac{1}{n+1}\sum_{i=1}^{n+1} x_i \\
& = &\bar{x}_{n} + \frac{x_{n+1}-\bar{x}_{n}}{n+1} \\
\end{array}
$