# 错误量的定义

对于一神经网络，定义其中第l层第j个神经元的加权输入（即准备送入sigmoid函数的量）为 $z^l_j$，对该量做微小改变（训练），使其该变量为 $\Delta{z^l_j}$. 定义网络最终输出的代价函数为$C$，$\frac{\partial{C}}{\partial{z^l_j}}$ 为代价函数对于该神经元加权输入的敏感度。

定义${\delta}^l_j = \frac{\partial{C}}{\partial{z^l_j}}$ 为“错误量”。直观上看，$|{\delta}^l_j|$接近于0时，此时神经网络到达局部最优点。但是神经网络的训练并非直接以使得${\delta}^l_j$的绝对值最小为目标。目标是使得C最小，而${\delta}^l_j$虽然直观上与C相关，但计算过程中所起的作用仅仅是一个中间变量，用于计算代价函数$C$相对于各神经元的权重、偏置的偏导数。

# 基本方程1

数量形式：
$${\delta}^L_j = \frac{\partial{C}}{\partial{a^L_j}} \sigma'(z^L_j)$$

向量形式：
$${\delta} = {\nabla}_a{C} \odot \sigma'(z^L)$$

证明：

想象在网络中$L$层之后取一个界面，则前面网络中所有的影响都通过改变$L$层所有神经元的激活输出$a^L$来影响$C$。换言之，$C$可以表示为函数$C(a^L_1,a^L_2,...,a^L_k)$

考虑${\delta}^L_j = \frac{\partial{C}}{\partial{z^L_j}}$这一定义，将$C$写成上述函数形式，应用链式求导法则，可得如下公式：
$${\delta}^L_j = \sum_{k}{\frac{\partial{C}}{\partial{a^L_k}}\frac{\partial{a^L_k}}{\partial{z^L_j}}}$$

显然，$a^L_1,a^L_2,...,a^L_k$ 之中，仅有 $a^L_j$ 依赖于 $z^L_j$，对于上述求和式中$k \neq j$的情况，$\frac{\partial{a^L_k}}{\partial{z^L_j}} = 0$。上述求和式可以简化为：
$${\delta}^L_j = \frac{\partial{C}}{\partial{a^L_j}}\frac{\partial{a^L_j}}{\partial{z^L_j}}$$

由于$a^L_j = \sigma(z^L_j)$,上式最后一项 $\frac{\partial{a^L_j}}{\partial{z^L_j}} = \sigma'(z^L_j)$，即为所要证明的公式。

# 基本方程2

依据后一层错误量 $\delta^{L+1}$ 求得前一层错误量 $\delta^{L}$，“反向传播”：
$$\delta^L = \left[(w^{L+1})^T \delta^{L+1}\right] \odot \sigma'(z^L)$$

证明：

类似上面，在$L+1$层后面取截面，可以将$C$表示为函数$C(z^{L+1}_1,z^{L+1}_2,...,z^{L+1}_k)$。再次对$\delta^L_j$的定义式使用链式求导法则，有：
$$ \delta^l_j =  \sum_{k}{\frac{\partial{C}}{\partial{z^{L+1}_k}}\frac{\partial{z^{L+1}_k}}{\partial{z^L_j}}} $$

注意到 $\frac{\partial{C}}{\partial{a^{L+1}_k}}$ 就是 $\delta^{L+1}_k$ 的定义，上式可以化简为：
$$ \delta^l_j =  \sum_{k}{\left({\delta^{L+1}_k}\frac{\partial{z^{L+1}_k}}{\partial{z^L_j}}\right)} $$

为了计算上面和式中的第二项，将$z^{L+1}_k$展开为$z^{L}_i$的函数：
$$z^{L+1}_k = \left[\sum_{i}{w^{l+1}_{ki} \sigma(z^L_i)}\right] + b^{l+1}_k$$

在上式两边对 $z^L_j$ 求导。类似上面，$b^{l+1}_k$ 项消失，求和式中各项只有$i=j$的一项不为零：
$$\frac{\partial{z^{L+1}_k}}{\partial{z^L_j}} = w^{L+1}_{kj} \sigma'(z^L_j)$$

代回上面的$\delta^l_j$计算式，有：
$$ \delta^L_j = \sum_k{\delta^{L+1}_k w^{L+1}_{kj} \sigma'(z^L_j)}  = \sigma'(z^L_j) \left( \sum_k{\delta^{L+1}_k w^{L+1}_{kj} } \right)$$

此即为所需证明公式的分量形式。

# 基本方程3

代价函数对神经元偏置的偏导数等于该神经元上定义的错误量：
$$ \frac{\partial{C}}{\partial{b^L_j}} = \delta^L_j $$

证明：

类似上面，将$C$写成函数$C(z^{L}_1,z^{L}_2,...,z^{L}_k)$，有：
$$\frac{\partial C}{\partial b^L_j} = \sum_k \left( {\frac{\partial C}{\partial z^L_k} \frac{\partial z^L_k}{\partial b^L_j}} \right)$$

显然，当且仅当 $k=j$ 时，求和式中右边项不为零，上式简化为：
$$\frac{\partial C}{\partial b^L_j} =  \frac{\partial C}{\partial z^L_j} \frac{\partial z^L_j}{\partial b^L_j} = \delta^L_j \frac{\partial z^L_j}{\partial b^L_j}$$

由于 $z^L_j = ... + b^L_j$，显然上式最后一项等于1.

原式得证。

# 基本方程4

如何计算代价函数对神经元权重的偏导数：
$$ \frac{\partial{C}}{\partial{w^L_{jk}}} = a^{l-1}_k \delta^L_j $$

证明：

与方程3的证明相同，将$C$写成函数$C(z^{L}_1,z^{L}_2,...,z^{L}_k)$，有：
$$\frac{\partial C}{\partial w^L_{j,k}} = \sum_i \left( {\frac{\partial C}{\partial z^L_i} \frac{\partial z^L_i}{\partial w^L_{j,k}}} \right)$$

类似的，求和式中只有$i=j$的一项有效：
$$\frac{\partial C}{\partial w^L_{j,k}} = \frac{\partial C}{\partial z^L_j} \frac{\partial z^L_j}{\partial w^L_{j,k}} = \delta^L_j \frac{\partial z^L_j}{\partial w^L_{j,k}} = a^{l-1}_k \delta^L_j $$

证明完毕。