## CRF

CRF的全称是条件随机场（Conditional Random Field），是概率图模型的一种，常用来做序列标注。这个名字掩盖了它的本质，它实际上是LR在序列上的扩展，因此CRF也被称为“序列最大熵”算法。

LR中，预测值是一个标量（尽管我们将他表示成了一个one-hot vector），LR建模了给定x时输出为y的后验概率\(p(y|x;w,b)\)。CRF中输出y是一个矢量，代表着\(x\)的每个位置对应的输出（标注）。记x和y的长度为n，y每一维的可能取值有k种，则y总的取值可能有\(k^n\)种。此时，CRF就可以看作是一个类别数为\(k^n\)的LR模型。

唯一尚需的工作是如何建模y取每一种可能时的概率。常用的Linear Chain CRF是如下建模，它只考虑当前\(y_j\)和前一个位置\(y_{j-1}\)，是一阶的，得名linear chain。

\[p(x,y;w)=\frac{\exp(w \sum\limits_{j=1}^{n}\phi(x,j,y_{j-1},y_j))}{\sum\limits_{y'\in Y}\exp(w \sum\limits_{j=1}^{n}\phi(x,j,y'_{j-1},y'_j))}\]

### Training

CRF的参数训练同样是Maximum log likelihood estimation。不同于LR的是，由于y的解空间较大，CRF对数似然代价函数的归一化项会导致梯度求解时的大量计算。动态规划算法"forward-backward"可以降低这一计算的复杂度，是CRF训练过程中必不可少的重要环节。

### Inference

Veterbi算法。归一化项不会影响判决，因此不用计算。
