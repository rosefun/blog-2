## primal&dual:linear regression as an example

### TL;DR

**primal**

problem: $\mathbf{y}=\mathbf{x}^T\mathbf{w}$

loss: $l=||\mathbf{X}^T\mathbf{w}-\mathbf{y}||^2+1/2*||\mathbf{w}||^2$

solution(learn $w$): $\mathbf{w}=(\mathbf{X}^T\mathbf{X}+\lambda\mathbf{I})^{-1}\mathbf{X}^T\mathbf{y}$

**dual**

fact: $w$ lies in the space spanned by training data.

problem: $\mathbf{y}=\mathbf{x}^T(\mathbf{X}^T\mathbf\alpha)=\sum\limits_{i=1}^M\alpha_i\mathbf{x}^T\mathbf{X_i}$ ($\mathbf{X_i}$ is the ith row)

solution(learn $\alpha$): $\mathbf\alpha=(\mathbf{X}\mathbf{X}^T+\lambda\mathbf{I})^{-1}\mathbf{y}$

### Matrix caculus

To get solutions of loss funtions, we need [matrix derivative calculations](./matrix+vector+derivatives+for+machine+learning.pdf). It is not a new math opertion but  partial derivations to vector/matrix element-wise. The matrix notation help us to have a compact represention instead of writing down derivation for each element explicitly.

People already work out quick lookup tables for fundamental identities. To calculate $\partial l / \partial \mathbf{w}$, here we need is $\frac {\partial (\mathbf{A}\mathbf{x}+\mathbf{b})\mathbf{C}(\mathbf{D}\mathbf{x}+\mathbf{E})} {\partial \mathbf{x}}$ in [scalar by matrix identities section](https://en.wikipedia.org/wiki/Matrix_calculus#Scalar-by-vector_identities).

### Calculation of $\alpha$

$$\mathbf{\alpha} = (\mathbf{X}^T)^{-1} \mathbf w = (\mathbf{X}^T)^{-1} (\mathbf{X}^T\mathbf{X}+\lambda\mathbf{I})^{-1}\mathbf{X}^T \mathbf{y} = {[(\mathbf{X}^T)^{-1} (\mathbf{X}^T\mathbf{X}+\lambda\mathbf{I}) \mathbf{X}^T]}^{-1} \mathbf{y}$$