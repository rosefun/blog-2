## Linear regression probabilistic perspective

### **TL;DR**

For a linear model, minimizing a sum-of-square error fucntion is equivalent to maximinzing the likelihood fucntion under a conditional Guassian noise distrubution.

minimizing a sum-of-square error fucntion with the **addition of a quadratic regularization term** is equivalent to maximizing posterior distribution(bayesian version of linear model).

### Likelihood, prior and posterior

Recall that a linear regression model is actually a transformation from input \(\mathbf x\) to output \(\mathbf y\), governed by parameter \(\mathbf w\). The bayesian theory thinks the parameter \(\mathbf w\) is not a constant but some distribution \(p(\mathbf w)\). During training phase, output \(\mathbf y\) is observed.

  * \(p(\mathbf w)\) -> prior distribution
  * \(p(\mathbf y|\mathbf w, \mathbf x)\) -> likelihood function
  * the conditional ditribution \(p(\mathbf w|\mathbf y, \mathbf x)\) represents the corresponding **posterior distribution** over \(\mathbf w\). 

We are not seeking to model the distribution of \(\mathbf x\). Thus it will always appear in the set of conditioning variables and we could even drop it to keep the notation compact. 

#### Readings

  * prml 2.3.3 bayes's theorem for Guassian variables
  * prml 3.1.1 maximum likelihood and least squares
  * prml 3.1.4 regularized least squares
  * prml 3.3.1 parameter distribution for bayesian linear regression
  * prml 1.5.4 Inference and decision
  * [Generative vs. Discriminative; Bayesian vs. Frequentist](https://lingpipe-blog.com/2013/04/12/generative-vs-discriminative-bayesian-vs-frequentist/)
  * [All Bayesian Models are Generative (in Theory)](https://lingpipe-blog.com/2013/05/23/all-bayesian-models-are-generative-in-theory/)