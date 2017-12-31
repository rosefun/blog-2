## Linear regression:probabilistic perspective

### **TL;DR**

For a linear model, minimizing a sum-of-square error fucntion is equivalent to maximinzing the likelihood fucntion under a conditional Guassian noise distrubution.

minimizing a sum-of-square error fucntion with the **addition of a quadratic regularization term** is equivalent to maximizing posterior distribution(bayesian version of linear model).

### Likelihood, prior and posterior

In statistics, a likelihood function (often simply the likelihood) is a function of the parameters of a statistical model given data. We call:

  * \(p(\theta)\) -> prior distribution
  * \(p(D|\theta)\) -> likelihood function
  * \(p(\theta|D)\) -> posterior distribution

They are linked through bayes' theorem.

Returning back to bayesian model, recall that a model is actually a transformation from input \(\mathbf x\) to output \(\mathbf y\), governed by parameter \(\mathbf w\). The bayesian theory thinks the parameter \(\mathbf w\) is not a constant but some distribution \(p(\mathbf w)\). During training phase, output \(\mathbf y\) is observed. Comopared to bayesian theory, \(\mathbf w \Rightarrow \theta, \mathbf y \Rightarrow D\),

  * \(p(\mathbf w)\) -> prior distribution
  * \(p(\mathbf y|\mathbf w, \mathbf x)\) -> likelihood function
  * the conditional ditribution \(p(\mathbf w|\mathbf y, \mathbf x)\) represents the corresponding **posterior distribution** over \(\mathbf w\). 

We are not seeking to model the distribution of \(\mathbf x\). Thus it will always appear in the set of conditioning variables and we could even drop it to keep the notation compact. It is clear that when we are talking about bayesian model, we often refer to the way we model parameters \(\mathbf w\) and data \(\mathbf D\)(likelihood or posterior), not model's input \(\mathbf x\) and \(\mathbf y\). 

Those concepts here are usually confused to discriminative vs generative. Read the appendix to distiguish them.

**Conjugate prior**

If the posterior distributions p(w|y) are in the same family as the prior probability distribution p(w), 

  * the prior and posterior are then called conjugate distributions
  * and the prior is called a [conjugate prior](https://en.wikipedia.org/wiki/Conjugate_prior) **for the likelihood function**.

Guassian prior is essential to link l2-regularized LSE loss to a bayesian model. 

    The Gaussian family is conjugate to itself (or self-conjugate) with respect to a Gaussian likelihood function: if the likelihood function is Gaussian, choosing a Gaussian prior over the mean will ensure that the posterior distribution is also Gaussian.

#### Discriminative/generative & frequentist/bayesian

The difference between discriminitive and generative is how they model the relationship between model's input x and output y. Generative models explicitly or implicitly model the distribution of inputs as well as outputs. Discriminative models model the posterior probabilities directly.

|     d/g            |      modeling       |
|--------------------|:-------------------:|
| discriminative     |  \(p(y\|x)\)           |
| generative         |    \(p(x,y)\)           | 

The debate between bayesian and frequist is how they model the relationship between model's data (x,y) and parameters w. The bayesian think everything is a variable.

|    f/b             |      modeling       |
|--------------------|:-------------------:|
|   frequist   |       \(p(x,y\|w)\) or \(p(y\|x, w)\)         |
|  bayesian        |    \(p(x,y,w)\)         | 

If you are going to go deeper, refer to some readings at the end.

#### Readings

  * prml 2.3.3 bayes's theorem for Guassian variables
  * prml 3.1.1 maximum likelihood and least squares
  * prml 3.1.4 regularized least squares
  * prml 3.3.1 parameter distribution for bayesian linear regression
  * prml 1.5.4 Inference and decision
  * [Generative vs. Discriminative; Bayesian vs. Frequentist](https://lingpipe-blog.com/2013/04/12/generative-vs-discriminative-bayesian-vs-frequentist/)
  * [All Bayesian Models are Generative (in Theory)](https://lingpipe-blog.com/2013/05/23/all-bayesian-models-are-generative-in-theory/)