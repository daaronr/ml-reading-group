(p1)
-??
matrix completion methods


(p1)
we highlight newly developed methods at the intersection of ML and econometrics, methods that typically perform better than either off-the-shelf ML or more traditional econometric methods when applied to particular classes of problems, problems that include causal inference for average treatment effects, optimal policy estimation, and estimation of the counterfactual effect of price changes in consumer choice models

The statistics community has by and large accepted the Machine Learning (ML) revolution that Breiman refers to
as the algorithm modeling culture

methods developed in the ML literature have been particularly successful in -big data” settings, where we observe information on a large number of units, or many pieces of information on each unit, or both, and often outside the simple setting with a single crosssection of units

Economics journals emphasize the use of methods with formal properties of a type that many of the ML methods do not naturally deliver. This includes large sample properties of estimators and tests, including consistency, Normality, and efficiency

\


(p3)
. In contrast, the focus in the machine learning literature is often on working properties of algorithms in specific settings, with the formal results of a different type, e.g., guarantees of error rates


-why and what is meant by this?


> Although the ability to construct valid large sample confidence intervals is important
in many cases

-Is it really true that in a general sense ml can't provide confidence intervals?

Although the ability to construct valid large sample confidence intervals is important
in many cases

(p4)

Often the ML techniques require careful tuning and adaptation to effectively address the specific problems economists are interested in. Perhaps the most important type of adaptation is to exploit the structure of the problems, e.g., the causal nature of many estimands, the endogeneity of variables, the configuration of data such as panel data, the nature of discrete choice among a set of substitutable products, or the presence of credible restrictions motivated by economic theory, such as monotonicity of demand in prices or other shape restrictions (Matzkin [1994, 2007])

adaptation involves changing the optimization criteria of machine learning algorithms to prioritize considerations from causal inference, such as controlling for confounders or discovering treatment effect heterogeneity

\

- does this recover confidence intervals? why do we care about asymptomatic normality? Does this allow us to contruct confidence intervals?

> Finally, techniques such as sample splitting (using different data to select models than to estimate parameters (e.g., Athey and Imbens [2016], Wager and Athey [2017]) and orthogonalization (e.g. Chernozhukov et al. [2016a]) can be used to improve the performance of machine learning estimators, in some cases leading to desirable properties such as asymptotic normality of machine learning estimators (e.g. Athey et al. [2017d], Farrell et al. [2018]).

(p5)
nonparametric regression, or in the terminology of the ML literature, supervised learning
for regression problems

ML approaches to experimental design, where bandit approaches are starting to revolutionize effective experimentation especially in online settings

\

- is this the shrinkage models ridge and lasso? why is it called nonparametric?

> s nonparametric regression, or in the terminology of the ML literature, supervised learning
for regression problems

\

 Second, supervised learning for classification problems, or closely
related, but not quite the same, nonparametric regression for discrete response models. This
is the area where ML methods have perhaps had their biggest successes

other recent reviews of ML methods aimed at economists,

(p6)

The traditional approach in econometrics, as exemplified in leading texts such as Wooldridge [2010], Angrist and Pischke [2008], Greene [2000] is to specify a target, an estimand, that is a functional of a joint distribution of the data. Often the target is a parameter of a statistical model that describes the distribution of a set of variables (typically conditional on some other variables) in terms of a set of parameters, which can be a finite or infinite set. Given a random sample from the population of interest the parameter of interest and the nuisance parameters are estimated by finding the parameter values that best fit the full sample, using an objective function such as the sum of squared errors, or the likelihood function. The focus is on the quality of the estimators of the target, traditionally measured through large sample efficiency. Often there is also interest in constructing confidence intervals. Researchers typically report point estimates and standard errors

\

- this seems problematic in abstract. why should I think that the parameters that best for the data also best estimate the true parameters?

> that best fit the full sample, using an objective function such as the sum of squared errors, or the likelihood function

\



- traditional measure of success

> The focus is on the quality of the estimators of the target, traditionally measured through large sample efficiency

\

In contrast, in the ML literature the focus is typically on developing algorithms to make predictions about some variables given others, or classify

(p7)

\

- which decision?

> The loss associated with this decision may be the squared error

$YN+1 - YˆN+12$.

\

In fact, when the dimension of the features exceeds two, we know from decision theory that we can do better in terms of expected squared error than the least squares estimator. The latter is not admissible, that is, there are other estimators that dominate the least squares estimator

Regression parameters are sometimes referred to as weights

In most discussions on linear regression in econometric textbooks there is little emphasis on model validation. The form of the regression model, be it parametric or nonparametric, and the set of regressors, is assumed to be given from the outside, e.g., economic theory

If there is discussion of model selection, it is
often in the form of testing null hypotheses concerning the validity of a particular model, with
the implication that there is a true model that should be selected and used for subsequent
tasks

(p8)

\

- unbiased in what sense?

> Second, the method uses out-of-sample comparisons, rather than in-sample goodness-of-fit measures. This ensures that we obtain unbiased comparisons of the fit.

\

a large set of models that differ in their complexity

Vapnik-Chervonenkis (VC) dimension that measures the capacity or complexity of a space of models

a term is added to the objective function to penalize the complexity
of the model

Over-fitting, Regularization, and Tuning Parameters

in likelihood settings researchers sometimes add a term to the logarithm of the likelihood function equal to minus the logarithm of the sample size times the number of free parameters divided by two, leading to the Bayesian Information Criterion, or simply the number of free parameters, the Akaike Information Criterion


There are antecedents of this practice in the traditional econometrics and statistics literature. One is that in likelihood settings researchers sometimes add a term to the logarithm of the likelihood function equal to minus the logarithm of the sample size times the number of free parameters divided by two, leading to the Bayesian Information
Criterion, or simply the number of free parameters, the Akaike Information Criterion

\

- on what basis? ad hoc?

> One is that in likelihood settings researchers sometimes add a term to the logarithm of the likelihood function equal to minus the logarithm of the sample size times the number of free parameters divided by two, leading to the Bayesian Information Criterion, or simply the number of free parameters, the Akaike Information Criterion


- interesting

> In Bayesian analyses of regression models the use of a prior distribution on the regression parameters, centered at zero, independent accross parameters with a constant prior variance,
is another way of regularizing estimation that has a long tradition

(p9)

Modern approaches to regularization is that they are more data driven, with the amount
of regularization determined explicitly by the out-of-sample predictive performance rather
than by, for example, a subjectively chosen prior distribution

\

- simple Bayesian updating with conjugacy I guess

> Given the value for the variance of the prior distribution, - 2 , the posterior
mean for - is the solution to

$arg min - X N i=1 Yi - β >Xi 2 + - 2 - 2 k-k 2 , where k-k2 = PK k=1 - 2 k 1/2$

\

-but many of these parameters are unknown with no prior?

> Given the value for the variance of the prior distribution, - 2 , the posterior
mean for - is the solution to
$\argmin - X N i=1 Yi - β >Xi 2 + - 2 - 2 k-k 2 , where k-k2 = PK k=1 - 2 k 1/2$

\

- not sure I see the connection to Bayesian:

chosen. In a formal Bayesian approach this reflects the (subjective) prior distribution on the parameters, and it would be chosen a priori. In an ML approach - would be chosen through out-of-sample cross-validation to optimize the out-of-sample predictive performance

(p10)

\

- but why not use l2 Norm ridge regression?

> Exact sparsity is in fact stronger than is necessary, in many cases it is sufficient to have approximate sparsity where most of the explanatory variables have very limited explanatory power, even if not zero, and only a few of the features are of substantial importance

\

Allowing the data to play a bigger role in the variable selection process appears a clear improvement

\

- best subset is not just testing down

> of LASSO is that there are effective methods for calculating the LASSO estimates with the number of regressors in the millions. Best subset selection regression, on the other hand, is an NP-hard problem

(p11)
current research (Bertsimas et al. [2016]) suggests it may be feasible with
the number of regressors in the 1000s


some indications that in settings with a low
signal to noise ratio, as is common in many social science applications, LASSO may have
better performance, although there remain many open questions


Classic gradient decscent methods involve an iterative approach, where -θk is updated from ˆθk−1 as follows:

-k = θk−1 − ηk 1 N X i -Qi( -θ), where -k is the learning rate, often chosen optimally through line search

\

- move the estimate in the direction of maximum climb or some such

> Classic gradient decscent methods involve an iterative approach, where -θk is updated from ˆθk−1 as follows

> The idea behind SGD is that it is better to take many small steps that
are noisy but on average in the right direction, than it is to spend equivalent computational
cost in very accurately figuring out in what direction to take a single small step

\

(p12)
Ensemble Methods and Model Averaging


In many cases a single model or algorithm does not perform as well as a combination of possibly quite different models, averaged using weights (sometimes called votes) obtained by optimizing out-of-sample performance

For example, one may have three predictive models, one based on a random forest, leading to predictions Y- RF i , one based on a neural net, with predictions Y- NN i , and one based on a linear model estimated by LASSO, leading to Y- LASSO i . Then, using a test sample, one can choose weights p RF , p NN, and p LASSO, by minimizing the sum of squared residuals in the test sample: (-p RF , p- NN , p- LASSO) = arg min pRF,pNN,pLASSO N Xtest i=1

Yi - p RFY- RF i - p NNY- NN i - p LASSOY- LASSO i

(p13)

\

- does this also hold for c confidence intervals on predictions?

> The ML literature has focused heavily on out-of-sample performance as the criterion of interest. This has come at the expense of one of the concerns that the statistics and econometrics
literature have traditionally focused on, namely the ability to do inference

\

keep in mind that the requirements that ensure this ability often come at the expense of predictive performance. One can see this tradeoff in traditional kernel regression, where the bandwidth that optimizes expected squared error balances the tradeoff between the square of the bias and the variance, so that the optimal
estimators have an asymptotic bias that invalidates the use of standard confidence intervals

(p14)
canonical problems in both the ML and econometric literatures is that of estimating the conditional mean of a scalar outcome given a set of of covariates or features


In the settings considered in the ML literature there are often many covariates, sometimes more than there are units in the sample. There is no presumption in the ML literature that the conditional distribution of
the outcomes given the covariates follows a particular parametric model


The derivatives of the conditional expectation for each of the covariates, which in the linear regression model correspond to the parameters, are not of intrinsic interest

\

-!

> there is less of a sense that the conditional expectation is monotone in each of the covariates compared to many economic applications


Often there is concern that the conditional expectation may be an extremely
non-monotone function with some higher order interactions of substantial importance


kernel regression methods have become a popular alternative when more flexibility is required, with subsequently series or sieve methods gaining interest (see Chen [2007]
for a survey). These methods have well established large sample properties, allowing for
the construction of confidence intervals


performing very poorly in settings with high-dimensional covariates, with the difference
g-(x) − g(x) of order Op(N −1/K).


applications of kernel methods in econometrics are generally limited to low-dimensional settings

(p15)

The differences in performance between some of the traditional methods such as kernel
regression and the modern methods such as random forests are particularly pronounced in
sparse settings with a large number of more or less irrelevant covariates. Random forests
are effective at picking up on the sparsity and ignoring the irrelevant features, even if there
are many of them, while the traditional implementations of kernel methods essentially waste
degrees of freedom on accounting for these covariates

\

- Why shouldn't higher order interactions matter in economics?

> modern methods are particularly good at detecting severe nonlinearities and high-order
interactions. The presence of such high-order interactions in some of the success stories of
these methods should not blind us to the fact that with many economic data we expect
high-order interactions to be of limited importance


This is also a reason for the superior performance of locally linear random forests (Friedberg et al. [2018]) relative to standard random forests


. Next we discuss methods based on partitioning the covariate space using regression trees and random forests

(p16)

- What does admissable mean?

> However, if the number of covariates K is large relative to the number of observations N the least squares estimator -ˆls k does not even have particularly good repeated sampling properties as an estimator for -k, let alone good predictive properties. In fact, with K ≥ 3 the least squares estimator is not even admissible and is dominated by estimators that shrink towards zero. With K very large, possibly even exceeding the sample size N, the least squares estimator has particularly poor properties, even if the conditional mean of the outcome given the covariates is in fact linear.

\

As q - 0, the solution penalizes the number of non-zero covariates, leading to best subset regression


relaxed lasso, which combines least squares estimates from the subset selected by LASSO and the LASSO estimates themselves


- agreed!

> It is not always important to have a sparse solution, and often the variable selection that is implicit in these solutions is over-interpreted

(p17)

-!!

> LASSO and ridge have a Bayesian interpretation. Ridge regression gives the posterior mean and mode under a Normal model for the conditional distribution of Yi given Xi , and Normal prior distributions for the parameters. LASSO gives the posterior mode given Laplace prior distributions. However, in contrast to formal Bayesian approaches, the coefficient - on the penalty term is in the modern literature choosen through out-of-sample crossvalidation rather than subjectively through the choice of prior distribution

\

Regression trees (Breiman et al. [1984]), and their extension random forests (Breiman [2001a]

flexibly estimating regression functions in

\

-feels like kernel

> split the sample into subsamples, and estimate the regression function within the subsamples simply as the average outcome. The splits are sequential and based on a single covariate Xik at a time exceeding a threshold c.

\

We split the sample using the covariate
k and threshold c that minimize the average squared error Q(k, c) over all covariates k =
1, . . . , K and all thresholds c - (−∞,∞).

\

- repeat within each group?

> repeat this, now optimizing also over the subsamples or leaves


- number if splits regularization

> One approach is to add a penalty term to the sum

(p18)
linear in the number of subsamples (the leaves).


In practice, a very deep tree is estimated, and then pruned to a more shallow tree using cross-validation to select the optimal tree depth. The sequence of first growing followed by pruning the tree avoids splits
that may be missed because their benefits rely on subtle interactions


the prediction in each leaf is a sample average, and the standard error of that sample average is easy to compute. However, it is not in general true that the sample average of the mean within a leaf is an unbiased estimate of what the mean would be within that same leaf in a new test set. Since the leaves were selected using the data, the leaf sample means in the training data will tend to be more extreme (in the sense of being different from the overall sample mean) than in an independent test set. Athey and Imbens [2016] suggest sample splitting as a way to avoid this issue. If a confidence interval for the prediction is desired, then the analyst can simply split the data in half. One half of the data is used to construct a regression tree. Then, the partition implied by this tree is taken to the other half of the data where the sample mean within a given leaf is an unbiased estimate of the true mean value for the leaf.  Although trees are easy to interpret, it is important not to go too far in interpreting the structure of the tree, including the selection of variables used for the splits. Standard intuitions from econometrics about -omitted variable bias” can be useful here. Particular covariates that have strong associations with the outcome may not show up in splits because the tree splits on covariates highly correlated with those covariates

(p19)
alternative to kernel regression. Within each tree, the prediction for a leaf is simply the sample average outcome within the leaf.  Thus, we can think of the leaf as defining the set of nearest neighbors for a given target observation in a leaf, and the estimator from a single regression tree is a matching estimator with non-standard ways of selecting the nearest neighbor to a target point. In particular, the neighborhoods will prioritize some covariates over others in determining which observations qualify as -nearby

Within each tree, the prediction for a leaf is simply the sample average outcome within the leaf

Kernel regression will create a neighborhood around a target observation based on the Euclidean distance to each point, while tree-based neighborhoods will be rectangles
\

-not great for making a single prediction

> In addition, a target observation may not be in the center of a rectangle. Thus, a single tree is generally not the best way to predict outcomes for any given test point x

\

Random forests induce smoothness by averaging over a large number of trees

First, each tree is based not on the original sample, but on a bootstrap sample (known as bagging (Breiman [1996])) or alternatively on a subsample of the data

Second, the splits at each stage are not optimized over all possible covariates,

(p20)
but over a random subset of the covariates, changing every split. These


average is relatively smooth (although still discontinuous)


Random forests and regression trees are
particularly effective in settings with a large number of features that are not related to the
outcome, that is, settings with sparsity


- it's good at ignoring irrelevant covariates

> Random forests and regression trees are particularly effective in settings with a large number of features that are not related to the outcome, that is, settings with sparsity

\


- useful.

> Wager and Athey [2017] show that a particular variant of random forests can produce
estimates -µ(x) with an asymptotically normal distribution centered on the true value µ(x),
and further, they provide an estimate of the variance of the estimator so that centered
confidence intervals can be constructed. The variant they study uses subsampling rather
than bagging; and further, each tree is built using two disjoint subsamples, one used to
define the tree, and the second used to estimate sample means for each leaf. This honest
estimation is crucial for the asymptotic analysis


\


- an advantage over typical matching

> the forest prioritizes more important covariates for selecting matches in a
data-driven way


, a
kernel regression makes a prediction at a point x by averaging nearby points, but weighting
closer points more heavily. A random forest, by averaging over many trees, will include
nearby points more often than distant points


weighting function for a given test point by counting the share of trees where a particular observation is in the

(p21)
same leaf as a test point


have been extended to settings where the interest is in causal effects


), as well as for
estimating parameters in general economic models

(p22)
    -???

> GMM model is estimated for each test point, where points that are nearby in
the sense of frequently occuring in the same leaf as the test point are weighted more heavily in
estimation


A weakness of forests is that they are not very efficient at capturing linear or quadratic
effects, or at exploiting smoothness of the underlying data generating process. In addition,
near the boundaries of the covariate space, they are likely to have bias, because the leaves of
the component trees of the random forest cannot be centered on points near the boundary


-need more intuition for this

> the leaves of the component trees of the random forest cannot be centered on points near the boundary


- what's the problem?

> Traditional econometrics encounters this boundary bias problem in analyses of regression
discontinuity designs


In their simplest form, local linear
forests just take the forest weights -i(x), and use them for local regression:

(-µ(x), -θ(x)) = argminµ,θ (Xn i=1 -i(x)(Yi − µ(x) − (Xi − x)θ(x))2 + λ||θ(x)||2 2) .

\

- what are all of these components?

(Xn i=1 -i(x)(Yi − µ(x) − (Xi − x)θ(x))2 + λ||θ(x)||2 2

\


(p23)
Deep Learning and Neural Nets


\

-why use these? I'm not convinced the authors understand these models

3.3 Deep Learning and Neural Nets



Given K covariates/features Xik, we model K1 la-


(p24)
tent/unobserved variables Zik (hidden nodes) that are linear in the original covariates:
Z (1) ik = X K j=1 - (1) kj Xij , for k = 1, . . . , K1.

We then modify these linear combinations using a simple nonlinear transformation


then model the outcome as a linear function of this nonlinear transformation of these
hidden nodes plus noise


-how exactly can we fit the coefficients for fully latent variables?
tent/unobserved variables Zik (hidden nodes) that are linear in the original covariates


