# ML reading group - workshop notes  - 5 Dec 2019 - Athey and Imbens reading

Only Economics people were present.


## Discussion questions

### Why doesn't ML typically allow us to compute CI's (of the parameters of interest)?

Standard econometrics relies on asymptotics of XXX

Can you do it with Lasso/Ridge etc?

Inference is usually done with bootstrap and sensitivity analysis

Can we validate or cross-validate in standard Econometrics? Not if these are being used as predictors of outcomes.

\

### (Related question)  Can I penalise only the *not interesting parameters* to get a more efficient measure of the variable of interest...

### Discussion of whether to use Lasso or another approach like Ridge (other than beleiving in sparsity) ...

- what is the current rules/choices

Does Lasso make sense when some variables only have a very small effect... or would Ridge be better

Is Elastic-net an ensemble method? How can I optimise over both lambda and alpha?

Does this reduce the variance?

\

### Training vs testing ... what is done with which part of the data?

You fit many times on training sample, only once on testing sample to asses the goodness?

Then you once on testing sample sample.

Cross-validation ... at it's extreme it is 'leave one out'

\
### Why do we need a 'set-aside' sample if we've already used cross-validation??

See 'recursive partitioning for heterogeneous causal effects'.


- And what do we need to understand Athey's honest approach


\

Using ML to choose a (?weighted combination) of instruments in an IV approach

### How can we use ML for IV


DR to share a folder/file for us to annotate and comment on

An online group on Yammer/facebook ... or slack?
