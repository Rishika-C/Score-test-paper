library(RTMB)

###########################################################################################

if (F) {
  ## Simple RTMB illustrative example for Appendix :)
  library(RTMB)

  # H0 true
  set.seed(2026)
  x = rnorm(20, 1, 1)
  y = rnorm(20, 0, 1)

  ## # H0 false
  ## set.seed(2026)
  ## x = rnorm(20, 1, 1)
  ## beta1 = 2
  ## y = rnorm(20, 0, 1) + beta1 * x

  # Start values, tailored to be appropriate in both cases :)
  beta0.start = mean(y)
  beta1.start = 1

  # We are working with a similar linear regression model
  # We assume that: Y_i ~ Normal(mu_i, 1), where i=1,...,20
  # Our H0 model is: mu_i = beta0
  # Our H1 model is: mu_i = beta0 + beta1 * x_i, where x is a vector of 20 covariate values

  # In the code below, we assume that:
  # (1) The response values are saved as an object labelled 'y'
  # (2) The covariate values are saved as an object labelled 'x'
  # (3) Appropriate starting values for optimisation for beta0 and beta1 are saved as objects labelled 'beta0.start' and 'beta1.start', respectively

  # Creating the data object
  data = list(x = x, y = y)

  # Defining the objective function (negative log-likelihood) for the H0 model
  H0.nll = function(params) {
    getAll(params, data)
    sum(dnorm(y, mean=beta0, sd=1, log=TRUE)) * (-1)
  }

  # Defining the objective function (negative log-likelihood) for the H1 model
  H1.nll = function(params) {
    getAll(params, data)
    sum(dnorm(y, beta0 + (beta1 * x), 1, log=TRUE)) * (-1)
  }

  # Creating the RTMB object for the H0 model
  H0.start = list(beta0=beta0.start)
  H0.object = MakeADFun(func=H0.nll, parameters=H0.start)
  # Fitting the H0 model
  H0.fit = nlminb(start=H0.object$par, objective=H0.object$fn, gradient=H0.object$gradient)
  # Extracting the H0 MLE
  H0.MLE = H0.fit$par

  # Creating the RTMB object for the H1 model, from which we can extract and evaluate gradients
  H1.start = list(beta0=beta0.start, beta1=beta1.start)
  H1.object = MakeADFun(func=H1.nll, parameters=H1.start)

  # Constructing a vector that shares the constraints and MLEs of the H0 model
  H0.MLE.for.H1 = list(beta0=H0.MLE["beta0"], beta1=0)
  # Evaluating the H1 score vector and information matrix at this vector (we multiply by -1 below as we are working with the negative log-likelihood)
  score.vec = H1.object$gr(H0.MLE.for.H1) * (-1)
  obs.info.mat = H1.object$he(H0.MLE.for.H1)

  # Calculating the score statistic
  score.statistic = score.vec %*% solve(obs.info.mat) %*% t(score.vec)
  # Extracting the p-value
  p.value = pchisq(score.statistic, df=1, lower=F)

}

###########################################################################################

## Code for Figure 1 :)

# H0: mu_i=0
# H1: mu_i=0 + beta_1

# Function to create y-values for plot
ll.diff.func = function(n, beta1.seq=seq(-1, 2, by=0.01), mean.y=1, which.beta.subtract=1) {
  sum.y.sq = (1 + (mean.y^2)) * n # Based on Var(Y)=1/so that Var(Y)=1 for one observation still holds :)
  ll = (((n/2) * log(2*pi)) + (1/2)*sum.y.sq - (beta1.seq)*(n)*(mean.y) + (n/2)*(beta1.seq^2)) * (-1) # Have manually confirmed this expression is correct :)
  ll.subtract = ll[beta1.seq==which.beta.subtract] # Shifting curves accordingly -- so that for scenario 1, curves intersect at peak; for scenario 2, curves intersect at theta1-tilde
  ll.diff = ll - ll.subtract
  ll.diff
}

# Scenario (1): different gradient at theta1-tilde, due to n only (not due to Var(Y) -- equivalently, not due to Y-bar or sum(Y_i^2). Shows how increasing sample size affects gradient, and therefore affects score statistic (as here, score = U^2/I)
beta1.seq = seq(-1, 2, by=0.01)
plot(beta1.seq, ll.diff.func(n=5), type="l", xlab=expression(beta[1]), ylab="log-likelihood difference")
grid()
lines(beta1.seq, ll.diff.func(n=10), col="blue")

# Scenario (2): same score at theta1-tilde --> gradient alone is not enough! :)
plot(beta1.seq, ll.diff.func(n=10, mean.y=0.5, which.beta.subtract=0), type="l", xlab=expression(beta[1]), ylab="log-likelihood difference", col='red', ylim=c(-10, 3))
lines(beta1.seq, ll.diff.func(n=5, which.beta.subtract=0))

###########################################################################################
