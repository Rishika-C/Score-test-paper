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

# H0: mu_i=0
# H1: mu_i=0 + beta_1

n.1 = 5
mean.y = 1
#sum.y.sq.1 = 2 * n.1

beta1.seq = seq(-1, 2, by=0.01)

ll.diff.func = function(n, beta1.seq=seq(-1, 2, by=0.01), mean.y=1, which.beta.subtract=1) {
  sum.y.sq = 2 * n
  ll = (((n/2) * log(2*pi)) + (1/2)*sum.y.sq - (beta1.seq)*(n)*(mean.y) + (n/2)*(beta1.seq^2)) * (-1)
  ll.subtract = ll[beta1.seq==which.beta.subtract]
  ll.diff = ll - ll.subtract
  ll.diff
}

plot(beta1.seq, ll.diff.func(n=10, mean.y=0.5, which.beta.subtract=0), type="l", xlab=expression(beta[1]), ylab="log-likelihood difference", col='red', ylim=c(-10, 3))
grid()
lines(beta1.seq, ll.diff.func(n=5, which.beta.subtract=0))

plot(beta1.seq, ll.diff.func(n=5), type="l", xlab=expression(beta[1]), ylab="log-likelihood difference")
lines(beta1.seq, ll.diff.func(n=10), col="blue")




2
###########################################################################################


## Code to create Figure 1: plots illustrating how gradients by themselves are not useful. Need to temper using info matrix (using score test statistics) to accurately reflect adequacy of model fit
# We will use the same models/H0/H1 as above (simple Appendix example). Will also use the same objective functions.

# Note that will refer to unconstrained theta1 MLE vector as 'theta1-hat', and theta1 vector evaluated at H0 MLEs and H0 constraints as 'theta1-tilde', using notation from paper

## Seems like perhaps taking two 'slices' through the log-likelihood surface could be a good way to visualise the gradient at our blue point
# However, there are no two perpendicular slices we could get that would contain both the red and blue point on the same curve.
# That is, any slices we take are likely to be misleading in what they show -- wouldn't actually be showing the discrepancy between theta1-tilde and theta1-hat
# For example, there is no way to take perpendicular slices through the blue point, such that the red point will lie on the same curve so we could visualise the discrepancy between the two, and vice versa.
# So are sticking with contour plots for now.

library(RTMB)

# Objective functions we will use
# H0 objective function
H0.nll = function(params) {
  getAll(params, data)
  sum(dnorm(y, mean=beta0, sd=1, log=TRUE)) * (-1)
}

# H1 objective function
H1.nll = function(params) {
  getAll(params, data)
  sum(dnorm(y, beta0 + (beta1 * x), 1, log=TRUE)) * (-1)
}

# Vectorised version of the H1 objective function (for call to outer() below, creating contour plot) -- returns nll values for all pairs of (beta0.vec, beta1.vec)
H1.nll.vec = function(beta0.vec, beta1.vec) {
  nll.vals = numeric(length(beta0.vec))
  for (i in 1:length(beta0.vec)) {
    nll.vals[i] = sum(dnorm(y, beta0.vec[i] + (beta1.vec[i] * x), 1, log=TRUE)) * (-1)
  }
  nll.vals
}

# Function to fit a model given the RTMB object, extract MLE for beta0 (and beta1, if applicable)
# * rtmb.obj = named RTMB object
# * mod = "H0" or "H1"
fit.info = function(rtmb.obj, mod) {
  fit = nlminb(start=rtmb.obj$par, objective=rtmb.obj$fn, gradient=rtmb.obj$gradient)
  beta0.mle = fit$par["beta0"]
  if (mod=="H1") {
    beta1.mle = fit$par["beta1"]
  } else {
    if (mod=="H0") {
      beta1.mle = NULL
    }
  }
  list(fit=fit, mle=c(beta0.mle, beta1.mle))
}

# Function that takes in H1 RTMB object, returns score statistic
# * rtmb.obj = named RTMB object
# * theta1.tilde = vector we want to evaluate score and info matrix at, should be given as a named list (e.g. list(beta0=H0.beta0, beta1=0))
# * df = df for Chi-sq approx
score.stat.func = function(rtmb.obj, theta1.tilde, df=1) {
  score.vec = rtmb.obj$gr(theta1.tilde) * (-1)
  obs.info.mat = rtmb.obj$he(theta1.tilde)
  score.statistic = score.vec %*% solve(obs.info.mat) %*% t(score.vec)
  p.value = pchisq(score.statistic, df=1, lower=F)
  list(score.vec=score.vec, obs.info.mat=obs.info.mat, score.statistic=score.statistic, p.value=p.value)
}

# Function that creates grid of values for contour plot for H1 log-likelihood
# * beta0.seq = sequence of beta0 values to consider
# * beta1.seq = sequence of beta1 values to consider
contour.grid.vals = function(beta0.seq, beta1.seq) {
  H1.ll.vals = outer(X=beta0.seq, Y=beta1.seq, function(X, Y) {
    H1.nll.vec(X, Y) * (-1)
  })
  H1.ll.vals
}

# Function to create contour plots
# * x.vals, y.vals, z.vals = x-, y- and z-values for contour plot
# * levels = levels for contour lines
# * xlim, ylim = xlim and ylim for plot
# * filled = if T, create filled contour plot using filled.contour(); else, create an unfilled contour plot using contour()
contour.plot.func = function(x.vals, y.vals, z.vals, levels, xlim, ylim, filled) {
  # Setting aspect ratio to 1
  par(pty="s")

  if (filled) {
    # Filled contour plot
    filled.contour(x=x.vals, y=y.vals, z=z.vals, xlab=expression(beta[0]), ylab=expression(beta[1]), main=expression(Contour~of~H[1]~"log-likelihood"), levels=levels, xlim=xlim, ylim=ylim,
                   plot.axes = {
                     axis(1)
                     axis(2)
                     points(x=H1.MLE["beta0"], y=H1.MLE["beta1"], col="red", pch=19) # theta1-hat point
                     points(x=H0.beta0, y=0, col="blue", pch=19) # theta1-tilde
                   })
  } else {
    # Unfilled contour plot
    contour(x=x.vals, y=y.vals, z=z.vals, xlab=expression(beta[0]), ylab=expression(beta[1]), main=expression(Contour~of~H[1]~"log-likelihood"), levels=levels, drawlabels=F, xlim=xlim, ylim=ylim)
    # theta1-hat point
    points(x=H1.MLE["beta0"], y=H1.MLE["beta1"], col="red", pch=19)
    # theta1-tilde
    points(x=H0.beta0, y=0, col="blue", pch=19)
  }
}



if (F) {
  ## Starting with well-informed ("pointy") likelihood surface, i.e. large sample size (n=50)

  # (1) Generating the data, covariate values
  # Using same generating values for y (H0 true, mean=0, sd=1)
  set.seed(3000)
  x = rnorm(100, 1, 1)
  y = rnorm(100, 0, 1)
  beta0.start = mean(y) + 1 # Seems sensible
  beta1.start = 1 # Why not
  data = list(x=x, y=y) # Data object

  # (2) Let's get the H0 estimate for beta0
  H0.object = MakeADFun(func=H0.nll, parameters=list(beta0=beta0.start))
  H0.model = fit.info(rtmb.obj=H0.object, mod="H0")
  H0.beta0 = H0.model$mle["beta0"]

  # (3) Let's get the score statistic!
  H1.object = MakeADFun(func=H1.nll, parameters=list(beta0=beta0.start, beta1=beta1.start))
  score.stat.info = score.stat.func(rtmb.obj=H1.object, theta1.tilde=list(beta0=H0.beta0, beta1=0), df=1)
  score.stat.info # Looks good, we don't reject H0, as we would expect!
  # * score statistic = 0.5662133, p-value = 0.4517676

  # (4) Now, let's look at the H1 log-likelihood at theta1-tilde using a contour plot

  # Unconstrained H1 MLE for plotting: theta1-hat
  H1.fit = fit.info(rtmb.obj=H1.object, mod="H1")
  H1.MLE = H1.fit$mle # For plotting later

  # Generating the grid of values for our contour plot
  # Let's consider beta0 and beta1 values in [-3,3] to begin with. (Since true beta0 around -0.2, true beta1 is 0 -- will get good range around these points)
  beta0.seq = seq(-3, 3, by=0.01)
  beta1.seq = seq(-3, 3, by=0.01)
  H1.ll.vals = contour.grid.vals(beta0.seq, beta1.seq)

  # Let's start with a contour plot
  # Seems like by default, our levels are: pretty(range(H1.ll.vals, finite=TRUE), 10) For more contours around the peak, we set:
  # levels = c(-1200, -1100, -1000, -900, -800, -700, -600, -500, -400, -300, -200, -100, -80, -75, -73, -72, -71, -70.5) # for n=50
  levels = c(-2400, -2200, -2000, -1800, -1600, -1400, -1200, -1000, -800, -600, -400, -200, -175, -150, -145, -140, -139, -138.5)
  contour.plot.func(x.vals=beta0.seq, y.vals=beta1.seq, z.vals=H1.ll.vals, levels=levels, xlim=c(-2,2), ylim=c(-2,2), filled="FALSE")

  # Let's try using 'filled.contour' to give an idea of steepness/gradient
  # Re-doing levels: extending beyond the peak to avoid a white fill at the very tippy top
  # levels = c(-1200, -1100, -1000, -900, -800, -700, -600, -500, -400, -300, -200, -100, -80, -75, -73, -72, -71, -70.5, -70) # for n=50
  levels = c(-2400, -2200, -2000, -1800, -1600, -1400, -1200, -1000, -800, -600, -400, -200, -175, -150, -145, -140, -139, -138.5, -138)
  contour.plot.func(x.vals=beta0.seq, y.vals=beta1.seq, z.vals=H1.ll.vals, levels=levels, xlim=c(-3,3), ylim=c(-3,3), filled="TRUE")

  # So have landed at -- a plain contour and/or filled contour plot to show where our "H0 for H1" point lies relative to the MLE, and what the gradient may be like near this point.
  # *** Does this sound okay??
}

# *** HOW should we engineer the score statistic to be the same for another example?
# Currently finding analytic form  of score statistic.
# Good to have consistency in distributions we simulate X and Y from in both examples??
# Concern is -- if tweak data for same score statistic, doesn't really come from chosen distributions anymore.
# Perhaps sufficient to have -- high gradient, score stat; low gradient, larger score stat??


if (F) {
  ## Some checks/tests from contour plot for 'pointy' likelihood surface:

  ## # Sanity check for grid of values, H1.ll.vals
  ## ind = which(H1.ll.vals == max(H1.ll.vals), arr.ind=TRUE)
  ## beta0.seq[ind[1]]
  ## beta1.seq[ind[2]]
  ## # So according to our grid, the H1 MLEs are: beta0=-0.43, beta1=0.25
  ## # Is this true?
  ## H1.start = list(beta0=beta0.start, beta1=1)
  ## H1.object = MakeADFun(func=H1.nll, parameters=H1.start)
  ## H1.fit = nlminb(start=H1.object$par, objective=H1.object$fn, gradient=H1.object$gradient)
  ## H1.MLE = H1.fit$par
  ## # Yes :)
  ## # So would say that grid of nll values looks sensible :)

  ## # Looking at plotting the gradients on the contour plot
  ## # Let's find the H1 score vector at (H0.beta0, 0) -- what we use to calculate the score statistic
  ## score.vec = H1.object$gr(list(beta0=H0.beta0, beta1=0)) * (-1)
  ## score.vec
  ## # It is basically 0 wrt beta0, which is interesting... Let's try and plot it anyways, see what happens
  ## arrows(x0=H0.beta0, y0=0, x1=H0.beta0 + score.vec[1], y1=0 + (score.vec[2]*0.2), col='blue')
  ## # Looks a bit funny, but seems correct.
  ## # I believe that based on the log-lik calculations, when beta1=0, the first derivative wrt beta0 should be:
  ## sum(y - H0.beta0)
  ## # And the first derivative wrt beta1 should be:
  ## sum(x*(y - H0.beta0))
  ## # So score is correct. Just isn't very 'pretty' on the plot. Turns out the score vector isn't at a diagonal!
  ## # And anyways, an arrow for the gradient doesn't really give an idea of the steepness, just the direction of the score vector.

}


if (F) {
  ## Now, moving on to a flatter surface, smaller sample size
  # Want to engineer the same score statistic, but have a completely different gradient at theta1-tilde
  # Previous score statistic = 0.1131113, previous p-value = 0.7366294

  # (1) Generating a data set so we get the same score statistic

  # For our example, with sd=1 known, we have that the score statistic is given by:
  # S = A^2/B, where A = sum(x_i * (Y_i - mean(Y))), B = sum(x_i^2)

  # For both data sets, we want a score statistic, S=0.5662133
  S = 0.1131113
  # So for this data set, to get the same score statistic, we can:
  # - Simulate some covariates, find B
  # - Then solve to find what A should be -- from there, we can generate Y values


  set.seed(3000)
  x = rnorm(100, 1, 1)
  y = rnorm(100, 0, 1)
  (sum(x * (y-mean(y))))^2 / (sum(x^2))



  # Let's work with a sample size of 20
  # Covariate values
  set.seed(3000)
  x = rnorm(100, 1, 1)
  B = sum(x^2)
  B # Our B value

  # So, we want A to be:
  A = sqrt(S * B) * (-1)
  A # We'll just say we want it to be the negative square root -- shouldn't matter, should get score statistic either way?

  # So now let's generate the response values corresponding to this value of A:
  x
  y = rnorm(100, 0, 1)
  sum(x * (y-mean(y)))
















  # With some trial and error, if we set the mean of Y to 0.43, we get something close to A with the following:
  first.y = rnorm(19, 1, 1)
  sum(x[-20] * (first.y-0.43))

  # (1) Generating a data set using same settings as above, smaller sample size, and same values for (a) and (b)
  # Will use a sample size of 20 -- randomly generate the first 19 values, then set the 20th x and y value so our values of a and b match!
  set.seed(3000)
  x = rnorm(19, 1, 1)
  y = rnorm(19, 0, 1)
  # The last value of x we need, based on the value of b:
  last.x = prev.b - sum(x^2)
  last.x
  x = c(x, last.x)
  x
  # The last value of y we need
  prev.a; sum(x[-20] * (y - mean(y))) # Where we're at
  last.y = (20 * mean(prev.y)) - sum(y) # Setting y so the mean is the same
  last.y
  y = c(y, last.y)
  y
  mean(y); mean(prev.y)
  prev.a; sum(x * (y - mean(y)))

  # (1) Generating the data, covariate values. Using the same settings as above, just a smaller sample size (so H0 is true, mean=0, sd=1)
  set.seed(3000)
  x = rnorm(15, 1, 1)
  y = rnorm(15, 0, 1)
  beta0.start = mean(y)
  beta1.start = 1
  data = list(x=x, y=y)

  # (2) H0 estimate for beta0
  H0.object = MakeADFun(func=H0.nll, parameters=list(beta0=beta0.start))
  H0.model = fit.info(rtmb.obj=H0.object, mod="H0")
  H0.beta0 = H0.model$mle["beta0"]

  # (3) Score statistic
  H1.object = MakeADFun(func=H1.nll, parameters=list(beta0=beta0.start, beta1=beta1.start))
  score.stat.info = score.stat.func(rtmb.obj=H1.object, theta1.tilde=list(beta0=H0.beta0, beta1=0), df=1)
  score.stat.info

  # (4) Contour plot

}
