library(RTMB)

###########################################################################################

if (F) {
  ## Simple RTMB illustrative example for Appendix :)

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

## Code to create Figure 1: plots illustrating how gradients by themselves are not useful. Need to temper using info matrix (using score test statistics) to accurately reflect adequacy of model fit
# We will use the same models/H0/H1 as above. Will also use the same objective functions:


## Starting with well-informed ("pointy") likelihood surface, i.e. large sample size (n=50)

# (1) Generating the data, covariate values

# Using same generating values for y (H0 true, mean=0, sd=1)
set.seed(3000)
x = rnorm(50, 1, 1)
y = rnorm(50, 0, 1)

# (2) Let's get the H0 estimate for beta0

beta0.start = mean(y) # Seems sensible

# Fitting the H0 model to get the MLE for beta0
data = list(x=x, y=y)

# H0 objective function
H0.nll = function(params) {
  getAll(params, data)
  sum(dnorm(y, mean=beta0, sd=1, log=TRUE)) * (-1)
}

# RTMB magic for H0 fit
H0.start = list(beta0=beta0.start)
H0.object = MakeADFun(func=H0.nll, parameters=H0.start)
H0.fit = nlminb(start=H0.object$par, objective=H0.object$fn, gradient=H0.object$gradient)
H0.beta0 = H0.fit$par["beta0"]

# (3) Now, let's look at the H1 log-likelihood at (H0 beta0 MLE, 0) using a contour plot

# H1 objective function
H1.nll = function(params) {
  getAll(params, data)
  sum(dnorm(y, beta0 + (beta1 * x), 1, log=TRUE)) * (-1)
}

# Vectorised version of this objective function (for call to outer() below, creating contour plot): returns nll values for all pairs of (beta0.vec, beta1.vec)
H1.nll.vec = function(beta0.vec, beta1.vec) {
  nll.vals = numeric(length(beta0.vec))
  for (i in 1:length(beta0.vec)) {
    nll.vals[i] = sum(dnorm(y, beta0.vec[i] + (beta1.vec[i] * x), 1, log=TRUE)) * (-1)
  }
  nll.vals
}

# Let's consider beta0 and beta1 values in [-5,5] to begin with
# (Since true beta0 around -0.2, true beta1 is 0 -- will get good range around these points)
beta0.seq = seq(-3, 3, by=0.01)
beta1.seq = seq(-3, 3, by=0.01)
# All resulting nll values
H1.ll.vals = outer(X=beta0.seq, Y=beta1.seq, function(X, Y) {
  H1.nll.vec(X, Y) * (-1)
})
head(H1.ll.vals)

# (Sanity check)
ind = which(H1.ll.vals == max(H1.ll.vals), arr.ind=TRUE)
beta0.seq[ind[1]]
beta1.seq[ind[2]]
# So according to our grid, the H1 MLEs are: beta0=-0.43, beta1=0.25
# Is this true?
H1.start = list(beta0=beta0.start, beta1=1)
H1.object = MakeADFun(func=H1.nll, parameters=H1.start)
H1.fit = nlminb(start=H1.object$par, objective=H1.object$fn, gradient=H1.object$gradient)
H1.MLE = H1.fit$par
# Yes :)
# So would say that grid of nll values looks sensible :)

# Let's start with a contour plot
#par(pty="s")
# Seems like by default, our levels are:
pretty(range(H1.ll.vals, finite=TRUE), 10)
# For more around the peak, we set:
levels = c(-1200, -1100, -1000, -900, -800, -700, -600, -500, -400, -300, -200, -100, -80, -75, -73, -72, -71)
contour(x=beta0.seq, y=beta1.seq, z=H1.ll.vals, xlab="beta0", ylab="beta1", main="Contour of H1 log-likelihood", levels=levels, xlim=c(-1,1))
points(x=H1.MLE["beta0"], y=H1.MLE["beta1"], col="red", pch=19) # Unconstrained H1 MLE
points(x=H0.beta0, y=0, col="blue", pch=19) # Vector w H0 MLE and constraints
# Can see our "H0 for H1" vector is close to the peak, probably at a steep gradient

## Based on the theory, the score vector will be pointing in the direction of the steepest increase. It seems unlikely there would be neat 'slices' we could take along the x (beta0) or y (beta1) axes to accurately show these gradients?
# For instance -- if want to take 'slices' along x and y axis at MLE. To then plot the blue point, how do we evaluate the log-likelihood? For example, if looking at the slice along the beta1 axis -- to plot the blue point, how do we find the y coord? We'd need to plug in the H1 MLE for beta0, not the H0 MLE, so it lies along the line -- but then this isn't truly the 'blue' point anymore, the gradient wouldn't really apply...

## # Maybe instead, we plot the gradients on the contour plot
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

# Let's try using 'filled.contour' to give an idea of steepness/gradient
# Re-doing levels: extending beyond the peak to avoid a white fill at the very tippy top
levels = c(-1200, -1100, -1000, -900, -800, -700, -600, -500, -400, -300, -200, -100, -80, -75, -73, -72, -71, -70)
filled.contour(x=beta0.seq, y=beta1.seq, z=H1.ll.vals, xlab="beta0", ylab="beta1", main="Contour of H1 log-likelihood", levels=levels, xlim=c(-1,1),
               plot.axes = {
                 axis(1)
                 axis(2)
                 points(x=H1.MLE["beta0"], y=H1.MLE["beta1"], col="red", pch=19) # Unconstrained H1 MLE
                 points(x=H0.beta0, y=0, col="blue", pch=19) # Vector w H0 MLE and constraints
               })

## So have landed at -- a plain contour and/or filled contour plot to show where our "H0 for H1" point lies relative to the MLE, and what the gradient may be like near this point.

# (4) Lastly, let's get the score statistic!
score.vec = H1.object$gr(list(beta0=H0.beta0, beta1=0)) * (-1)
obs.info.mat = H1.object$he(list(beta0=H0.beta0, beta1=0))
score.statistic = score.vec %*% solve(obs.info.mat) %*% t(score.vec)
score.statistic
pchisq(score.statistic, df=1, lower=F) # p-value confirms H0 likely true, so good sanity check!



## Now, moving on to a flatter surface, smaller sample size
# Want to engineer the same score statistic, but have a completely different gradient at the "H0 for H1" point
