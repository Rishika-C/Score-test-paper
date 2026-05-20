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
  beta0.start = mean(y)+1
  beta1.start = 1

  # We are working with a similar linear regression model
  # We assume that: Y_i ~ Normal(mu_i, 1), where i=1,...,20
  # Our H0 model is: mu_i = beta0
  # Our H1 model is: mu_i = beta0 + beta1 * x_i, where x is a vector of 20 covariate values

  # In the code below, we assume that:
  # (1) The response values are saved as an object labelled 'y'
  # (2) The covariate values are saved as an object labelled 'x'
  # (3) Appropriate starting values for optimisation for beta0 and beta1 are saved as objects labelled 'beta0.start' and 'beta1.start', respectively

  # Simple code for paper, based on Ben's example:

  # Negative log-likelihood function
  nll <- function(pars) -sum(dnorm(y, pars$beta0 + pars$beta1*x, 1, log = TRUE))

  # RTMB object for H0 and H1
  obj.null <- MakeADFun(nll, list(beta0 = beta0.start, beta1 = 0), map = list(beta1 = factor(NA)))
  obj.alt <- MakeADFun(nll, list(beta0 = beta0.start, beta1 = beta1.start))

  # Fitting H0 model
  fit.null <- nlminb(obj.null$par, obj.null$fn, obj.null$gr)

  # theta1-tilde vector
  pars.null <- list(beta0 = fit.null$par["beta0"], beta1 = 0)

  # H1 score vector at theta1-tilde
  score.null <- -obj.alt$gr(pars.null)
  # H1 obs info matrix at theta1-tilde
  info.null <- obj.alt$he(pars.null)

  # Score statistic and p-vaue
  S <- score.null %*% solve(info.null) %*% t(score.null)
  p.val <- 1 - pchisq(S, df=1)

  ## # Creating the data object
  ## data = list(x = x, y = y)

  ## # Defining the objective function (negative log-likelihood) for the H0 model
  ## H0.nll = function(params) {
  ##   getAll(params, data)
  ##   sum(dnorm(y, mean=beta0, sd=1, log=TRUE)) * (-1)
  ## }

  ## # Defining the objective function (negative log-likelihood) for the H1 model
  ## H1.nll = function(params) {
  ##   getAll(params, data)
  ##   sum(dnorm(y, beta0 + (beta1 * x), 1, log=TRUE)) * (-1)
  ## }

  ## # Creating the RTMB object for the H0 model
  ## H0.start = list(beta0=beta0.start)
  ## H0.object = MakeADFun(func=H0.nll, parameters=H0.start)
  ## # Fitting the H0 model
  ## H0.fit = nlminb(start=H0.object$par, objective=H0.object$fn, gradient=H0.object$gradient)
  ## # Extracting the H0 MLE
  ## H0.MLE = H0.fit$par

  ## # Creating the RTMB object for the H1 model, from which we can extract and evaluate gradients
  ## H1.start = list(beta0=beta0.start, beta1=beta1.start)
  ## H1.object = MakeADFun(func=H1.nll, parameters=H1.start)

  ## # Constructing a vector that shares the constraints and MLEs of the H0 model
  ## H0.MLE.for.H1 = list(beta0=H0.MLE["beta0"], beta1=0)
  ## # Evaluating the H1 score vector and information matrix at this vector (we multiply by -1 below as we are working with the negative log-likelihood)
  ## score.vec = H1.object$gr(H0.MLE.for.H1) * (-1)
  ## obs.info.mat = H1.object$he(H0.MLE.for.H1)

  ## # Calculating the score statistic
  ## score.statistic = score.vec %*% solve(obs.info.mat) %*% t(score.vec)
  ## # Extracting the p-value
  ## p.value = pchisq(score.statistic, df=1, lower=F)

}

###########################################################################################

## Code for Figure 1 :)

# H0: mu_i=0
# H1: mu_i=0 + beta_1

# Function to evaluate log-lik. beta1 may be a sequence or scalar
ll.func = function(beta1, n, mean.y) {
  sum.y.sq = (1 + (mean.y^2)) * n
  (((n/2) * log(2*pi)) + (1/2)*sum.y.sq - (beta1)*(n)*(mean.y) + (n/2)*(beta1^2)) * (-1)
}

# Function to calculate LRT statistic comparing maximised (unconstrained) H1 model with H0 model
lrt.func = function(n, mean.y) {
  h1.ll = ll.func(beta1=mean.y, n=n, mean.y=mean.y)
  h0.ll = ll.func(beta1=0, n=n, mean.y=mean.y)
  lrt.stat = 2 * (h1.ll - h0.ll)
  lrt.stat
}

# Function to create y-values for plot: shifted log-lik values
ll.diff.func = function(n, beta1.seq=seq(-2, 2, by=0.01), mean.y=1, which.beta.subtract=1) {
  #browser()
  sum.y.sq = (1 + (mean.y^2)) * n # Based on Var(Y)=1/so that Var(Y)=1 for one observation still holds :)
  # Log-likelihood values for all beta1.seq -- have manually confirmed this expression is correct :)
  #ll = (((n/2) * log(2*pi)) + (1/2)*sum.y.sq - (beta1.seq)*(n)*(mean.y) + (n/2)*(beta1.seq^2)) * (-1)
  ll = ll.func(beta1=beta1.seq, n=n, mean.y=mean.y)
  # Shifting curves accordingly -- so that for scenario 1, curves intersect at theta1-tilde
  ll.subtract = ll[beta1.seq==which.beta.subtract]
  # Log-lik values
  ll.diff = ll - ll.subtract
  ll.diff
}

# ------
if (F) {

  # Scenario (1): same score at theta1-tilde --> gradient alone is not enough! :)
  cairo_pdf("Figures/Fig1a.pdf")

  beta1.seq = seq(-2, 2, by=0.01) # x seq for plotting

  # Distance bw y-axis label and y-axis and margins :)
  par(mgp=c(3.2, 1, 0), mar=c(5.5, 5.5, 4.1, 2.1))

  # Parameter settings
  n.1 = 5
  n.2 = 20
  mean.y.1 = 1
  mean.y.2 = 0.25

  # We'll use the information that MLE(theta) = mean(y) for each sample below. We'll also use info that theta1-tilde=0, and at theta-tilde, the y-value of both peaks is 0, by construction  (as we use which.beta.substract=0 below)

  # Blue curve: larger discrepancy between theta-tilde and theta1-hat
  # n=5, mean.y=1, S=5
  plot(beta1.seq, ll.diff.func(n=n.1, mean.y=mean.y.1, which.beta.subtract=0), type="l", xlab="", ylab=expression("\u2113"(bold(theta)[1]) - "\u2113"(hat(bold(theta))[1])), col='blue', lwd=3, cex.lab=1.6, xaxt="n", xaxs="i", ylim=c(-6,4), cex.axis=1.5)# xlim=c(-1.5, 2), ylim=c(-10, 3)

  # x-axis
  axis(side=1, at=c(0, mean.y.2, mean.y.1), labels=c(expression(tilde(bold(theta))[1]), expression(hat(bold(theta))[1*","*1]), expression(hat(bold(theta))[1*","*2])), cex.axis=1.5, padj=0.3)
  title(xlab=expression(bold(theta)[1]), mgp=c(3, 1, 0), cex.lab=1.6)

  # Grid lines
  grid()

  # Orange curve: smaller discrepancy between theta1-tilde and theta1-hat
  # n=10, mean.y=0.5, S=2.5
  lines(beta1.seq, ll.diff.func(n=n.2, mean.y=mean.y.2, which.beta.subtract=0), col="orange", lwd=3)

  # Gradient lines. Since they'll overlap, we'll do the first one solid, the second one dotted, and will do both thicker (so can see both overlaid)
  # This time, we know that the y-value at our point of interest (beta1-tilde, i.e. beta1=0) is 0 due to the way we have shifted the parabolas
  # Instead of going 1 either unit of theta1-hat, will go 0.5 units either side. Hence the 0.5's below :)
  lines(x=c(-0.5, 0.5), y=c(0 - 0.5*n.1*mean.y.1, 0 + 0.5*n.1*mean.y.1), col='blue', lty=1, lwd=4)
  lines(x=c(-0.5, 0.5), y=c(0 - 0.5*n.2*mean.y.2, 0 + 0.5*n.2*mean.y.2), col='orange', lty=3, lwd=4)

  # Dashed lines at theta1-tilde and theta1-hat (both of them)
  # theta1-tilde
  lines(x=c(0, 0),  y=c(-12, 0), col='darkgreen', lty=2, lwd=2)
  # theta1-hat for blue curve
  ll.y.n.1 = ll.diff.func(n=n.1, mean.y=mean.y.1, which.beta.subtract=0)[which(beta1.seq==mean.y.1)]
  lines(x=c(mean.y.1, mean.y.1), y=c(-20, ll.y.n.1), col='darkgreen', lty=2, lwd=2)
  # theta1-hat for orange curve
  ll.y.n.2 = ll.diff.func(n=n.2, mean.y=mean.y.2, which.beta.subtract=0)[which(beta1.seq==mean.y.2)]
  lines(x=c(mean.y.2, mean.y.2), y=c(-20, ll.y.n.2), col='darkgreen', lty=2, lwd=2)

  # Points at ll(theta1-tilde), ll(theta1-hat) for both curves. For latter, choose x-values=mean(y)=MLE for theta
  points(x=0, y=0, col="purple", pch=19, cex=1.7)
  points(x=mean.y.1, y=ll.y.n.1, col="purple", pch=19, cex=1.7)
  points(x=mean.y.2, y=ll.y.n.2, col="purple", pch=19, cex=1.7)

  # For our first settings, score test statistic
  s.1 = n.1 * mean.y.1 * mean.y.1
  # For second settings
  s.2 = n.2 * mean.y.2 * mean.y.2
  s.1; s.2

  # Score test p-values
  p.1 = pchisq(s.1, df=1, lower=F)
  p.2 = pchisq(s.2, df=1, lower=F)
  p.1; p.2

  # Confirming LRT stats are the same
  lrt.func(n=n.1, mean=mean.y.1)
  lrt.func(n=n.2, mean=mean.y.2)

  # Legend -- manually writing out score/LRT stats here out of necessity w expression()
  legend("topleft", legend=c(expression(S~"="~LRT~"="~5~", "~p~"="~0.03),
                             expression(S~"="~LRT~"="~1.25~", "~p~"="~0.26)), lwd=2, col=c("blue", "orange"),
         bg="white", cex=1.3)

  dev.off()

}
# ------

# Code to highlight equivalence of LRT and score statistic under a quadratic log-likelihood

# Log-likelihood function
# Assume that mu is provided as a vector
gamma.ll.known.var = function(mu, y, known.var) {

  ll = length(mu) # Initialising log-lik vec

  for (i in 1:length(mu)) {
    # Mu (mean) is the only parameter we estimate here
    # So, calculating shape and rate from known variance and unknown mu
    shape = mu[i]^2/known.var
    rate = mu[i]/known.var

    # Log-lik
    ll[i] = sum(dgamma(y, shape=shape, rate=rate, log=TRUE))
  }

  ll
}

# Negative log-likelihood function for optimisation :) -- so is in usual RTMB nll func format, don't need to vectorise
# Are using 'gamma.sample', 'gamma.variance' as are objects defined in the environment
gamma.nll.known.var = function(params) {
  -sum(dgamma(gamma.sample, shape=(params$mu^2)/gamma.variance, rate=params$mu/gamma.variance, log=TRUE))
}

# A 'diff' func like ll.diff.func, so we can have (value on y-axis)=0 at beta1=0
gamma.ll.diff.func = function(mu, y, known.var, which.mu.subtract=0) {
  #browser()
  # Finding log-lik values
  ll = gamma.ll.known.var(mu=mu, y=y, known.var=known.var)
  # Shifting curve
  ll.subtract = ll[mu==which.mu.subtract]
  ll.diff = ll - ll.subtract
  ll.diff
}

if (F) {
  # Scenario (2): curves highlighting equivalence between LRT and score test when log-likelihod is quadratic

  cairo_pdf("Figures/Fig1b.pdf")

  # Distance bw y-axis label and y-axis and margins :)
  par(mgp=c(3.2, 1, 0), mar=c(5.5, 5.5, 4.1, 2.1))

  # Let's draw a non-quadratic log-likelihood -- specifically the log-likelihood for a Gamma distribution with known variance
  # Things aren't as simple as the Normal case we've been working with so far -- for example, the log-likelihood can't just be expressed in terms of mean(y), sum of squared y's, etc... We need to actually simulate a data set
  set.seed(2026)
  gamma.true.mean = 10 # For simulation
  gamma.variance = 10 # The known variance
  # Note that we used some trial and error to choose a mean and variance that produced a curve that was non-quadratic, but not too out there in shape
  n.gamma = 50 # Sample size

  # Generating a sample
  # Note that when we have the mean and var, shape = mean^2/var, rate = mean/var
  gamma.sample = rgamma(n.gamma, shape=gamma.true.mean^2/gamma.variance, rate=gamma.true.mean/gamma.variance)

  # We are choosing thetaOneTilde=2 so things are nice and easy to see on our plot -- is okay, is an arbitrary decision :) So as we see below, we shift both curves so that we can see them intersect at thetaOneTilde

  # Plotting the log-likelihood
  mu.seq = seq(0.1, 15, by=0.01)
  plot(mu.seq, gamma.ll.diff.func(mu.seq, y=gamma.sample, known.var=gamma.variance, which.mu.subtract=2), type='l', lwd=3, col='black', ylim=c(-150, 150), xlim=c(0, 14), ylab=expression("\u2113"(bold(theta)[1]) - "\u2113"(tilde(bold(theta)[1]))), cex.lab=1.6, cex.axis=1.5, xlab=expression(bold(theta)[1]), yaxt="n", xaxt="n")

  # Finding the MLE of the sample :)
  obj.gamma = MakeADFun(gamma.nll.known.var, list(mu=5))
  gamma.fit = nlminb(obj.gamma$par, obj.gamma$fn, obj.gamma$gr)
  gamma.mle = gamma.fit$par["mu"]

  grid()

  # Adding a nice parabola log-likelihood :)
  n.normal = 30
  mean.normal = 3.5 # We know this is the MLE!
  beta1.seq = seq(-3, 8, by=0.01)
  lines(beta1.seq, ll.diff.func(beta1=beta1.seq, n=n.normal, mean.y=mean.normal, which.beta.subtract=2), type="l", col='red', lwd=3)

  # x-axis :)
  axis(side=1, at=c(2, mean.normal, gamma.mle), labels=c(expression(tilde(bold(theta))[1]), expression(hat(bold(theta))[1*","*1]), expression(hat(bold(theta))[1*","*2])), cex.axis=1.5, padj=0.3)

  # y-values for both MLEs -- we'll need these below :)
  theta11.mle.yval = ll.diff.func(beta1=beta1.seq, n=n.normal, mean.y=mean.normal, which.beta.subtract=2)[beta1.seq==3.5]
  theta12.mle.yval = gamma.ll.diff.func(mu.seq, y=gamma.sample, known.var=gamma.variance, which.mu.subtract=2)[mu.seq == round(gamma.mle, 1)]

  ## # y-axis :) -- no y-axis for now as labels jumble together, and I think it would be okay without these labels? Would be kind of confusing anyways, as the y-label is a difference and we're not acknowledging it in these labels
  ## axis(side=2, at=c(0, theta11.mle.yval, theta12.mle.yval), labels=c(expression("\u2113"(tilde(bold(theta))[1])), expression("\u2113"(hat(bold(theta))[1*","*1])), expression("\u2113"(hat(bold(theta))[1*","*2]))), cex.axis=1, padj=0.3)

  # Horizontal dashed lines representing LRT distances :)
  abline(h=0, lty=2, lwd=2, col='darkgreen')
  abline(h=theta11.mle.yval, lty=2, lwd=2, col='darkgreen')
  abline(h=theta12.mle.yval, lty=2, lwd=2, col='darkgreen')

  # Point and vertical line at thetaOneTilde=2
  # Given how we have shifted the curves, we know that y-value=0 at thetaOne=2
  lines(x=c(2, 2), y=c(-200, 0), col='darkgreen', lty=2, lwd=1)
  points(2, 0, col='purple', pch=19, cex=1.7)
  # Point and vertical line at thetaOneMLE{1,1} -- i.e. at MLE for parabola :)
  lines(x=c(mean.normal, mean.normal), y=c(-200, theta11.mle.yval), col='darkgreen', lty=2, lwd=1)
  points(mean.normal, theta11.mle.yval, col='purple', pch=19, cex=1.7)
  # And the same for thetaOneMLE{1,2} -- MLE for the Gamma log-lik :)
  lines(x=c(gamma.mle, gamma.mle), y=c(-200, theta12.mle.yval), col='darkgreen', lty=2, lwd=1)
  points(gamma.mle, theta12.mle.yval, col='purple', pch=19, cex=1.7)

  # And arrows for LRT distances :)
  arrows(x0=6, y0=0, x1=6, y1=theta11.mle.yval, length=0.15, col='red', lwd=2, code=3)
  arrows(x0=11, y0=0, x1=11, y1=theta12.mle.yval, length=0.15, col='black', lwd=2, code=3)
  # And text indicating equivalence/non-equivalence to score test statistics :)
  text(x=7.5, y=20, labels="LRT = S", col='red', font=4)
  text(x=12.3, y=52, labels=expression(bolditalic("LRT"!="S")), col='black')


  # Then, add horizontal dashed lines with distances for LRTs :)

  dev.off()

  ## # ** Ultimately: will not proceed with visualising the second-order Taylor approximation, as we had the slightly wrong end of the stick...
  ## # It is ONLY when the log-likelihood is quadratic that the LRT statistic is represented by a second-order Taylor series expansion.
  ## # Otherwise, if we use this expansion, all we get is an approximation to the LRT statistic
  ## # And as we see below, when our log-likelihood is non-quadratic and thetaOneTilde is far from thetaOneMLE, this approximation can be quite poor...
  ## # Note that we used copilot to compute the second order Taylor series expansion below, given the sample and known variance, and assuming that mu=1.5 is the thetaOneTilde value we are working with
  ## # Log-lik value at 1.5
  ## ll.1.5 = gamma.ll.known.var(mu.seq, y=gamma.sample, known.var=gamma.variance)[141] # Got 141 from visual inspection of mu.seq
  ## # Plotting the Taylor series approximation
  ## lines(mu.seq, ll.1.5 + 42.57*(mu.seq-1.5) - 83.618*(mu.seq-1.5)^2, type="l", lty=5, col='red', lwd=3)

}

###########################################################################################
