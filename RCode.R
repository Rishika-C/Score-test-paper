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

  # ** Note that we have updated the plot, so all theta1's become alpha's -- so any references to theta1 below should be references to alpha (as for our model, theta1 = alpha). Have left theta1 in comments; code should now refer to 'alpha'
  # And note that therefore, theta1-tilde = alpha-tilde = 0 -- thus, in the code, theta1-tilde is simply replaced by 0 :)

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
  plot(beta1.seq, ll.diff.func(n=n.1, mean.y=mean.y.1, which.beta.subtract=0), type="l", xlab="", ylab=expression("\u2113"("\u03B1") - "\u2113"(0)), col='blue', lwd=3, cex.lab=1.6, xaxs="i", ylim=c(-6,4), cex.axis=1.5, xaxt="n", las=2)# xlim=c(-1.5, 2), ylim=c(-10, 3)

  # x-axis
  axis(side=1, at=c(0, mean.y.2, mean.y.1), labels=c(expression(0), expression(hat("\u03B1")[1]), expression(hat("\u03B1")[2])), cex.axis=1.5, padj=0.3)
  title(xlab=expression("\u03B1"), mgp=c(3, 1, 0), cex.lab=1.6, adj=1)

  # Grid lines
  #grid()

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
  lines(x=c(mean.y.1, mean.y.1), y=c(-20, ll.y.n.1), col='seagreen', lty=2, lwd=2)
  # theta1-hat for orange curve
  ll.y.n.2 = ll.diff.func(n=n.2, mean.y=mean.y.2, which.beta.subtract=0)[which(beta1.seq==mean.y.2)]
  lines(x=c(mean.y.2, mean.y.2), y=c(-20, ll.y.n.2), col='seagreen', lty=2, lwd=2)

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

  # Adding text to indicate this is plot (a)
  text(x=1.8, y=3.8, labels="(a)", cex=2, col='black')

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
    scale = known.var/mu[i]

    # Log-lik
    ll[i] = sum(dgamma(y, shape=shape, scale=scale, log=TRUE))
  }

  ll
}

# Negative log-likelihood function for optimisation :) -- so is in usual RTMB nll func format, don't need to vectorise
# Are using 'gamma.sample', 'gamma.variance' as are objects defined in the environment
gamma.nll.known.var = function(params) {
  -sum(dgamma(gamma.sample, shape=(params$mu^2)/gamma.variance, scale=gamma.variance/params$mu, log=TRUE))
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
  # Note: below, will see '+90' for shifting up y-axis -- part of change from unshifted curves, to curves shifted so that y=0 at theta1-tilde (so are just adjusting old settings accordingly) :)

  # Scenario (1): visualising equivalence between LRT and S when quadratic log-likelihood
  cairo_pdf("Figures/Fig1b.pdf")

  # Distance bw y-axis label and y-axis and margins :)
  par(mgp=c(3.2, 1, 0), mar=c(5.5, 5.5, 4.1, 2.1))

  # Colours
  col.1 = "darkblue"
  col.2 = "deeppink"
  line.col="seagreen"

  # Let's draw a non-quadratic log-likelihood -- specifically the log-likelihood for a Gamma distribution with known variance
  # Things aren't as simple as the Normal case we've been working with so far -- for example, the log-likelihood can't just be expressed in terms of mean(y), sum of squared y's, etc... We need to actually simulate a data set
  set.seed(2026)
  gamma.true.mean = 8
  gamma.variance = 8
  n.gamma = 30

  gamma.sample = rgamma(n.gamma, shape=gamma.true.mean^2/gamma.variance, rate=gamma.true.mean/gamma.variance)

  # According to Copilot, for the asymptotics to hold (i.e. for LRT \sim S), we want theta1-tilde only about 1-1.5 standard errors away from theta1-hat
  # (standard error refers to standard deviation of distribution of sample mean under the null model -- as we assume known variance, note that the standard deviation of the sample mean is the same under the null and alternative model!)
  # MLE
  obj.gamma = MakeADFun(gamma.nll.known.var, list(mu=8))
  gamma.fit = nlminb(obj.gamma$par, obj.gamma$fn, obj.gamma$gr)
  gamma.mle = gamma.fit$par["mu"]
  gamma.mle
  # obj.gamma$gr(gamma.mle) # MLE looks correct -- gradient is basically 0 :)
  se = sqrt(gamma.variance/n.gamma)
  se * 1.5
  # Therefore, we will give theta1-tilde of:
  theta1.tilde = 5

  mu.seq = seq(0.1, 15, by=0.01)
  # Log-likelihood, which is a function of the mean (i.e. theta=mean)
  plot(mu.seq, gamma.ll.diff.func(mu=mu.seq, y=gamma.sample, known.var=gamma.variance, which.mu.subtract=theta1.tilde), type='l', lwd=3, col=col.1, ylab=expression("\u2113"(bold(theta)[1]) - "\u2113"(tilde(bold(theta))[1])), cex.lab=1.6, cex.axis=1.5, xlab="", xaxs="i", xaxt="n", yaxt="n", ylim=c(-175+90, -50+90))
  #grid()

  # x-axis
  axis(side=1, at=c(theta1.tilde, gamma.mle), labels=c(expression(bold(tilde(theta))[1]), expression(bold(hat(theta)[1]))), cex.axis=1.5, padj=0.3)
  title(xlab=expression(bold(theta)[1]), mgp=c(3, 1, 0), cex.lab=1.6, adj=1)

  # Plotting the second-order Taylor series expansion about theta1.tilde
  # Some calculations pulled from Copilot to calculate this expansion (can confirm later):
  sigma2 <- gamma.variance        # known variance
  mu0 <- theta1.tilde             # expansion point
  k0 <- mu0^2 / sigma2            # shape at mu0
  x <- gamma.sample
  n <- length(x)
  S <- sum(x)
  T <- sum(log(x))
  # A(mu0) -- from Copilot formula
  A0 <- T -  n * log(sigma2 / mu0) - n * digamma(k0)
  # U(mu0) = l'(mu0) -- first deriv for expansion
  U0 <- (2 * mu0 / sigma2) * A0 - (S - n * mu0) / sigma2
  # U'(mu0) = l''(mu0) -- second deriv
  U0prime <- (2 / sigma2) * A0 + ((3 * n) / sigma2) - ((4 * n * mu0^2) / sigma2^2) * trigamma(k0)

  # Log-lik at theta1-tilde
  ll.theta1.tilde = 0 # by construction, using gamma.ll.diff.func()
  # gamma.ll.diff.func(mu.seq, y=gamma.sample, known.var=gamma.variance, which.mu.subtrac)[which(mu.seq==theta1.tilde)]

  # Second-order Taylor series expansion, calculated using Copilot (based on the U0 and U0prime values above)
  # Non-shifted Taylor series expansion
  taylor.expansion = ll.theta1.tilde + U0 * (mu.seq - theta1.tilde) + (U0prime/2) * (mu.seq - theta1.tilde)^2
  # Shifting so is 0 at theta1.tilde
  taylor.subtract = taylor.expansion[mu.seq==theta1.tilde]
  taylor.expansion.shift = taylor.expansion - taylor.subtract
  lines(mu.seq, taylor.expansion.shift, type="l", lty=5, col=col.2, lwd=3)

  # Plotting theta1-tilde and theta1-hat on graph
  ll.gamma.mle = gamma.ll.diff.func(mu=mu.seq, y=gamma.sample, known.var=gamma.variance, which.mu.subtract=theta1.tilde)[which(mu.seq==round(gamma.mle))]
  lines(x=c(gamma.mle, gamma.mle), y=c(-300, ll.gamma.mle), col=line.col, lty=2)
  lines(x=c(theta1.tilde, theta1.tilde), y=c(-300,ll.theta1.tilde) , col=line.col, lty=2)
  # Horizontal lines to define LRT/2 and S/2
  abline(h=ll.theta1.tilde, lty=2, col=line.col, lwd=2)
  abline(h=ll.gamma.mle, lty=2, col=line.col, lwd=2)
  # Based on the equation of a parabola, the peak of our second-order Taylor series expansion should be at:
  taylor.mle.x = (-1) * U0/(2 * (U0prime/2)) + 5
  # Plugging this value into the formula for our parabola
  yval.taylor.mle = ll.theta1.tilde + U0 * (taylor.mle.x - theta1.tilde) + (U0prime/2) * (taylor.mle.x - theta1.tilde)^2
  abline(h=yval.taylor.mle, lty=2, col=line.col, lwd=2)

  # y-axis: only want to show y-values corresponding to these three horizontal lines :)
  axis(side=2, at=c(ll.theta1.tilde, ll.gamma.mle, yval.taylor.mle), labels=c("0", "18", "26"), las=2, cex.axis=1.5, padj=0.3)

  # Adding points
  points(x=theta1.tilde, y=ll.theta1.tilde, col="purple", pch=19, cex=1.7)
  points(x=gamma.mle, y=ll.gamma.mle, col="purple", pch=19, cex=1.7)
  points(x=taylor.mle.x, y=yval.taylor.mle, col="purple", pch=19, cex=1.7)

  # Arrows
  arrows(x0=4, x1=4, y0=ll.theta1.tilde, y1=ll.gamma.mle, col=col.1, length=0.15, code=3, lwd=2)
  arrows(x0=1.7, x1=1.7, y0=ll.theta1.tilde, y1=yval.taylor.mle, col=col.2, length=0.15, code=3, lwd=2)

  # Text
  text(x=2.8, y=-80+90, labels="LRT/2", font=2, cex=1.1, col=col.1)
  text(x=1, y=-80+90, labels="S/2", font=2, cex=1.2, col=col.2)

  # Adding text to indicate this is plot (b)
  text(x=14.2, y=37.5, labels="(b)", cex=2, col='black')

  dev.off()

}

###########################################################################################
