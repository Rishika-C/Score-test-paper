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
  # Shifting curves accordingly -- so that for scenario 1, curves intersect at peak; for scenario 2, curves intersect at theta1-tilde
  ll.subtract = ll[beta1.seq==which.beta.subtract]
  # Log-lik values
  ll.diff = ll - ll.subtract
  ll.diff
}


# Scenario (1): simple plot building intuition for score test statistic (based on distance summarised w LRT stat)
cairo_pdf("Figures/Fig1a.pdf")

beta1.seq = seq(-1.5, 3.5, by=0.01) # x seq for plotting

# Reducing space between y-axis label and y-axis
par(mgp=c(3, 1, 0), mar=c(5.1, 5.5, 4.1, 2.1))

# Settings to use
# Sample size, mean
n.1 = 10
mean.y.1 = 1

# Plotting log-lik curve
plot(beta1.seq, ll.diff.func(n=n.1, mean.y=mean.y.1, beta1.seq=beta1.seq, which.beta.subtract=mean.y.1), type="l", xlab="", ylab=expression("\u2113"(bold(theta)[1]) - "\u2113"(hat(bold(theta))[1])), lwd=3, cex.lab=1.6, cex.axis=1.5, xaxs="i", col="red", xaxt="n", ylim=c(-13, 4)) # , yaxt="n")
ll.y.1 = ll.diff.func(n=n.1, mean.y=mean.y.1, beta1.seq=beta1.seq)[which(beta1.seq==0)] # log-lik(theta1-tilde) for n=10, mean=1 -- useful below
# Note that due to ur settings, we know MLE = mean.y, and due to shifting, peak is at y=0. We'll use this info below

# Adding y axis with l(beta1) labels
#axis(side=2, at=c(ll.y.1, 0), labels=c(expression("\u2113"(tilde(bold(theta))[1])), expression("\u2113"(hat(bold(theta))[1]))), cex.axis=1.5, padj=0.3)

# Adding x axis with beta1 labels
axis(side=1, at=c(0, 1), labels=c(expression(tilde(bold(theta))[1]), expression(hat(bold(theta))[1])), cex.axis=1.5, padj=0.3)
title(xlab=expression(bold(theta)[1]), mgp=c(3, 1, 0), cex.lab=1.6) # xlab

# Grid lines
grid()

n.2 = 160
mean.y.2 = 0.25
lines(beta1.seq, ll.diff.func(n=n.2, mean.y=mean.y.2, beta1.seq=beta1.seq, which.beta.subtract=mean.y.2), col='black', lwd=3)

# Adding tangent line at theta1-tilde
# Score at theta-tilde=sum(y)=n*mean.y. To plot gradient, will go 0.5 units either side of beta1=0, hence the 0.5's below :)
# If was going one unit either side of -1: for tangent line at beta1=-1, is log-lik(theta-tilde) - n*mean.y; at beta1=1, is log-lik(theta-tilde) + n*mean.y. We'll use this.
lines(x=c(-0.5, 0.5), y=c(ll.y.1 - 0.5*n.1*mean.y.1, ll.y.1 + 0.5*n.1*mean.y.1), col="red", lty=2, lwd=3) # Tangent line

lines(x=c(-0.2, 0.2), y=c(ll.y.1 - 0.2*n.2*mean.y.2, ll.y.1 + 0.2*n.2*mean.y.2), col="black", lty=2, lwd=3)

# Dashed line at theta-tilde and theta-hat
lines(x=c(0, 0), y=c(-20, ll.y.1), col="darkgreen", lty=2, lwd=2)
lines(x=c(1, 1), y=c(-20, 0), col="darkgreen", lty=2, lwd=2)

## # Dashed lines to highlight difference along y-axis
## abline(h=0, col="purple", lty=2, lwd=2)
## abline(h=ll.y, col="purple",  lty=2, lwd=2)
# Points at ll(theta1-tilde), ll(theta1-hat)
points(x=0, y=ll.y.1, col="purple", pch=19, cex=1.7)
points(x=mean.y.1, y=0, col="purple", pch=19, cex=1.7) # Remember, we know MLE = mean.y, and due to shifting, peak is at y=0
points(x=mean.y.2, y=0, col="purple", pch=19, cex=1.7)
## # Arrowing indicating difference is of interest
## arrows(x0=-0.5, y0=ll.y, x1=-0.5, y1=0, col='purple', length=0.15, code=3, lwd=2.5)

# LRT = 10 for both :)
lrt.func(n=n.1, mean.y=mean.y.1)
lrt.func(n=n.2, mean.y=mean.y.2)

# Legend
legend("topleft", legend=c(expression(S~"="~LRT~"="~10~", "~p~"="~0.002),
                           expression(S~"="~LRT~"="~10~", "~p~"="~0.002)), lwd=2, col=c("red", "black"), bg="white", cex=1.3)

dev.off()



## # Scenario (1): simple plot building intuition for score test statistic (based on distance summarised w LRT stat)
## cairo_pdf("Figures/Temp.pdf")

## beta1.seq = seq(-1.75, 2.25, by=0.01) # x seq for plotting

## # Reducing space between y-axis label and y-axis
## par(mgp=c(3, 1, 0), mar=c(5.1, 5.5, 4.1, 2.1))

## # Settings to use
## # Sample size, mean
## n.1 = 160
## mean.y.1 = 0.25

## # Plotting log-lik curve
## plot(beta1.seq, ll.diff.func(n=n.1, mean.y=mean.y.1, beta1.seq=beta1.seq, which.beta.subtract=mean.y.1), type="l", xlab="", ylab=expression("\u2113"(bold(theta)[1]) - "\u2113"(hat(bold(theta))[1])), lwd=3, cex.lab=1.6, cex.axis=1.5, xaxs="i", col="red", xaxt="n", ylim=c(-15, 2))#, yaxt="n")
## ll.y.1 = ll.diff.func(n=n.1, mean.y=mean.y.1, beta1.seq=beta1.seq, which.beta.subtract=mean.y.1)[which(beta1.seq==0)] # log-lik(theta1-tilde) -- useful below
## # Note that due to ur settings, we know MLE = mean.y, and due to shifting, peak is at y=0. We'll use this info below

## # Adding y axis with l(beta1) labels
## #axis(side=2, at=c(ll.y.1, 0), labels=c(expression("\u2113"(tilde(bold(theta))[1])), expression("\u2113"(hat(bold(theta))[1]))), cex.axis=1.5, padj=0.3)

## # Adding x axis with beta1 labels
## axis(side=1, at=c(0, mean.y.1), labels=c(expression(tilde(bold(theta))[1]), expression(hat(bold(theta))[1])), cex.axis=1.5, padj=0.3)
## title(xlab=expression(bold(theta)[1]), mgp=c(3, 1, 0), cex.lab=1.6) # xlab

## # Grid lines
## grid()

## # Adding tangent line at theta1-
## # Score at theta-tilde=sum(y)=n*mean.y. To plot gradient, will go 0.4 units either side of beta1=0, hence the 0.4's below :)
## # If was going one unit either side of -1: for tangent line at beta1=-1, is log-lik(theta-tilde) - n*mean.y; at beta1=1, is log-lik(theta-tilde) + n*mean.y. We'll use this.
## lines(x=c(-0.2, 0.2), y=c(ll.y.1 - 0.2*n.1*mean.y.1, ll.y.1 + 0.2*n.1*mean.y.1), col="red", lty=2, lwd=3) # Tangent line

## # Dashed line at theta-tilde and theta-hat
## lines(x=c(0, 0), y=c(-20, ll.y.1), col="darkgreen", lty=2, lwd=2)
## lines(x=c(mean.y.1, mean.y.1), y=c(-20, 0), col="darkgreen", lty=2, lwd=2)

## # Adding points at theta's
## points(x=0, y=ll.y.1, col="purple", pch=19, cex=1.7)
## points(x=mean.y.1, y=0, col="purple", pch=19, cex=1.7)

## # Confirming LRT also = 10
## lrt.func(n=160, mean=0.25)

## # Legend
## legend("topleft", legend=c(expression(S~"="~LRT~"="~10~", "~p~"="~0.002)), lwd=2, col=c("red"), bg="white", cex=1.3)

## dev.off()




# Scenario (2): same score at theta1-tilde --> gradient alone is not enough! :)
cairo_pdf("Figures/Fig1b.pdf")

beta1.seq = seq(-2, 2, by=0.01) # x seq for plotting

# Distance bw y-axis label and y-axis
par(mgp=c(3.2, 1, 0), mar=c(5.5, 5.5, 4.1, 2.1))

# Parameter settings
n.1 = 5
n.2 = 20
mean.y.1 = 1
mean.y.2 = 0.25

# As above, we'll use the information that MLE(theta) = mean(y) for each sample below. We'll also use info that theta1-tilde=0, and at theta-tilde, the y-value of both peaks is 0, by construction  (as we use which.beta.substract=0 below)

# Blue curve: larger discrepancy between theta-tilde and theta1-hat
# n=5, mean.y=1, S=5
plot(beta1.seq, ll.diff.func(n=n.1, mean.y=mean.y.1, which.beta.subtract=0), type="l", xlab="", ylab=expression("\u2113"(bold(theta)[1]) - "\u2113"(hat(bold(theta))[1])), col='blue', lwd=3, cex.lab=1.6, xaxt="n", cex.axis=1.2, xaxs="i", ylim=c(-6,4), cex.axis=1.5)# xlim=c(-1.5, 2), ylim=c(-10, 3)

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

## # Dashed lines highlighting discrepancy along y-axis
## abline(h=0, col="purple", lty=2, lwd=2)
## abline(h=ll.y.n.1, col="purple",  lty=2, lwd=2)
## abline(h=ll.y.n.2, col="purple",  lty=2, lwd=2)
# Points at ll(theta1-tilde), ll(theta1-hat) for both curves. For latter, choose x-values=mean(y)=MLE for theta
points(x=0, y=0, col="purple", pch=19, cex=1.5)
points(x=mean.y.1, y=ll.y.n.1, col="purple", pch=19, cex=1.7)
points(x=mean.y.2, y=ll.y.n.2, col="purple", pch=19, cex=1.7)
## # Arrowing indicating difference is of interest
## arrows(x0=-1, y0=0, x1=-1, y1=ll.y.n.2, col='purple', length=0.1, code=3, lwd=2.5)
## arrows(x0=-0.5, y0=0, x1=-0.5, y1=ll.y.n.1, col='purple', length=0.15, code=3, lwd=2.5)

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

###########################################################################################
