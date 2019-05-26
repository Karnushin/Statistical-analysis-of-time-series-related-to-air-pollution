library(MASS)
library(rriskDistributions)
library(fitdistrplus)
library(CDFt)
library(goftest)
library(zoo)

data <- read.csv("/Users/Barnett/RProjects/GammaMixEMDiploma/TSP/TSPall.csv",header = TRUE)
d = sort(sample(nrow(data), nrow(data)*.2))
test<-data[d,]
train<-data[-d,]
gam <- fitdist(data$x, "gamma", method="mle")
ks.test(test, 'pgamma', gam[["estimate"]][["shape"]], gam[["estimate"]][["rate"]])
gam
sh = gam[['estimate']][['shape']]
sc = 1/gam[['estimate']][['rate']]
num_of_samples = length(data$x)
y <- pgamma(num_of_samples, shape=sh,scale=sc)
res <- CramerVonMisesTwoSamples(data$x,y)
1/6*exp(-res)

x=data$x
p1 <- hist(x)
breaks_cdf <- pgamma(p1$breaks, shape=sh, scale=sc)
null.probs <- rollapply(breaks_cdf, 2, function(x) x[2]-x[1])
chisq.test(p1$counts, p=null.probs, rescale.p=TRUE, simulate.p.value=TRUE)

