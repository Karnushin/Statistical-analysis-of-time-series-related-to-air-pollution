library(mixtools)

data <- read.csv(".../GammaMixEMDiploma/HG/HGall.csv",header = TRUE)
x <- data$x
hist(x, col = 'blue', breaks = 40,lwd=1,freq=FALSE)

gp <- gammamixEM(x, alpha = c(1.8,1.7,1.5,1.7),beta = c(2, 3.1, 4.8, 3),
                  lambda=c(0.23,0.27,0.25,0.25),
                  k=4, maxit = 500, epsilon = 1e-6, verb = TRUE)
gp["gamma.pars"]
gp["lambda"]
