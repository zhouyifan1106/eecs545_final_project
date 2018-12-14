# Generate 10000*6 matrix for X (N(0,1) distribution)
m=data.frame(matrix(rnorm(10000*6),10000,6))

# Generate y1
m$y1 = m$X1+(m$X2-1)^2-exp(m$X3+2)+sin(m$X4)+1/(3-m$X5)*m$X6
for (i in 1:10000) {
  if (m[i,7] < mean(m$y1)) {
    m[i,7] = 0
  } else {
    m[i,7] = 1
  }
}

# Generate y2, positively correlated to y1
m$y2 = NA
for (i in 1:10000) {
  if (m[i,7] == 1) {
    m[i,8] = rbinom(1,1,6/7)
  } else {
    m[i,8] = rbinom(1,1,1/7)
  }
}

# Generate y3, genatively correlated to y1
m$y3 = NA
for (i in 1:10000) {
  if (m[i,7] == 1) {
    m[i,9] = rbinom(1,1,1/7)
  } else {
    m[i,9] = rbinom(1,1,6/7)
  }
}


# Generate y4
m$y4 = NA
for (i in 1:10000) {
  if (m[i,7] == 1 && m[i,8] == 1) {
    m[i,10] = rbinom(1,1,1/7)
  } else if (m[i,7] == 0 && m[i,8] == 0) {
    m[i,10] = rbinom(1,1,6/7)
  } else {
    m[i,10] = rbinom(1,1,1/2)
  }
}

# Generate y5
m$y5=2*m$X1+(m$X2-1)^3-exp(m$X3-2)+cos(m$X4)+5/(4-m$X5)+log(abs(m$X6))
for (i in 1:10000) {
  if (m[i,11] > mean(m$y5)) {
    m[i,11] = 0
  } else {
    m[i,11] = 1
  }
}


# Generate y6
m$y6=2^(m$X1)+(m$X2-1)/3-abs(m$X3-2)+sin(m$X4+m$X6)+(2-1/m$X5)
for (i in 1:10000) {
  if (m[i,12] > mean(m$y6)) {
    m[i,12] = 0
  } else {
    m[i,12] = 1
  }
}


x <- m[,1:6]
y <- m[,7:12]
y = y[,c(1,2,5,3,6,4)]

write.csv(x,"x.csv",row.names = F)
write.csv(y,"y.csv",row.names = F)


