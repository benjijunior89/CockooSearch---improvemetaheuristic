
xbest<-c(3,4)
iterations<-10000
points<-matrix(ncol=2,nrow=iterations)

points[1,1]<-0
points[1,2]<-0

lambda=2
epsilon=0.001
beta<-2/3
sigma<-(gamma(1.0+beta)*sin((pi*beta)/2))/(gamma(((1.0+beta)/2) * beta * 2^((beta-1.0)/2)))

for(i in 2:iterations){
  u<-rnorm(1,mean=0,sd=sigma)
  v<-rnorm(1,mean=0,sd=1)
  
  step <- u / (abs(v)^(-lambda))
  
  nu<-epsilon*step*(points[i-1,]-xbest)  
  
  nuGamma<-nu*rnorm(2,mean=0,sd=1)
  
  points[i,]<-points[i-1,]+nuGamma
}







showNests<-FALSE

plot(points[,1],points[,2],type="l", main="Ejemplo Levy Flights", xlab="x", ylab="y")
if(showNests){
  for(i in 2:nrow(points)){
    points(points[i,1],points[i,2],col=i,cex=0.3)
  }
}

points(points[1,1],points[1,2],col="red", pch=4,  cex=2, lwd=2)
points(points[nrow(points),1],points[nrow(points),2],col="blue", pch=4,  cex=2, lwd=2)

