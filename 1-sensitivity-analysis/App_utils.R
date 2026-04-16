
###############################
#Define rank based mahalanobis distance
##################################
smahal=
  function(z,X, weight = rep(1, ncol(X))){
    X<-as.matrix(X)
    n<-dim(X)[1]
    rownames(X)<-1:n
    k<-dim(X)[2]
    m<-sum(z)
    for (j in 1:k){ X[,j]<-rank(X[,j])}
    cv<-cov(X)
    vuntied<-var(1:n)
    rat<-sqrt(vuntied/diag(cv))
    cv<-diag(rat)%*%cv%*%diag(rat)
    out<-matrix(NA,m,n-m)
    Xc<-X[z==0,]
    Xt<-X[z==1,]
    rownames(out)<-rownames(X)[z==1]
    colnames(out)<-rownames(X)[z==0]
    library(MASS)
    W = diag(1/sqrt(weight))
    icov<-ginv(W%*%cv%*%W)
    
    for (i in 1:m) out[i,]<-mahalanobis(Xc,Xt[i,],icov,inverted=T)
    out
  }
exact.match=function(dmat,z,exact){
  penalty = max(dmat)*100
  adif=abs(outer(exact[z==1],exact[z==0],"-"))
  for(i in 1:nrow(dmat))
  {
    for(j in 1:ncol(dmat))
    {
      if(adif[i,j]!= 0)
      {
        dmat[i,j] = penalty
      }
    }
  }
  dmat
}


########################
#Add a caliper
#######################
addcaliper=function(dmat,z,logitp,calipersd=.2,penalty=1000){
  sd.logitp=sd(logitp)
  adif=abs(outer(logitp[z==1],logitp[z==0],"-"))
  adif=(adif-(calipersd*sd.logitp))*(adif>(calipersd*sd.logitp))
  dmat=dmat+adif*penalty
  dmat
}

#################################
#Standardized Differences for Full Matching
#################################
standardized.diff.func=function(x,treatment,stratum.myindex,missing=rep(0,length(x))){
  xtreated=x[treatment==1 & missing==0];
  xcontrol=x[treatment==0 & missing==0];
  var.xtreated=var(xtreated);
  var.xcontrol=var(xcontrol);
  combinedsd=sqrt(.5*(var.xtreated+var.xcontrol));
  std.diff.before.matching=(mean(xtreated)-mean(xcontrol))/combinedsd;
  nostratum=length(unique(stratum.myindex))-1*min(stratum.myindex==0);
  diff.in.stratum=rep(0,nostratum);
  treated.in.stratum=rep(0,nostratum);
  stratum.size=rep(0,nostratum);
  for(i in 1:nostratum){
    diff.in.stratum[i]=mean(x[stratum.myindex==i & treatment==1 & missing==0])-mean(x[stratum.myindex==i & treatment==0 & missing==0]);
    stratum.size[i] = sum(stratum.myindex==i & missing == 0)
    treated.in.stratum[i]=sum(stratum.myindex==i & treatment==1 & missing==0);
    if(sum(stratum.myindex==i & treatment==0 & missing==0)==0 || sum(stratum.myindex==i & treatment==1 & missing==0)==0){
      treated.in.stratum[i]=0;
      diff.in.stratum[i]=0;
    }
  }
  std.diff.after.matching=(sum(treated.in.stratum*diff.in.stratum)/sum(treated.in.stratum))/combinedsd;
  list(std.diff.before.matching=std.diff.before.matching,std.diff.after.matching=std.diff.after.matching);
}


#################################
#Plot Standardized Differences
############################

plotBalancesign <- function(stdDiff.Before,stdDiff.After,covName,covGroup,titleOfPlot, maxValue = NULL) {
  #in case someone doesn't take the absolute value ahead of time:	
  # Load necessary package
  library(lattice)
  
  # Length of covariate vector
  p = length(covName)
  
  if(is.null(maxValue))
  {
    maxValue = max(stdDiff.Before,stdDiff.After)
  }
  minValue = min(stdDiff.Before, stdDiff.After)
  maxValue = max(abs(maxValue), abs(minValue))
  if(missing(covGroup)) {
    # Setup data frame for lattice package to work
    # In short, it creates a p-by-3 matrix where the rows
    # represent covariates, the first column represents stdDiff before matching
    # the second column represents stdDiff after matching,
    # and the third column represents the names of the covariates. 
    plot.dataframe = data.frame(stdDiff.Before = stdDiff.Before,
                                stdDiff.After = stdDiff.After,
                                covName = covName)
    plot.dataframe$covName = as.factor(plot.dataframe$covName)	
    # This reorder step is necessary to achieve an alphabet-ordering of the covariates on the dot plot
    plot.dataframe$covName = reorder(plot.dataframe$covName,-1*(1:length(plot.dataframe$covName)))
  } else {
    # Set up data frame for lattice package to work
    # We have to reorganize the stdDiff values so that they are grouped 
    # by the covGroup values.
    uniqueCovGroupNames = unique(covGroup)
    nGroups = length(uniqueCovGroupNames)
    
    # We have to add dummy variables for covGroup names
    # Hence, we have p + nGruops, instead of just p
    stdDiff.Before.grouped = rep(0,p + nGroups) 
    stdDiff.After.grouped = rep(0,p + nGroups)
    covName.grouped = rep("",p + nGroups)
    
    index = 1# counter for the for loop
    # Iterate through all the covGroup names
    for(i in 1:length(uniqueCovGroupNames)) {
      # Find covNames that belong to one particular covGroup name
      covInGroup.i = which(covGroup == uniqueCovGroupNames[i])
      
      # Append the name of the covGroup into the stdDiff vectors
      stdDiff.Before.grouped[index] = -3
      stdDiff.After.grouped[index] = -3
      covName.grouped[index] = paste(uniqueCovGroupNames[i],":   ",sep="")
      
      # Copy all the covNames in this particular covGroup into the stdDiff group vector
      stdDiff.Before.grouped[(index+1):(index + 1 + length(covInGroup.i) -1)] = stdDiff.Before[covInGroup.i]
      stdDiff.After.grouped[(index+1):(index + 1 + length(covInGroup.i) - 1)] = stdDiff.After[covInGroup.i]
      covName.grouped[(index+1):(index + 1 + length(covInGroup.i) - 1)] = covName[covInGroup.i]
      
      # Update the index vector
      index = (index + 1 + length(covInGroup.i))
    }
    nPergroup = table(covGroup)
    ln = length(nPergroup)
    index = cbind(c(2, cumsum(nPergroup[1:(ln-1)]) + 1 + (2:ln)),c(cumsum(nPergroup) + (1:ln)))
    
    # Creating the data frame for lattice package. 
    plot.dataframe = data.frame(stdDiff.Before = stdDiff.Before.grouped,
                                stdDiff.After = stdDiff.After.grouped,
                                covName = covName.grouped)
    plot.dataframe$covName = as.factor(plot.dataframe$covName)
    # This reorder step is necessary to achieve an alpha-ordering of the covariates on the dot plot.
    plot.dataframe$covName = reorder(plot.dataframe$covName,-1*(1:length(plot.dataframe$covName)))
    
  }
  dotplot(covName ~ stdDiff.Before,data=plot.dataframe,
          xlim = c(-0.025 - maxValue ,maxValue + 0.025),
          xlab = list("Standardized Differences",cex=0.75),
          main = titleOfPlot,
          panel = function(x,y,...) {
            panel.abline(h = as.numeric(y),lty=2,col="gray") #draws the gray horizonal lines
            
            panel.segments(.2,y[1], .2, y[length(covName)], lty = 6, lwd = 2, col = "red")
            panel.segments(-.2,y[1], -.2, y[length(covName)], lty = 6, lwd = 2, col = "red")
            
            panel.xyplot(x,y,pch=16,col="black",cex=0.75) #plots the before matching stdDiff values
            panel.xyplot(plot.dataframe$stdDiff.After,y,pch=5,col="black",cex=0.6)}, #plots the after matching stdDiff values
          key = list(text = list(c("Before","After"),cex=0.75), #legend
                     points = list(pch = c(16,5),col="black",cex=0.75), #note that pch controls the type of dots
                     space = "right", border=TRUE),
          scales = list(y = list(cex = 0.6)) )
  
} 


