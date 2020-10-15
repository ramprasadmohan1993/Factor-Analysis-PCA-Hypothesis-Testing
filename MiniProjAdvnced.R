getwd()
setwd("D:/R_wd")
#######################################################################################################################
#                                                                                                                    #
# Purpose:       Advanced Stat - Mini Project                                                                        #
#                                                                                                                    #
# Author:        Ramprasad                                                                                           #                                                    
#                                                                                                                    #
# Code created:  2020-09-08                                                                                          #
# Last updated:  2020-09-21                                                                                          #
# Source:        C:/Users/indway/Desktop                                                                             #
#                                                                                                                    #
######################################################################################################################

pacman::p_load(ggplot2,ggpubr,corrplot,grid,gridBase,psych,readr,summarytools,
               MVN,DataExplorer,ggfortify,car,nFactors,GPArotation,ppcor)

hair <- read.csv(file.choose(),header = TRUE)
class(hair)

hairorginal <- hair[,2:13]

## or hairorginal <- hair[,which(names(data)!="ID")]

haircustrm <- hair[,2:12]

### EDA - Univariate, Bivariate graphs ###

pairs.panels(hairorginal,method = "pearson",hist.col = "Green",density = TRUE,lm = TRUE,ellipses = FALSE,rug = TRUE)
summarytools::view(dfSummary(hairorginal))

### EDA - Outliers, summary and missing values ###
summary(is.na(hairorginal))
summary(hairorginal)

plot.new()
vp <- viewport(x = .15, y = 0, just = c("left", "bottom"), 
               width = .85, height = 1)
pushViewport(vp)
par(new = TRUE, fig = gridFIG())
boxplot(hairorginal,las =1,horizontal = TRUE,col="Blue")

### Check Multicollinearity #####

corr <- cor(hairorginal)
corrplot(corr,method = "color",type = "lower",outline = TRUE,addCoef.col = "Black",is.corr = FALSE)
corrplot(corr,method = "pie",type = "lower",outline = TRUE)
cor.plot(corr,upper = FALSE,las=2)

mvn(hairorginal)

lmtesting <- lm(hairorginal$Satisfaction ~ . ,data = hairorginal)
car::vif(lmtesting)

library(mctest)
mctest::mc.plot(lmtesting)
mctest::imcdiag(lmtesting)
mctest::omcdiag(lmtesting)

vif_custfn<-dget("D:/BACP/Predictive Modelling/Final Project/vif_fun.R")
vif_custfn(in_frame = hairorginal[,-12],thresh = 5,trace = TRUE)

## Simple Linear Regression with every variable ####

attach(hairorginal)
for (i in 1:ncol(hairorginal)-1)
{
  print("---------------------------------------")
  print(paste0("Model for combination -",i))
  field <- as.data.frame(hairorginal[,c(i,12)])
  formula <- paste0("Satisfaction ~ ",names(field)[1])
  print(formula)
  slm <- lm(formula = formula, data = field)
  print(summary(slm))
  }

#----------------------------------------------------Factor Analysis---------------------------------------------------#

### KMO - Barlett test ###

corr <- cor(haircustrm)
corr
KMO(r = corr)

### Eigen value ###

ev <- eigen(corr)
ev

## Scree Plot ###

eigenvalues <- ev$values
nooffactors <- c(1:11)
scree <- data.frame(nooffactors,eigenvalues)
plot(scree,col = "Red",main = "Scree Plot",xlab = "No of Factors",ylab="Eigen Values",pch = 18)
lines(scree,col = "Blue")
abline(h = 1.0,lty=2)

## Kaiser Rule to get no of factors ###

count <- 0
for(val in eigenvalues)
{
  if(val > 1)
  {
    count <- count + 1
  }
}

### Factor Analysis ####
library(factoextra)
library(FactoMineR)

unrotatedone<-principal(haircustrm,nfactors = count,rotate = "none")
print.psych(unrotatedone,sort = TRUE)
psych::fa.diagram(unrotatedone)
psych::biplot.psych(unrotatedone)
plot(unrotatedone,row.names(unrotatedone$loadings))


res.pca <- prcomp(haircustrm,nfactors = count,rotate = "none")
factoextra::fviz_pca_ind(res.pca)
factoextra::fviz_pca_biplot(res.pca)
factoextra::fviz_pca_var(res.pca,col.var = "contrib",gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),repel = TRUE)

## ----------- With Varimax rotation -----------------------------------------###

rotated <- principal(haircustrm,nfactors = count,rotate = "varimax")
print.psych(rotated,sort = TRUE)
psych::fa.diagram(rotated)
psych::biplot.psych(rotated)
plot(rotated,row.names(rotated$loadings))

fa.diagram(vari.pca)

vari.pca <- fa(r = haircustrm,nfactors = count,rotate = "varimax",fm = "pa")

class(rotated)
psych::
  
factoextra::fviz_pca_ind(rotated)
factoextra::fviz_pca_biplot(res.pca)
factoextra::fviz_pca_var(res.pca,col.var = "contrib",gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),repel = TRUE)

### ------------------------------- Multiple Linear Regression ---------------------------------------###

indp <- data.frame(rotated$scores)
colnames(indp) <- c("Postsale.customerservice","Marketing","Technical.Support","Product.Value")
colnames(indp)

cust <- hair[,c("ID","Satisfaction")]


mldata <- data.frame(cust,indp)

mvn(mldata)
attach(mldata)

mlrm<-lm(Satisfaction~Postsale.customerservice+Marketing+Technical.Support+Product.Value)
mlrm
summary(mlrm)
mlrtable<- anova(mlrm)
car::vif(mlrm)