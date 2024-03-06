library(torch)
library(abind)
#  set the working directory to the one containing the following files

source("create_finance_multiplier.R")
source("fin_budget_ratio_parallel.R")
################################################################

t = 10 # 2 # 
n = nlgt # 3 # 
nprod = 30 # 4 # 
natt = 40 # 4 # 
ncars = 7 # 4 # 

###########################################################################
# Data 
gpu = torch_device("cuda",0)
cpu = torch_device("cpu")
device = gpu #cpu #gpu
Xk_stack_r = NULL   #Xk_stack_r dim:  nprod x natt x n x t    
yk_stack_r = NULL   # y_k_stack_r dim: 1 x nprod x n x t   
for(j in 1:t){
  Xk_ind=NULL
  yk_ind=NULL
  for(i in 1:n){
    #X
    Xk = as.matrix(simlgtdata[[i]]$Xind[[j]])  
    Xk = Xk[1:nprod,1:natt]
    Xk_ind = abind(Xk_ind,Xk,along = 3) 
    #y
    yk = t(as.matrix(simlgtdata[[i]]$y[j,])[1:nprod])
    yk_ind = abind(yk_ind,yk,along=3)
  }
  #X
  Xk_stack_r = abind(Xk_stack_r,Xk_ind,along = 4) 
  #y
  yk_stack_r = abind(yk_stack_r,yk_ind,along = 4)
}

Xk_stack = torch_tensor(aperm(Xk_stack_r,perm = c(4,3,1,2)),dtype = torch_double(),device = device)   #Xk_stack_r dim:  t x n x nprod x natt 
yk_stack = torch_tensor(aperm(yk_stack_r,perm = c(4,3,1,2)),dtype = torch_double(),device = device)   #Xk_stack_r dim:  t x n x 1 x nprod 
prices = Xk_stack[,,,1]
prices = prices[,,,NULL]
# prices_r = matrix(as.matrix(torch_tensor(prices,device = cpu)),ncol = nprod)
###########################################################################
beta_stack_r = NULL  # beta_Stack dim : n x natt x 1
for(i in 1:n) beta_stack_r =   abind(beta_stack_r, matrix(rep(0,natt),natt,1),along = 3)  # abind(beta_stack_r,matrix(rpois(natt,2),natt,1),along = 3) #
beta_stack = torch_tensor(aperm(beta_stack_r,perm = c(3,1,2)),dtype = torch_double(),device = device)

# theta
theta_stack = NULL   # beta_Stack dim : n x nprod x nprod

theta = matrix(0,nprod,nprod) #Thi[[i]]    
test = rep(0,435)
test[th_mnl] = -1000
theta[lower.tri(theta)] = test
theta


for(i in 1:n) theta_stack = abind(theta_stack, theta, along = 3)
theta_stack = torch_tensor(aperm(theta_stack,perm = c(3,1,2)),dtype = torch_double(),device = device)

# index vectors for indexing in torch
mat_idx1 = torch_tensor(1:(n*t),dtype = torch_long(),device = device)

idx1 = idx2 = NULL
for(i in 1:(n*t)){
  for(j in 1:nprod){
    idx1 = c(idx1,i)
    idx2 = c(idx2,j)
  }
}
cube_idx1 = torch_tensor(idx1, dtype = torch_long(),device = device)
cube_idx2 = torch_tensor(idx2, dtype = torch_long(),device = device)

extraBi_full_r = matrix(round(rnorm(t*n*ncars)*500),nrow= c(t*n)) # dimension: (t*n) x ncars
extraBi_full_r = NULL  # extraBi_full_r dim: (t*n) x ncars we truncate extraBi because we only need the columns corresponding the cars
for(i in 1:n){
  for(j in 1:t){
    extraBi_full_r = rbind(extraBi_full_r,simlgtdata[[i]]$extraBi[j,1:ncars])
  }
}
extraBi_full = torch_tensor(extraBi_full_r,dtype = torch_double(),device = device)


finmult = c(1,create_finmult())

fin_idx =  Xk_stack_r[1:ncars,35:38,,]


#################
# finance multiplier
finmult_full_r = NULL    # dimension of finmult_full_r = (n*t) x ncars 
for(i in 1:n){
  for(j in 1:t){
    finmult_temp = rep(0,ncars)
    columns = (which(fin_idx[,,i,j]==1)-1)%/%ncars + 1   #computes the column of the all elements = 1 in the i,j slice, basically the finance multiplier of the financeable cars
    rows = (which(fin_idx[,,i,j]==1)-1)%%ncars + 1   #computes the rows of the all elements = 1 in the i,j slice. This is the the cars associated with finmults in columns 
    finmult_temp[rows] = columns
    if(length(rows)!=length(columns)) cat("i=",i,"j=",j)
    finmult_temp = finmult_temp + 1  #every car that has no finance options is cash and = 1
    finmult_full_r = rbind(finmult_full_r,finmult[finmult_temp])
  }
}
finmult_full = torch_tensor(finmult_full_r,dtype = torch_double(),device = device)

budget_idx = matrix(rep(1:(n*t),ncars),ncol=ncars) # matrix that points to the respective row of the budget_draws
cash_ind = ifelse(finmult_full_r==1,1,0) # if the car comes with cash offer is 1 otherwise 0   dimension: (n*t) x ncars
#################
# inputs for the car block update
bonuses_array = torch_tensor(array(extraBi_full_r,dim = c(t,n,ncars,1)),dtype = torch_double(),device = device)
finmults_array = torch_tensor(array(finmult_full_r,dim = c(t,n,ncars,1)),dtype = torch_double(),device = device)
car_prices_array = prices[,,1:ncars,]
car_prices_full = torch_reshape(torch_transpose(car_prices_array,1,2),shape = c(n*t,ncars))

# objects previously constructed outside the function
M= 100
ybm = ybm
yG = torch_zeros(c(M,n*t,nprod),dtype = torch_double(),device = device)
block_obj = NULL #vector(mode = "list",max(ybm))
obj = NULL
for (i in 1:max(ybm)) {
  obj$comb = which(ybm==i)  # comb is the element(s) in yk corresponding to the current block [[1]] gives us the vector instead of the whole object
  obj$mincomb =  min(obj$comb)
  obj$maxcomb =  max(obj$comb) 
  obj$ncomb = torch_tensor(length(obj$comb),dtype = torch_int(),device = device)   # number of comb elements
  obj$ncomb1 = torch_tensor(length(obj$comb)+1,dtype = torch_int(),device = device) #torch_tensor(obj$ncomb+1,dtype = torch_long(),device = device) # mostly used in the bonus function   
  obj$eye = torch_eye(length(obj$comb), dtype = torch_double(),device = device)
  obj$start = torch_tensor(obj$comb[1]-1,dtype = torch_long(),device = device)
  obj$one = torch_tensor(1,dtype = torch_double(),device = device)
  block_obj[[i]] = obj
  
}
Mvec_double = torch_tensor(1:M,dtype = torch_long(),device = device) # just a predefined Long tensor sequence from 1 to M 
zero_double = torch_zeros(1,dtype = torch_double(),device = device)



##########################
# current budgets 
budget = maxpsum[1:n,] + .001
budget_draws =NULL
for(i in 1:n) budget_draws = c(budget_draws,rep(budget[i,],t))
budget_draws = matrix(budget_draws,ncol = 2,byrow = T) # budget_draws are in fact nx2 but we replicate each budget row t times *** use maxpsum + .001
# candidate budgets 
budgetcand_draws = budget_draws
fin_budget_cand = t(matrix(budgetcand_draws[,1],t, n)) + rnorm(n,mean=0,sd=10)
budgetcand_draws[,1] = t(fin_budget_cand)     # here we are testing the update of finance budget, so the cash budget remains the same
budgetcand_draws = ifelse(budgetcand_draws<0,.25,budgetcand_draws)  # non-zero budgets

# comparing current budget to candidate budget  
budget_indicator_r = ifelse(budgetcand_draws[,1]>budget_draws[,1],1,0)
budget_indicator_temp = array(budget_indicator_r,dim = c(1,1,t,n))
budget_indicator = torch_tensor(aperm(budget_indicator_temp,perm=c(3,4,1,2)),dtype = torch_double(),device = device)
# we generate choices with the higher budget
budget1 = budget_draws  
budget1[which(budget_indicator_r==1),1] = budgetcand_draws[which(budget_indicator_r==1),1]  # replace the finance budgets that are lower than the candidate
# we evaluate the normalizing constant (ratio of within budget choices) with the lower budget
budget2 = budgetcand_draws
budget2[which(budget_indicator_r==1),1] = budget_draws[which(budget_indicator_r==1),1]

# Gibbs sampler input
budget_full_r = matrix(budget1[c(t(budget_idx+cash_ind*n*t))], ncol=ncars, byrow = T)  
budget_full = torch_tensor(budget_full_r,dtype = torch_double(),device = device)       # dimension of budget_full = (n*t) x ncars , and contains the corresponding budget for each car in each menu (depending on the individual's budget and whether the car can be financed)
budget2_full_r = matrix(budget2[c(t(budget_idx+cash_ind*n*t))], ncol=ncars, byrow = T)
budget2_full = torch_tensor(budget2_full_r,dtype = torch_double(),device = device)

budgets_array = torch_tensor(array(budget_full_r,dim = c(t,n,ncars,1)),dtype = torch_double(),device = device)

#######################
start = Sys.time()
out_Simr_torch = fin_budget_ratio_parallel(block_obj, Mvec_double, zero_double, yG, Xk_stack, yk_stack, prices, beta_stack, theta_stack, budget_full, budget2_full, budget_indicator,  # defined at runtime
                                                  extraBi_full, finmult, finmult_full, car_prices_full, 
                                                  bonuses_array, finmults_array, car_prices_array, budgets_array, 
                                                  mat_idx1,cube_idx1,cube_idx2, 
                                                  n,t,nprod, ybm, M, device)


duration = Sys.time() - start
duration

log_NC_stacked = as.matrix(out_Simr_torch)
# ############################################################################################################
# ############################# profiling  Simr_finoptions_extraBi_torch_parallel ############################
# ############################################################################################################
# 
# # Start profiling
# Rprof("my_profile.out")
# 
# # Code that you want to profile
# out_Simr_torch = fin_budget_ratio_parallel(block_obj, Mvec_double, zero_double, yG, Xk_stack, yk_stack, prices, beta_stack, theta_stack, budget_full, budget2_full, budget_indicator,  # defined at runtime
#                                            extraBi_full, finmult, finmult_full, car_prices_full, 
#                                            bonuses_array, finmults_array, car_prices_array, budgets_array, 
#                                            mat_idx1,cube_idx1,cube_idx2, 
#                                            n,t,nprod, ybm, M, device)
# 
# # Stop profiling
# Rprof(NULL)
# 
# # View the profile data
# summaryRprof("my_profile.out")