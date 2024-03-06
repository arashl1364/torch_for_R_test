fin_budget_ratio_parallel = function( block_obj, Mvec_double, zero_double, yG,Xk_stack, yk_stack, prices,  # predefined 
                                                      beta_stack, theta_stack, budget_full, budget2_full, budget_indicator,  # defined at runtime
                                                      extraBi_full, finmult, finmult_full, car_prices_full,        # predefined
                                                      bonuses_array, finmults_array, car_prices_array, budgets_array,       # predefined
                                                      mat_idx1,cube_idx1,cube_idx2,    # predefined
                                                      nreg,t,nprod, ybm, M, device){    # predefined  
  # Arash, September 2023  
  # This function returns for each individual's repeated measurements the ratio of financed choices made with budget are within budget2, given the stacked parameters and data in torch tensors
  # It accounts for different financing options and bonuses for add-ons
  # the operation is done in parallel across repeated measurements and individuals
  
  # start = Sys.time()
  # time=NULL
  # time = c(time,Sys.time()-start)
  
  compute_car_price_bonus_finmult_budget = function(mat_idx1,car_idx,extraBi_full,finmult_full,budget_full,car_prices_full, nprod,t,nreg){
    # this function gets the index of cars chosen in the current iteration of gibbs sampler 
    # and returns their prices, bonuses, finance multiplier and corresponding budget of them in four (txnx1x1) tensors 

    bonus = torch_transpose(torch_reshape(torch_index(self = extraBi_full,indices = list(mat_idx1,car_idx)),shape = c(nreg,t,1,1)),1,2)
    car_finmult = torch_transpose(torch_reshape(torch_index(self = finmult_full,indices = list(mat_idx1,car_idx)),shape = c(nreg,t,1,1)),1,2)
    car_prices = torch_transpose(torch_reshape(torch_index(self = car_prices_full,indices = list(mat_idx1,car_idx)),shape = c(nreg,t,1,1)),1,2)
    car_budget = torch_transpose(torch_reshape(torch_index(self = budget_full,indices = list(mat_idx1,car_idx)),shape = c(nreg,t,1,1)),1,2)

    return(list(bonus = bonus, car_finmult = car_finmult, car_prices = car_prices, car_budget = car_budget))  
  } 
  
  
  # yG = torch_zeros(c(M,nreg*t,nprod),dtype = torch_double(),device = gpu)
  car_idx = NULL   #car_idx will be updated at each car block 
  
  # financing options
  maxybm = max(ybm)
  for(r in 1:M){    # MCMC loop
    for(kk in 1: maxybm){  # loop over blocks
      car = kk == 1 
      
      # creating y_mat (potential choices with different options in the current block)
      if(car){
        
        y_mat_stack = torch_repeat_interleave(yk_stack, block_obj[[kk]]$ncomb, dim=3) #y_mat = torch_repeat_interleave(yk_tr, ncomb, dim=1)
        y_mat_stack[,,,block_obj[[kk]]$comb] = block_obj[[kk]]$eye
      }else{
        # note: in add-on blocks we have a no-choice option too
        y_mat_stack =  torch_repeat_interleave(yk_stack, block_obj[[kk]]$ncomb1, dim=3)
        y_mat_stack[,,1, block_obj[[kk]]$mincomb:block_obj[[kk]]$maxcomb] = 0   # no-choice of add-on  
        y_mat_stack[,,2:y_mat_stack$size(1), block_obj[[kk]]$mincomb:block_obj[[kk]]$maxcomb] =  block_obj[[kk]]$eye
      }
      

      # creating mu_temp and p_temp (Xbeta and price of potential choices with different options in the block)
      # p_temp and mu_temp dimensions: t x nreg x ncomb x 1 (note: in the car block ncomb = ncars)
      res1 = torch_matmul(torch_matmul(y_mat_stack,Xk_stack),beta_stack) # *** torch_chain_matmul
      y_mat_stack_tr = y_mat_stack$permute(c(1,2,4,3)) # *** integrate this to the next line?
      res2 = torch_matmul(torch_matmul(y_mat_stack,theta_stack),y_mat_stack_tr)  # *** torch_chain_matmul
      res2_diag = torch_diagonal(res2,dim1 = 3, dim2 = 4)
      res2_diag = res2_diag[,,,NULL] # adding a 4th dimension to the diagonal array to be compatible with res1
      mu_temp = res1 + res2_diag
      ## testing mu_computation
      # t(as.matrix(y_mat_stack[1,1,1,]))%*%as.matrix(Xk_stack[1,1])%*%as.matrix(beta_stack[1,,]) + t(as.matrix(y_mat_stack[1,1,1,]))%*%as.matrix(theta_stack[1,,])%*% (as.matrix(y_mat_stack[1,1,1,]))
      # prices = Xk_stack[,,,1]
      # prices = prices[,,,NULL]
      p_temp = torch_matmul(y_mat_stack,prices)
      if(car) p_temp[,,1,] =0   # price of the outside option is zero
      
      
      # in budget?

      # computing car prices and the extra budget (bonus)
      if(car){  # if we are in the car block, bonuses are different for different combinations in y_mat and the vector of bonuses = vector of extraBi for the repeated measurement 
        # making extraBi a tensor and changing its dimensions to be compatible with p_temp (same applies to car_prices)
        bonus = bonuses_array
        car_prices = car_prices_array
        car_finmult = finmults_array
        car_budget = budgets_array
        
      }else{  # if we are in a non-car block, bonuses for different combinations are the same (all combinations have the same car), same applies to car_prices
        # bonus = compute_car_price_bonus_finmult_budget(car_idx_r,ncomb,extraBi_full,finmult_full,budget_full,prices_r,nprod)$bonus
        
        out = compute_car_price_bonus_finmult_budget(mat_idx1,car_idx,extraBi_full,finmult_full,budget_full,car_prices_full,nprod,t,nreg)
        bonus = out$bonus
        car_prices = out$car_prices
        car_finmult = out$car_finmult
        car_budget = out$car_budget
      }
      

      # accounting for the extra budget
      add_on_payable = torch_maximum(zero_double,p_temp - car_prices - bonus)  # deducts the bonus from the add_on_price (we use max because bonus can't be used for car)
      
      out_of_budget = (car_prices + add_on_payable) * car_finmult >= car_budget
      ##########################
      mu_in_budget = mu_temp - 1e10 * out_of_budget   # we decrease the utility out of budget options by 1e10 units (note that utility will be exponentiated hence not setting to zero)
      Pr_num = exp(mu_in_budget - torch_max(mu_in_budget,dim = 3,keepdim = T)[[1]])
      Pr_denom = torch_sum(Pr_num, dim = 3,keepdim = T)
      Pr_in_budget = Pr_num/Pr_denom 
      Pr_in_budget = Pr_in_budget/torch_sum(Pr_in_budget,dim = 3,keepdim = T)  #re-normalize the in_budget probs
      ##########################

      
      Pr_in_budget_mat = torch_reshape(torch_transpose(Pr_in_budget,1,2),shape = c(nreg*t,Pr_in_budget$shape[3]))
      choice = torch_reshape(torch_multinomial(Pr_in_budget_mat,num_samples = 1),shape = c(nreg*t))
      
      yk_stack[,,,block_obj[[kk]]$comb] = 0  # reset the yk elements associated to the current block
      yk_stack_mat = torch_reshape(torch_transpose(yk_stack,1,2), shape = c(nreg*t,yk_stack$shape[4]))

      if(car){
        car_idx = torch_tensor(choice,dtype = torch_long(),device = device) # updating car_idx  
        yk_stack_mat = torch_index_put(self = yk_stack_mat,indices = list(mat_idx1,car_idx),values = block_obj[[kk]]$one) # *** think about the types
        
      }else{  
        choice_idx = torch_tensor(torch_where(choice==1,1,choice-1),dtype = torch_long(),device = device) # subtract 1 unit from the inside choices
        yk_stack_mat = torch_index_put(self = yk_stack_mat, indices = list(mat_idx1,choice_idx + block_obj[[kk]]$start),values = torch_tensor(torch_where(choice==1,0,1),dtype = torch_double(),device = device)) # for choice = 1 (outside), choice_idx = 1 and the value will be = 0, for choice>1 (inside), choice idx =choice and value will be 1
      }
      yk_stack = torch_transpose(torch_reshape(yk_stack_mat,shape = c(nreg,t,1,yk_stack_mat$shape[2])),1,2)  
    } 
  yG = torch_index_put(self = yG, indices = list(Mvec_double[r],cube_idx1,cube_idx2), values = torch_reshape(yk_stack_mat, shape=c(nreg*t*yk_stack_mat$shape[2])))   # *** changed this???

  ##########################################
  # checking if choices in iteration r are in budget2 
  # note: car_idx is up to date so we can find the bonuses, prices and finance multipliers 
  out = compute_car_price_bonus_finmult_budget(mat_idx1,car_idx,extraBi_full,finmult_full,budget2_full,car_prices_full,nprod,t,nreg) 
  bonus = out$bonus
  car_prices = out$car_prices
  car_finmult = out$car_finmult
  car_budget = out$car_budget  
  
  yk_prices = torch_matmul(yk_stack,prices)
  
  # accounting for the extra budget
  add_on_payable = torch_maximum(zero_double,yk_prices - car_prices - bonus)  # deducts the bonus from the add_on_price (we use max because bonus can't be used for car)
  
  if(r==1){
    NC_t = ((car_prices + add_on_payable) * car_finmult <= car_budget)*1
    }else{
    NC_t = NC_t + ((car_prices + add_on_payable) * car_finmult <= car_budget)*1 # normalizing constant (ratio of choices within budget2) for each repeated measurement (*1 is to make the array numeric)  
  }
}
  
  NC_t = (-1*(-1)^budget_indicator) * (log(NC_t) - log(M))  # this term will be -log(NC_t) + log(M) for choices where candidate budget is higher than current budget
  NC = torch_sum(NC_t,dim = 1)  # normalizing constant for each individual
 
  return(NC)
} 
