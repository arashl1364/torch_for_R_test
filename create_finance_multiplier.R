create_finmult = function(){
  
  # create the finance multiplier 
  
  #fin3: 3 years (36 months), 0% interest
  #fin4: 4 years (48 months), 2.99% interest
  #fin5: 5 years (60 months), 5.99% interest
  #fin7: 7 years (84 months), 8.99% interest
  
  #e.g., 
  # int_rate = 0.0599
  # term = 60
  # car_tot_price = 20000
  # monthly_rate = car_tot_price*int_rate/12/(1/(1+int_rate/12)^term-1)*(-1)
  finmult = rep(0,4)
  int_rate = rep(0,4)
  term = rep(0,4)
  
  int_rate = c(0, 0.0299, 0.0599, 0.0899)
  term = c(36,48,60,84)
  for(i in 1:4){
    if(i==1){ finmult[i] = 1/term[i] }else{
      finmult[i] = int_rate[i]/12/(1/(1+int_rate[i]/12)^term[i]-1)*(-1)    
    }
  }
  return(finmult)
}
