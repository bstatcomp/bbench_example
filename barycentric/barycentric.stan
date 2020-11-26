functions {
  real calc_th(vector lam, real ttime, vector tk, vector w) {
    
    int N = num_elements(lam);
    
    vector[N] a1 = rows_dot_product(w, lam);
    vector[N] a2 = 1 ./ (ttime - tk);
    return sum(rows_dot_product(a1, a2)) / sum(rows_dot_product(w, a2));
  
  }
}


data {
  int<lower=2> N_teams;     // number of teams
  int<lower=1> N_train;     // number of train games
  int<lower=1> N_test;      // number of test games
  
  int<lower=1> n_nodes;     // number of nodes
  vector[n_nodes] t_k;      // node times
  vector[n_nodes] w;        // weights
  
  int<lower=1> N_tt;        // unique sorted concat(train time, test time ) points
  real tt[N_tt];
  
  int<lower=1> N_tt_train;  // unique sorted train time points
  real tt_train[N_tt_train];
  
  int i_train[N_train,2];   // team indices
  int i_test[N_test,2];
  int idx_train[N_train];
  int idx_test[N_test];
  
  int y[N_train];           // bernoulli observations
}

parameters {
  vector[n_nodes] lambda[N_teams - 1];
  real hta;
}

transformed parameters {
  vector[N_tt_train] theta[N_teams];
  
  theta[N_teams] = rep_vector(0, N_tt_train);
  for (i in 1:(N_teams - 1)) {
    for (j in 1:N_tt_train) {
      theta[i][j] = calc_th(lambda[i], tt_train[j], t_k, w);
    }
  }

}

model {
  
  for (i in 1:(N_teams - 1)) {
    lambda[i] ~ normal(0, 2);
  }
  
  hta ~ normal(0, 1);
  
  for (i in 1:N_train) {
    y[i] ~ bernoulli(inv_logit(theta[i_train[i, 1]][idx_train[i]] + hta
                             - theta[i_train[i, 2]][idx_train[i]]));
  }
  
}

generated quantities {

  vector[N_tt] th[N_teams];
  vector[N_test] p;
  vector[N_train] log_lik;

  th[N_teams] = rep_vector(0, N_tt);
  
  for (i in 1:(N_teams - 1)) {
    for (j in 1:N_tt) {
      th[i][j] = calc_th(lambda[i], tt[j], t_k, w);
    }
  }
  
  for (i in 1:N_test) {
    p[i] = inv_logit(th[i_test[i, 1]][idx_test[i]] + hta
                   - th[i_test[i, 2]][idx_test[i]]);
  }
  
  for (i in 1:N_train) {
    log_lik[i] = bernoulli_lpmf(y[i] | inv_logit(theta[i_train[i, 1]][idx_train[i]]
                                               + hta
                                               - theta[i_train[i, 2]][idx_train[i]]));
  }
  
}
