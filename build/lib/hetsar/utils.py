import numpy as np
from scipy.optimize import minimize
import pandas as pd



def fn_inv_partitioned_a(invA,m_B,m_C,m_D):
    m_C_invA = m_C@invA
    m_E = m_D - m_C_invA@m_B
    invE = np.linalg.inv(m_E)
    m_invA_B_invE = invA@m_B@invE

    invH11 = invA + (m_invA_B_invE @ m_C_invA);
    invH12 = -m_invA_B_invE;
    invH21 = -invE @ m_C_invA;
    invH22 = invE;

    row1 = np.concatenate((invH11, invH12),axis = 1)
    row2 = np.concatenate((invH21, invH22),axis = 1)
    invH = np.concatenate((row1,row2),axis = 0)

    return invH

def fn_inv_partitioned_b(m_A,m_B,m_C,invD):
    m_B_invD = m_B @ invD;
    m_F = m_A - (m_B_invD @ m_C);
    invF = np.linalg.inv(m_F);
    m_invD_C_invF = invD @ m_C @ invF;

    invH11 = invF;
    invH12 = -invF @ m_B_invD;
    invH21 = -m_invD_C_invF;
    invH22 = invD + (m_invD_C_invF @ m_B_invD);

    row1 = np.concatenate((invH11, invH12),axis = 1)
    row2 = np.concatenate((invH21, invH22),axis = 1)
    invH = np.concatenate((row1,row2),axis = 0)

    return invH

def fn_varml_sandwich_Npsi_NKbeta_Nsgmsq(v_theta,m_y,m_ys,a_x,m_W):
    N,T,K = a_x.shape
    v_psi = v_theta[:N,0].reshape(N,1)
    v_beta = v_theta[N:(N+K*N),0].reshape(N*K,1)
    v_sgmsq = v_theta[(N+K*N):,0].reshape(N,1)
#     m_beta = v_beta.reshape([N,K], order = 'F')
    m_beta = v_beta.reshape([N,K], order = 'C')

    v_sgm4h = v_sgmsq**2
    v_sgm6h = v_sgmsq**3

    a_x2 = np.transpose(a_x,(0,2,1))
    m_beta2 = m_beta[:,:,np.newaxis]
    m_beta_x = m_beta2*a_x2
    m_beta_x2 = np.transpose(m_beta_x,(0,2,1))
    m_beta_x_sum = np.sum(m_beta_x2,2)

    # residuals
    m_eps = m_y-v_psi*m_ys-m_beta_x_sum
    m_epssq = m_eps**2
    v_ssr = np.sum(m_eps**2,1).reshape(N,1)
    sssr = np.sum(v_ssr/v_sgmsq)
    m_Psi = v_psi * np.identity(len(v_psi))
    m_A = np.identity(N)-m_Psi@m_W
    det_mA= np.linalg.det(m_A)
    if det_mA<=0:
        print('Error: determinant(A)<=0!')


    m_Q = m_W@np.linalg.inv(m_A)

    m_H11 = m_Q*np.transpose(m_Q) + np.diag(np.sum(m_ys**2,1))/(T*v_sgmsq)
    m_H13a = np.sum(m_ys*m_eps,1).reshape(N,1)/(T*v_sgm4h)
    m_H13 = m_H13a*np.identity(len(m_H13a))
    m_H33a = -(0.5/v_sgm4h)+(v_ssr/v_sgm6h)/T
    m_H33 = m_H33a*np.identity(len(m_H33a))

    m_H12 = np.zeros([N,N*K])
    invH22 = np.zeros([N*K,N*K])
    m_H23 = np.zeros([N*K,N])

    for i in range(N):
        ind = (i * K + 1,(i+1) * K)
        v_ysi = m_ys[i,:].reshape(T,1)
        m_Xi = a_x[i,:,:] # TxK
        v_epsi = m_eps[i,:].reshape(T,1)
        sgmsqi = v_sgmsq[i,0]
        sgm4hi = v_sgm4h[i,0]

        m_H12[i,ind[0]-1:ind[1]] = np.transpose(v_ysi)@m_Xi/sgmsqi/T
        invH22[ind[0]-1:ind[1],ind[0]-1:ind[1]] = np.linalg.inv(np.transpose(m_Xi)@m_Xi)*sgmsqi*T
        m_H23[ind[0]-1:ind[1],i] = np.transpose(np.transpose(m_Xi)@v_epsi/sgm4hi/T)

    m_Z11 = m_H11;
    m_Z12 = np.concatenate((m_H12,m_H13),axis = 1)
    invZ22 = fn_inv_partitioned_a(invH22,m_H23,np.transpose(m_H23),m_H33)
    invH = fn_inv_partitioned_b(m_Z11, m_Z12, np.transpose(m_Z12), invZ22)

    # J matrix
    v_q = np.diag(m_Q).reshape(N,1)
    m_dlogft_dvpsi = (m_ys*m_eps/v_sgmsq) - v_q
    v_dlogft_dvsgmsq = (m_epssq/v_sgm4h/2) - 0.5/v_sgmsq

    a_dlogft_dvbeta = m_eps.reshape(N,T,1)*a_x/v_sgmsq.reshape(N,1,1)

    m_dlogft_dvbeta = np.zeros([K*N,T])

    for i in range(N):
        ind = (i * K + 1,(i+1) * K)
        m_dlogft_dvbeta[ind[0]-1:ind[1],:] = np.transpose(a_dlogft_dvbeta[i,:,:])

    m_dlogft_dvtheta = np.concatenate((m_dlogft_dvpsi,m_dlogft_dvbeta,v_dlogft_dvsgmsq))

    m_J = (m_dlogft_dvtheta@np.transpose(m_dlogft_dvtheta))/T

    # standard variance
    v_var0 = np.diag(invH)/T
    v_var = v_var0.reshape(len(v_var0),1)
    m_variance = np.zeros([N,K+2])
    for i in [0,K+1]:
        m_variance[:,i] = v_var[i*N:(i+1)*N,0]
    for k_val in range(K):
        i = 1
        m_variance[:,i+k_val] = v_var[[j+k_val for j in range(i*N+i-1,(i+K)*N+i-1,K)],0]


    # sandwich variance
    m_invH_J_invH = invH@m_J@invH
    v_var0 = np.diag(m_invH_J_invH)/T
    v_var = v_var0.reshape(len(v_var0),1)
    m_sandwich = np.zeros([N,K+2])
    for i  in [0,K+1]:
        m_sandwich[:,i] = v_var[i*N:(i+1)*N,0]
    for k_val in range(K):
        i = 1
        m_sandwich[:,i+k_val] = v_var[[j+k_val for j in range(i*N+i-1,(i+K)*N+i-1,K)],0]

    return (m_variance,m_sandwich)

def format_output(res,N,T,K,var,var_sand,dep_var,exog_labels):
    res_psi = res.x[:N].reshape([N,1])
    res_beta = res.x[N:(K+1)*N].reshape([N,K],order = 'C')
    res_sigma = res.x[(K+1)*N:].reshape([N,1])
    data_r = np.concatenate([res_psi,res_beta,res_sigma],axis = 1)
    dim_exog = res_beta.shape[1]
    if exog_labels==None:
        exog_labels = ['x{}'.format(i) for i in range(dim_exog)]
    else:
        if len(exog_labels)!=dim_exog:
            print('Wrong number of labels for exogenous covariates, using default labels')
            exog_labels = ['x{}'.format(i) for i in range(dim_exog)]

    colnames = [f'W{dep_var}'] + exog_labels + ['sgmsq']
    df_r = pd.DataFrame(data=data_r,columns = colnames)
    
    for i in range(len(colnames)):
        df_r['var_{}'.format(colnames[i])] = var[:,i]
        df_r['var_sandw_{}'.format(colnames[i])] = var_sand[:,i]
    df_r.insert(0, 'i', [i for i in range(1,N+1)])
    return df_r



# mean-group estimator

def fn_mg_est(df_theta,var_hat,group):
    countN = df_theta[[group,var_hat]].groupby(group).count().reset_index().rename(columns = {var_hat:'N'})
    df_mg = df_theta[[var_hat,group]].groupby(group).mean().reset_index().rename(columns = {var_hat:'var_hat_mg'})
    df_est2 = df_theta[[var_hat,group]].merge(df_mg[[group,'var_hat_mg']],on = group,how = 'left').\
    rename(columns = {var_hat:'var_hat'})

    df_est2['sq_er'] = (df_est2.var_hat-df_est2.var_hat_mg)**2
    df_sgm = df_est2[[group,'sq_er']].groupby(group).sum().reset_index().\
    merge(countN,on = group)
    df_sgm['s_{}_mg'.format(var_hat)] = np.sqrt(df_sgm.sq_er/(df_sgm.N*(df_sgm.N-1)))
    return df_sgm.merge(df_mg[['var_hat_mg',group]],on = group).\
    rename(columns = {'var_hat_mg':'{}_mg'.format(var_hat)})[[group,'s_{}_mg'.format(var_hat),'{}_mg'.format(var_hat)]]



def fn_mg_bias_rmse_size(df_results,var_hat,var0,N,cval = 1.96):
    df_est = df_results[[var_hat,'r','N']].rename(columns = {var_hat:'var_hat'})
    df_est['var0'] = var0
    res_mean = df_est[['var_hat','var0','r']].groupby('r').mean().reset_index()
    res_mean['bias'] = res_mean['var_hat']-res_mean['var0']
    res_mean = res_mean.rename(columns = {'var_hat':'var_hat_mg'})
    res_mean['rmse'] = (res_mean['var_hat_mg']-res_mean['var0'])**2
    bias_r = res_mean.mean().bias
    rmse_r = (res_mean.mean().rmse)**(1/2)
    df_est2 = df_est.merge(res_mean[['r','var_hat_mg']],on = 'r',how = 'left')
    df_est2['sq_er'] = (df_est2.var_hat-df_est2.var_hat_mg)**2
    df_sgm = df_est2[['r','sq_er']].groupby('r').sum().reset_index()
    df_sgm['s2_r'] = df_sgm.sq_er/(N*(N-1))
    df_sgm['s'] = np.sqrt(df_sgm.s2_r)
    df_sgm2 = df_sgm.merge(res_mean[['r','var_hat_mg']],on = 'r',how = 'left')
    df_sgm2['var0'] = var0
    df_sgm2['t']= (df_sgm2.var_hat_mg-df_sgm2.var0)/df_sgm2.s
    df_sgm2['size'] = 1*(np.abs(df_sgm2.t)>cval)
    size_r = df_sgm2.mean()['size']
    
    return (bias_r,rmse_r,size_r)
