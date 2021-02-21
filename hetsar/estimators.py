import numpy as np
from .utils import fn_inv_partitioned_a, fn_inv_partitioned_b, fn_varml_sandwich_Npsi_NKbeta_Nsgmsq, format_output

def f(v_theta0,args):
    v_theta = v_theta0.reshape(len(v_theta0),1)
    m_y,m_ys,a_x,m_W = args
    N,T,K = a_x.shape
    v_psi = v_theta[:N,0].reshape(N,1)
    v_beta = v_theta[N:(N+K*N),0].reshape(N*K,1)
    v_sgmsq = v_theta[(N+K*N):,0].reshape(N,1)
    m_beta = v_beta.reshape([N,K])

    v_sgm4h = v_sgmsq**2
    v_sgm6h = v_sgmsq**3
    
    a_x2 = np.transpose(a_x,(0,2,1))
    m_beta2 = m_beta[:,:,np.newaxis]
    m_beta_x = m_beta2*a_x2
    m_beta_x2 = np.transpose(m_beta_x,(0,2,1))
    m_beta_x_sum = np.sum(m_beta_x2,2)

    m_eps = m_y-v_psi*m_ys-m_beta_x_sum
    v_ssr = np.sum(m_eps**2,1).reshape(N,1)
    sssr = np.sum(v_ssr/v_sgmsq)
    m_Psi = v_psi * np.identity(len(v_psi))
    m_A = np.identity(N)-m_Psi@m_W
    det_mA= np.linalg.det(m_A)
    if det_mA<=0:
        print('Error: determinant(A)<=0!')
    constant = np.log(2*np.pi)*N/2
    first_part = -np.log(det_mA)
    second_part = np.sum(np.log(v_sgmsq))/2
    third_part = sssr/(2*T)
    return constant + first_part+second_part+third_part

def gradient(v_theta0,args):
    v_theta = v_theta0.reshape(len(v_theta0),1)
    m_y,m_ys,a_x,m_W = args
    N,T,K = a_x.shape
    v_psi = v_theta[:N,0].reshape(N,1)
    v_beta = v_theta[N:(N+K*N),0].reshape(N*K,1)
    v_sgmsq = v_theta[(N+K*N):,0].reshape(N,1)
    m_beta = v_beta.reshape([N,K])

    v_sgm4h = v_sgmsq**2
    v_sgm6h = v_sgmsq**3
    a_x2 = np.transpose(a_x,(0,2,1))
    m_beta2 = m_beta[:,:,np.newaxis]
    m_beta_x = m_beta2*a_x2
    m_beta_x2 = np.transpose(m_beta_x,(0,2,1))
    m_beta_x_sum = np.sum(m_beta_x2,2)

    m_eps = m_y-v_psi*m_ys-m_beta_x_sum
    v_ssr = np.sum(m_eps**2,1).reshape(N,1)
    m_Psi = v_psi * np.identity(len(v_psi))
    m_A = np.identity(N)-m_Psi@m_W
    m_Q = m_W@np.linalg.inv(m_A)
    v_dphi_dvpsi = np.diag(m_Q).reshape(N,1)-np.sum(m_ys*m_eps,1).reshape(N,1)/v_sgmsq/T
    m_eps2 = m_eps[:,:,np.newaxis]
    a_x_times_eps = m_eps2*a_x
    m_x_times_eps = np.sum(a_x_times_eps,1)
    m_X_times_eps_divided_sgmsq = m_x_times_eps/v_sgmsq
    m_X_times_eps_divided_sgmsq_tr = np.transpose(m_X_times_eps_divided_sgmsq)
    v_dphi_dvbeta = -m_X_times_eps_divided_sgmsq_tr.flatten('F')/T
    v_dphi_dvsgmsq = (0.5/v_sgmsq)-v_ssr/v_sgm4h/T/2
    grad = np.concatenate([v_dphi_dvpsi,v_dphi_dvbeta.reshape(N*K,1),v_dphi_dvsgmsq])
    return grad.reshape((K+2)*N,)

def hessian(v_theta0,args):
    v_theta = v_theta0.reshape(len(v_theta0),1)
    m_y,m_ys,a_x,m_W = args
    N,T,K = a_x.shape
    v_psi = v_theta[:N,0].reshape(N,1)
    v_beta = v_theta[N:(N+K*N),0].reshape(N*K,1)
    v_sgmsq = v_theta[(N+K*N):,0].reshape(N,1)
    m_beta = v_beta.reshape([N,K])

    v_sgm4h = v_sgmsq**2
    v_sgm6h = v_sgmsq**3
    a_x2 = np.transpose(a_x,(0,2,1))
    m_beta2 = m_beta[:,:,np.newaxis]
    m_beta_x = m_beta2*a_x2
    m_beta_x2 = np.transpose(m_beta_x,(0,2,1))
    m_beta_x_sum = np.sum(m_beta_x2,2)

    m_eps = m_y-v_psi*m_ys-m_beta_x_sum
    v_ssr = np.sum(m_eps**2,1).reshape(N,1)
    m_Psi = v_psi * np.identity(len(v_psi))
    m_A = np.identity(N)-m_Psi@m_W
    m_Q = m_W@np.linalg.inv(m_A)
    m_H11a = m_Q*np.transpose(m_Q)
    m_H11b = np.sum(m_ys**2,1).reshape(N,1)*np.identity(N)/v_sgmsq/T
    m_H11 = m_H11a+m_H11b
    m_H13 = np.sum(m_ys*m_eps,1).reshape(N,1)*np.identity(N)/v_sgm4h/T
    m_H33 = (-0.5/v_sgm4h+v_ssr/v_sgm6h/T)*np.identity(N)
    m_H12 = np.zeros([N,N*K])
    m_H22 = np.zeros([N*K,N*K])
    m_H23 = np.zeros([N*K,N])

    for i in range(N):
        ind = (i * K + 1,(i+1) * K)
        v_ysi = m_ys[i,:].reshape(T,1)
        m_Xi = a_x[i,:,:] # TxK
        v_epsi = m_eps[i,:].reshape(T,1)
        sgmsqi = v_sgmsq[i,0]
        sgm4hi = v_sgm4h[i,0]
        m_H12[i,ind[0]-1:ind[1]] = np.transpose(v_ysi)@m_Xi/sgmsqi/T
        m_H22[ind[0]-1:ind[1],ind[0]-1:ind[1]] = np.transpose(m_Xi)@m_Xi/sgmsqi/T
        m_H23[ind[0]-1:ind[1],i] = np.transpose(np.transpose(m_Xi)@v_epsi/sgm4hi/T)

    row1 = np.concatenate((m_H11,m_H12,m_H13),axis = 1)
    row2 = np.concatenate((m_H12.T,m_H22,m_H23),axis = 1)
    row3 = np.concatenate((m_H13.T,m_H23.T,m_H33),axis = 1)

    return np.concatenate((row1,row2,row3),axis = 0)

class Hetsar(object):
    def __init__(self):
        self.modelstring = 'hetsar'
        
    def fit(self,m_y,a_x,m_W,exog_labels = None): 
        N,T,K = a_x.shape
        m_ys = m_W@m_y
        args = (m_y,m_ys,a_x,m_W) 
        
        v_theta_ini = np.concatenate([np.zeros([(K+1)*N]),np.ones([N])])
        v_lb = -np.concatenate([0.995*np.ones(N),np.inf*np.ones(K*N),0.01*np.ones(N)])
        v_ub = np.concatenate([0.995*np.ones(N),np.inf*np.ones((K+1)*N)])
        bnds = tuple([(v_lb[i],v_ub[i]) for i in range(len(v_lb))])

        res = minimize(f, v_theta_ini, args=(args,),jac=gradient,
#                        hess=hessian,
                       bounds = bnds,method = 'L-BFGS-B',
                      options = {'maxiter':1000})
        self.res = res

        (var,var_sand) = fn_varml_sandwich_Npsi_NKbeta_Nsgmsq(res.x.reshape([N*(K+2),1]),m_y,m_ys,a_x,m_W)
        
        self.param_df = format_output(res,N,T,K,var,var_sand,exog_labels)
              
        return self
        
   