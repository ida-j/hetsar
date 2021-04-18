# def get_answer():
#     """Get an answer."""
#     return True

import numpy as np

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
