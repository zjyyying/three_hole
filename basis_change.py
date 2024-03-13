import variational_space as vs
import lattice as lat
import bisect
import numpy as np
import scipy.sparse as sps
import parameters as pam

def find_singlet_triplet_partner_d_double(VS, d_part, index, h3_part):
    '''
    For a given state find its partner state to form a singlet/triplet.
    Right now only applied for d_double states
    
    Note: idx is to label which hole is not on Ni

    Returns
    -------
    index: index of the singlet/triplet partner state in the VS
    phase: phase factor with which the partner state needs to be multiplied.
    '''
    if index==1:
        slabel = h3_part + ['up']+d_part[6:10] + ['dn']+d_part[1:5]
    elif index==2:
        slabel = ['up']+d_part[6:10] + h3_part + ['dn']+d_part[1:5]
    elif index==3:
        slabel = ['up']+d_part[6:10] + ['dn']+d_part[1:5] + h3_part 
                 
    #print('original state=', estate + ['up']+d_part[1:5] + ['dn']+d_part[6:10] + h3_part)
    
    tmp_state = vs.create_three_hole_state(slabel)
    partner_state,_ = vs.make_state_canonical(tmp_state)
    phase = -1.0
    state_id = VS.get_index(partner_state)
    
#     if state_id==None:
#         print(slabel)
#         tstate = partner_state
#         tse = tstate['e_spin']
#         ts1 = tstate['hole1_spin']
#         ts2 = tstate['hole2_spin']
#         ts3 = tstate['hole3_spin']
#         torbe = tstate['e_orb']
#         torb1 = tstate['hole1_orb']
#         torb2 = tstate['hole2_orb']
#         torb3 = tstate['hole3_orb']
#         txe, tye, tze = tstate['e_coord']
#         tx1, ty1, tz1 = tstate['hole1_coord']
#         tx2, ty2, tz2 = tstate['hole2_coord']
#         tx3, ty3, tz3 = tstate['hole3_coord']
#         print ('Error state', tse,torbe,txe,tye,tze,ts1,torb1,tx1,ty1,tz1,ts2,torb2,tx2,ty2,tz2,ts3,torb3,tx3,ty3,tz3)
#         print (VS.get_index(tstate))
#         return
    
    return state_id, phase


def create_singlet_triplet_basis_change_matrix_d_double(VS, d_double, double_part, idx, hole3_part):
    '''
    Similar to above create_singlet_triplet_basis_change_matrix but only applies
    basis change for d_double states
    
    Note that for three hole state, its partner state must have exactly the same
    spin and positions of L and Nd-electron
    
    This function is required for create_interaction_matrix_ALL_syms !!!
    '''
    data = []
    row = []
    col = []
    
    count_singlet = 0
    count_triplet = 0
    
    # store index of partner state in d_double to avoid double counting
    # otherwise, when arriving at i's partner j, its partner would be i
    count_list = []
    
    # denote if the new state is singlet (0) or triplet (1)
    S_val  = np.zeros(VS.dim, dtype=int)
    Sz_val = np.zeros(VS.dim, dtype=int)
    AorB_sym = np.zeros(VS.dim, dtype=int)
    
    # first set the matrix to be identity matrix (for states not d_double)
    for i in range(VS.dim):
        if i not in d_double:
            data.append(np.sqrt(2.0)); row.append(i); col.append(i)
     
    #################################################################################
    for i, double_id in enumerate(d_double):
        s1 = double_part[i][0]
        o1 = double_part[i][1]
        s2 = double_part[i][5]
        o2 = double_part[i][6]
        dpos = double_part[i][2:5]
        
        if s1==s2:
            # must be triplet
            # see case 2 of make_state_canonical in vs.py, namely
            # for same spin states, always order the orbitals
            S_val[double_id] = 1
            data.append(np.sqrt(2.0));  row.append(double_id); col.append(double_id)
            if s1=='up':
                Sz_val[double_id] = 1
            elif s1=='dn':
                Sz_val[double_id] = -1
            count_triplet += 1

        elif s1=='dn' and s2=='up':
            print ('Error: d_double cannot have states with s1=dn, s2=up !')
            tstate = VS.get_state(VS.lookup_tbl[i])
            ts1 = tstate['hole1_spin']
            ts2 = tstate['hole2_spin']
            ts3 = tstate['hole3_spin']
            torb1 = tstate['hole1_orb']
            torb2 = tstate['hole2_orb']
            torb3 = tstate['hole3_orb']
            tx1, ty1, tz1 = tstate['hole1_coord']
            tx2, ty2, tz2 = tstate['hole2_coord']
            tx3, ty3, tz3 = tstate['hole3_coord']
            print ('Error state', i, ts1,torb1,tx1,ty1,ts2,torb2,tx2,ty2,ts3,torb3,tx3,ty3)
            break

        elif s1=='up' and s2=='dn':
            if o1==o2:               
                # get state as (e1e1 +- e2e2)/sqrt(2) for A and B sym separately 
                # instead of e1e1 and e2e2
                if o1!='dxz' and o1!='dyz':
                    data.append(np.sqrt(2.0));  row.append(double_id); col.append(double_id)
                    S_val[double_id]  = 0
                    Sz_val[double_id] = 0
                    count_singlet += 1
                    
                elif o1=='dxz':  # no need to consider e2='dyz' case
                    # generate paired e2e2 state:
                    if idx[i]==3:
                        slabel = [s1,'dyz']+dpos + [s2,'dyz']+dpos + hole3_part[i]
                    elif idx[i]==2:
                        slabel = [s1,'dyz']+dpos + hole3_part[i] + [s2,'dyz']+dpos 
                    elif idx[i]==1:
                        slabel = hole3_part[i] + [s1,'dyz']+dpos + [s2,'dyz']+dpos
                        
                    tmp_state = vs.create_three_hole_state(slabel)
                    new_state,_ = vs.make_state_canonical(tmp_state)
                    e2 = VS.get_index(new_state)
                        
                    data.append(1.0);  row.append(double_id);  col.append(double_id)
                    data.append(1.0);  row.append(e2); col.append(double_id)
                    AorB_sym[double_id]  = 1
                    S_val[double_id]  = 0
                    Sz_val[double_id] = 0
                    count_singlet += 1
                    data.append(1.0);  row.append(double_id);  col.append(e2)
                    data.append(-1.0); row.append(e2); col.append(e2)
                    AorB_sym[e2] = -1
                    S_val[e2]  = 0
                    Sz_val[e2] = 0
                    count_singlet += 1

            else:
                if double_id not in count_list:
                    # debug:
#                     tstate = VS.get_state(VS.lookup_tbl[double_id])
#                     tse = tstate['e_spin']
#                     ts1 = tstate['hole1_spin']
#                     ts2 = tstate['hole2_spin']
#                     ts3 = tstate['hole3_spin']
#                     torbe = tstate['e_orb']
#                     torb1 = tstate['hole1_orb']
#                     torb2 = tstate['hole2_orb']
#                     torb3 = tstate['hole3_orb']
#                     txe, tye, tze = tstate['e_coord']
#                     tx1, ty1, tz1 = tstate['hole1_coord']
#                     tx2, ty2, tz2 = tstate['hole2_coord']
#                     tx3, ty3, tz3 = tstate['hole3_coord']
#                     print ('state needing partner', tse,torbe,txe,tye,tze,ts1,torb1,tx1,ty1,tz1,ts2,torb2,tx2,ty2,tz2,ts3,torb3,tx3,ty3,tz3)
#                     print (VS.get_index(tstate), double_id)
                    
#                     print('estate=',estate)
                    
                    j, ph = find_singlet_triplet_partner_d_double(VS, double_part[i], idx[i], hole3_part[i])

                    # append matrix elements for singlet states
                    # convention: original state col i stores singlet and 
                    #             partner state col j stores triplet
                    data.append(1.0);  row.append(double_id); col.append(double_id)
                    data.append(-ph);  row.append(j); col.append(double_id)
                    S_val[double_id]  = 0
                    Sz_val[double_id] = 0

#                     print ("partner states:", i,j)
#                     print ("state i = ", s1, o1, s2, o2)
#                     print ("state j = ",'up',o2,'dn',o1)

                    # append matrix elements for triplet states
                    data.append(1.0);  row.append(double_id); col.append(j)
                    data.append(ph);   row.append(j); col.append(j)
                    S_val[j]  = 1
                    Sz_val[j] = 0

                    count_list.append(j)

                    count_singlet += 1
                    count_triplet += 1
    
#     print(row)
#     print(col)
#     print(data)
    
    return sps.coo_matrix((data,(row,col)),shape=(VS.dim,VS.dim))/np.sqrt(2.0), S_val, Sz_val, AorB_sym


def print_VS_after_basis_change(VS,S_val,Sz_val):
    print ('print_VS_after_basis_change:')
    for i in range(0,VS.dim):
        state = VS.get_state(VS.lookup_tbl[i])
        ts1 = state['hole1_spin']
        ts2 = state['hole2_spin']
        torb1 = state['hole1_orb']
        torb2 = state['hole2_orb']
        tx1, ty1, tz1 = state['hole1_coord']
        tx2, ty2, tz2 = state['hole2_coord']
        #if ts1=='up' and ts2=='up':
        if torb1=='dx2y2' and torb2=='px':
            print (i, ts1,torb1,tx1,ty1,tz1,ts2,torb2,tx2,ty2,tz2,'S=',S_val[i],'Sz=',Sz_val[i])
            
