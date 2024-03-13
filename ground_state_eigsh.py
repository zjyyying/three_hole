import subprocess
import os
import sys
import time
import shutil
import math
import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg
import time

import parameters as pam
import hamiltonian as ham
import lattice as lat
import variational_space as vs 
import utility as util


def reorder_z(slabel):
    '''
    reorder orbs such that d orb is always before p orb and Ni layer (z=1) before Cu layer (z=0)
    '''
    s1 = slabel[0]; orb1 = slabel[1]; x1 = slabel[2]; y1 = slabel[3]; z1 = slabel[4];
    s2 = slabel[5]; orb2 = slabel[6]; x2 = slabel[7]; y2 = slabel[8]; z2 = slabel[9];
    
    state_label = [s1,orb1,x1,y1,z1,s2,orb2,x2,y2,z2]
    
    if orb1 in pam.Ni_orbs and orb2 in pam.Ni_orbs:
        if x2<x1:
            state_label = [s2,orb2,x2,y2,z2,s1,orb1,x1,y1,z1]
        elif x2==x1 and orb1=='dx2y2' and orb2=='d3z2r2':
            state_label = [s2,orb2,x2,y2,z2,s1,orb1,x1,y1,z1]          
           
    elif orb1 in pam.O_orbs and orb2 in pam.Ni_orbs:
        state_label = [s2,orb2,x2,y2,z2,s1,orb1,x1,y1,z1]
        
    elif orb1 in pam.O_orbs and orb2 in pam.O_orbs:
        if z2>z1:
            state_label = [s2,orb2,x2,y2,z2,s1,orb1,x1,y1,z1]
            
    elif orb1 in pam.H_orbs and orb2 in pam.Ni_orbs:
        state_label = [s2,orb2,x2,y2,z2,s1,orb1,x1,y1,z1]  
        
    elif orb1 in pam.O_orbs and orb2 in pam.H_orbs:
        state_label = [s2,orb2,x2,y2,z2,s1,orb1,x1,y1,z1]           
            
    return state_label
                
def make_z_canonical(slabel):
    
    s1 = slabel[0]; orb1 = slabel[1]; x1 = slabel[2]; y1 = slabel[3]; z1 = slabel[4];
    s2 = slabel[5]; orb2 = slabel[6]; x2 = slabel[7]; y2 = slabel[8]; z2 = slabel[9];
    s3 = slabel[10]; orb3 = slabel[11]; x3 = slabel[12]; y3 = slabel[13]; z3 = slabel[14];  
    '''
    For three holes, the original candidate state is c_1*c_2*c_3|vac>
    To generate the canonical_state:
    1. reorder c_1*c_2 if needed to have a tmp12;
    2. reorder tmp12's 2nd hole part and c_3 to have a tmp23;
    3. reorder tmp12's 1st hole part and tmp23's 1st hole part
    '''
    tlabel = [s1,orb1,x1,y1,z1,s2,orb2,x2,y2,z2]
    tmp12 = reorder_z(tlabel)

    tlabel = tmp12[5:10]+[s3,orb3,x3,y3,z3]
    tmp23 = reorder_z(tlabel)

    tlabel = tmp12[0:5]+tmp23[0:5]
    tmp = reorder_z(tlabel)

    slabel = tmp+tmp23[5:10]
   
                
    return slabel




def get_ground_state(matrix, VS, S_val,Sz_val):  
    '''
    Obtain the ground state info, namely the lowest peak in Aw_dd's component
    in particular how much weight of various d8 channels: a1^2, b1^2, b2^2, e^2
    '''        
    t1 = time.time()
    print ('start getting ground state')
#     # in case eigsh does not work but matrix is actually small, e.g. Mc=1 (CuO4)
#     M_dense = matrix.todense()
#     #print 'H='
#     #print M_dense
    
#     for ii in range(0,1325):
#         for jj in range(0,1325):
#             if M_dense[ii,jj]>0 and ii!=jj:
#                 print ii,jj,M_dense[ii,jj]
#             if M_dense[ii,jj]==0 and ii==jj:
#                 print ii,jj,M_dense[ii,jj]
                    
                
#     vals, vecs = np.linalg.eigh(M_dense)
#     vals.sort()
#     print 'lowest eigenvalue of H from np.linalg.eigh = '
#     print vals
    
    # in case eigsh works:
    Neval = pam.Neval
    vals, vecs = sps.linalg.eigsh(matrix, k=Neval, which='SA')
    vals.sort()
    print ('lowest eigenvalue of H from np.linalg.eigsh = ')
    print (vals)
    
    print("---get_ground_state_eigsh %s seconds ---" % (time.time() - t1))
    

    
    # get state components in GS and another 9 higher states; note that indices is a tuple
    for k in range(0,2):
        #if vals[k]<pam.w_start or vals[k]>pam.w_stop:
        #if vals[k]<11.5 or vals[k]>14.5:
        #if k<Neval:
        #    continue
            
        print ('E',k,' = ', vals[k])
        indices = np.nonzero(abs(vecs[:,k])>0.05)
        
        # stores all weights for sorting later
        dim = len(vecs[:,k])
        allwgts = np.zeros(dim)
        allwgts = abs(vecs[:,k])**2
        ilead = np.argsort(-allwgts)   # argsort returns small value first by default
            
        wgt_d9d8 = np.zeros(20)
        wgt_d9Ld9 = np.zeros(20)  
        wgt_d8d9 = np.zeros(20) 
        wgt_d9d9L = np.zeros(20)         
        wgt_d9d9H = np.zeros(20)           
        wgt_d9H2 = np.zeros(20)   
        wgt_d9L2 = np.zeros(20)          
        total = 0
        total2 = 0
        print ("Compute the weights in GS (lowest Aw peak)")
        
        #for i in indices[0]:
        for i in range(dim):
            # state is original state but its orbital info remains after basis change
            istate = ilead[i]
            weight = allwgts[istate]
            
            #if weight>0.01:
            if total<0.999:
                total += weight
                
                state = VS.get_state(VS.lookup_tbl[istate])

                s1 = state['hole1_spin']
                s2 = state['hole2_spin']
                s3 = state['hole3_spin']
                orb1 = state['hole1_orb']
                orb2 = state['hole2_orb']
                orb3 = state['hole3_orb']
                x1, y1, z1 = state['hole1_coord']
                x2, y2, z2 = state['hole2_coord']
                x3, y3, z3 = state['hole3_coord']

                # also obtain the total S and Sz of the state
                S12  = S_val[istate]
                Sz12 = Sz_val[istate]
                
                slabel=[s1,orb1,x1,y1,z1,s2,orb2,x2,y2,z2,s3,orb3,x3,y3,z3]
                slabel= make_z_canonical(slabel)
                s1 = slabel[0]; orb1 = slabel[1]; x1 = slabel[2]; y1 = slabel[3]; z1 = slabel[4];
                s2 = slabel[5]; orb2 = slabel[6]; x2 = slabel[7]; y2 = slabel[8]; z2 = slabel[9];
                s3 = slabel[10]; orb3 = slabel[11]; x3 = slabel[12]; y3 = slabel[13]; z3 = slabel[14]                
                
                
                #if i in indices[0]:
                #if not(abs(x1)>1. or abs(y1)>1. or abs(x2)>1. or abs(y2)>1. or abs(x3)>1. or abs(y3)>1.):
                if weight>0.01:
                    print ('state ', istate, s1,orb1,x1,y1,z1,',',s2,orb2,x2,y2,z2,',',s3,orb3,x3,y3,z3, \
                       ', S=',S12,'Sz=',Sz12, ", weight = ", weight)
                    total2 += weight
                
                # analyze the states on Ni at (0,0,0)
                

                if orb1 in pam.Ni_orbs and orb2 in pam.Ni_orbs and orb3 in pam.Ni_orbs and x1==0 and x2==x3==2:
                    wgt_d9d8[0] += weight
                    if orb1=='dx2y2' and  orb2=='dx2y2'  and  orb3=='dx2y2':
                        wgt_d9d8[1] += weight
                        if S12==0:
                            wgt_d9d8[4] += weight
                    if orb1=='dx2y2' and  orb2=='d3z2r2'  and  orb3=='dx2y2':
                        wgt_d9d8[2] += weight          
                        if S12==1:
                            wgt_d9d8[5] += weight                        
                    if orb1=='d3z2r2' and  orb2=='dx2y2'  and  orb3=='dx2y2':
                        wgt_d9d8[3] += weight
                        if S12==1:
                            wgt_d9d8[6] += weight                           
                        
                if orb1 in pam.Ni_orbs and orb2 in pam.Ni_orbs and orb3 in pam.Ni_orbs and x1==x2==0 and x3==2:
                    wgt_d8d9[0] += weight
                    if orb1=='dx2y2' and  orb2=='dx2y2'  and  orb3=='dx2y2':
                        wgt_d8d9[1] += weight
                        if S12==0:
                            wgt_d8d9[4] += weight 
                    if orb1=='d3z2r2' and  orb2=='dx2y2'  and  orb3=='dx2y2':
                        wgt_d8d9[2] += weight          
                        if S12==1:
                            wgt_d8d9[5] += weight                             
                    if orb1=='d3z2r2' and  orb2=='d3z2r2'  and  orb3=='dx2y2':
                        wgt_d8d9[3] += weight          
                        if S12==0:
                            wgt_d8d9[6] += weight      
                            
                if orb1 in pam.Ni_orbs and orb2 in pam.Ni_orbs and orb3 in pam.O_orbs and x1==0 and x2==2 and ((x3==1 and y3==0) or (x3==-1 and y3==0) or (x3==0 and y3==-1) or (x3==0 and y3==1)):
                    wgt_d9Ld9[0] += weight                            
                    if orb1=='dx2y2' and  orb2=='dx2y2':
                        wgt_d9Ld9[1] += weight
                    if orb1=='d3z2r2' and  orb2=='dx2y2':
                        wgt_d9Ld9[2] += weight
                    if orb1=='dx2y2' and  orb2=='d3z2r2':
                        wgt_d9Ld9[3] += weight
                    if orb1=='d3z2r2' and  orb2=='d3z2r2':
                        wgt_d9Ld9[4] += weight      
     
                if orb1 in pam.Ni_orbs and orb2 in pam.Ni_orbs and orb3 in pam.O_orbs and x1==0 and x2==2 and ((x3==3 and y3==0) or (x3==2 and y3==-1) or (x3==2 and y3==1)):
                    wgt_d9d9L[0] += weight                            
                    if orb1=='dx2y2' and  orb2=='dx2y2':
                        wgt_d9d9L[1] += weight
                    if orb1=='d3z2r2' and  orb2=='dx2y2':
                        wgt_d9d9L[2] += weight
                    if orb1=='dx2y2' and  orb2=='d3z2r2':
                        wgt_d9d9L[3] += weight
                    if orb1=='d3z2r2' and  orb2=='d3z2r2':
                        wgt_d9d9L[4] += weight   
 
                if orb1 in pam.Ni_orbs and orb2 in pam.Ni_orbs and orb3 in pam.H_orbs and x1==0 and x2==2:
                    wgt_d9d9H[0] += weight                            
                    if orb1=='dx2y2' and  orb2=='dx2y2':
                        wgt_d9d9H[1] += weight
                    if orb1=='d3z2r2' and  orb2=='dx2y2':
                        wgt_d9d9H[2] += weight
                    if orb1=='dx2y2' and  orb2=='d3z2r2':
                        wgt_d9d9H[3] += weight
                    if orb1=='d3z2r2' and  orb2=='d3z2r2':
                        wgt_d9d9H[4] += weight   
                                                                                                       
                if orb1 in pam.Ni_orbs and orb2 in pam.H_orbs and orb3 in pam.H_orbs :
                    wgt_d9H2[0] += weight        
                    
                if orb1 in pam.Ni_orbs and orb2 in pam.O_orbs and orb3 in pam.O_orbs :
                    wgt_d9L2[0] += weight                            
                    
                                                                                                     
        print('printed states total weight =', total)
        
        print('wgt_d9d8 = ',wgt_d9d8[0])
        print('wgt_d8d9= ',wgt_d8d9[0])
        print('wgt_d9Ld9 = ',wgt_d9Ld9[0])
        print('wgt_d9d9L = ',wgt_d9d9L[0])
        print('wgt_d9d9H = ',wgt_d9d9H[0])
        print('wgt_d9H2 = ',wgt_d9H2[0])
        print('wgt_d9L2 = ',wgt_d9L2[0])
        print('total weight = ', wgt_d9d8[0]+ wgt_d8d9[0]+wgt_d9Ld9[0]+ wgt_d9d9L[0]+ wgt_d9d9H[0]+wgt_d9H2[0]+wgt_d9L2[0])
        print('total2 = ',total2)  
        
        
        
        path = './data'		# create file

        if os.path.isdir(path) == False:
            os.mkdir(path) 
        txt=open('./data/wgt_d9d8','a')                                  
        txt.write(str(wgt_d9d8[0])+'\n')
        txt.close() 
        txt=open('./data/wgt_d9d8_b1b1b1','a')                                  
        txt.write(str(wgt_d9d8[1])+'\n')
        txt.close()         
        txt=open('./data/wgt_d9d8_b1a1b1','a')                                  
        txt.write(str(wgt_d9d8[2])+'\n')
        txt.close()              
        txt=open('./data/wgt_d9d8_a1b1b1','a')                                  
        txt.write(str(wgt_d9d8[3])+'\n')
        txt.close()        
        txt=open('./data/wgt_d9d8_b1b1b1_0','a')                                  
        txt.write(str(wgt_d9d8[4])+'\n')
        txt.close()         
        txt=open('./data/wgt_d9d8_b1a1b1_1','a')                                  
        txt.write(str(wgt_d9d8[5])+'\n')
        txt.close()              
        txt=open('./data/wgt_d9d8_a1b1b1_1','a')                                  
        txt.write(str(wgt_d9d8[6])+'\n')
        txt.close()               
        

        txt=open('./data/wgt_d8d9','a')                                  
        txt.write(str(wgt_d8d9[0])+'\n')
        txt.close()      
        txt=open('./data/wgt_d8d9_b1b1b1','a')                                  
        txt.write(str(wgt_d8d9[1])+'\n')
        txt.close()    
        txt=open('./data/wgt_d8d9_a1b1b1','a')                                  
        txt.write(str(wgt_d8d9[2])+'\n')
        txt.close()            
        txt=open('./data/wgt_d8d9_a1a1b1','a')                                  
        txt.write(str(wgt_d8d9[3])+'\n')
        txt.close()            
        txt=open('./data/wgt_d8d9_b1b1b1_0','a')                                  
        txt.write(str(wgt_d8d9[4])+'\n')
        txt.close()    
        txt=open('./data/wgt_d8d9_a1b1b1_1','a')                                  
        txt.write(str(wgt_d8d9[5])+'\n')
        txt.close()            
        txt=open('./data/wgt_d8d9_a1a1b1_0','a')                                  
        txt.write(str(wgt_d8d9[6])+'\n')
        txt.close()                         
        
        
        
        
        txt=open('./data/wgt_d9Ld9','a')                                  
        txt.write(str(wgt_d9Ld9[0])+'\n')
        txt.close()    
        txt=open('./data/wgt_d9Ld9_b1b1','a')                                  
        txt.write(str(wgt_d9Ld9[1])+'\n')
        txt.close()           
        txt=open('./data/wgt_d9Ld9_a1b1','a')                                  
        txt.write(str(wgt_d9Ld9[2])+'\n')
        txt.close()      
        txt=open('./data/wgt_d9Ld9_b1a1','a')                                  
        txt.write(str(wgt_d9Ld9[3])+'\n')
        txt.close()              
        txt=open('./data/wgt_d9Ld9_a1a1','a')                                  
        txt.write(str(wgt_d9Ld9[4])+'\n')
        txt.close()                  
        

        
        txt=open('./data/wgt_d9d9L','a')                                  
        txt.write(str(wgt_d9d9L[0])+'\n')
        txt.close()  
        txt=open('./data/wgt_d9d9L_b1b1','a')                                  
        txt.write(str(wgt_d9d9L[1])+'\n')
        txt.close()           
        txt=open('./data/wgt_d9d9L_a1b1','a')                                  
        txt.write(str(wgt_d9d9L[2])+'\n')
        txt.close()      
        txt=open('./data/wgt_d9d9L_b1a1','a')                                  
        txt.write(str(wgt_d9d9L[3])+'\n')
        txt.close()              
        txt=open('./data/wgt_d9d9L_a1a1','a')                                  
        txt.write(str(wgt_d9d9L[4])+'\n')
        txt.close()          
        
        
        
        txt=open('./data/wgt_d9d9H','a')                                  
        txt.write(str(wgt_d9d9H[0])+'\n')
        txt.close()   
        txt=open('./data/wgt_d9d9H_b1b1','a')                                  
        txt.write(str(wgt_d9d9H[1])+'\n')
        txt.close()           
        txt=open('./data/wgt_d9d9H_a1b1','a')                                  
        txt.write(str(wgt_d9d9H[2])+'\n')
        txt.close()      
        txt=open('./data/wgt_d9d9H_b1a1','a')                                  
        txt.write(str(wgt_d9d9H[3])+'\n')
        txt.close()              
        txt=open('./data/wgt_d9d9H_a1a1','a')                                  
        txt.write(str(wgt_d9d9H[4])+'\n')
        txt.close()
        
        
        
        txt=open('./data/wgt_d9H2','a')                                  
        txt.write(str(wgt_d9H2[0])+'\n')
        txt.close()           
        txt=open('./data/wgt_d9L2','a')                                  
        txt.write(str(wgt_d9L2[0])+'\n')
        txt.close()           
        
        
        
 
                                                                                              
        

    return vals #, vecs, wgt_d8, wgt_d9L, wgt_d10L2
