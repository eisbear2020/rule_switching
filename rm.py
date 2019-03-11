import numpy as np
import scipy.ndimage as nd

def clures_to_spkl(clu,res): # clu,res are already integers
    spkls=[[] for i in range(clu[0]+1)]
    i=0
    for re in res:
        i+=1
        spkls[clu[i]].append(int(re / 512 ))
    return spkls

def comp_speed(whl): #compute speed from the whl - use it for filtering!
    temp = np.zeros(whl.shape[0])
    speed = np.zeros(whl.shape[0])
    for i in range(5,whl.shape[0]-5):
        if whl[i,0] > 0 and whl[i+1,0] > 0 and whl[i,1] > 0 and whl[i+1,1] > 0:
            temp[i] = np.sqrt((whl[i,0] - whl[i+1,0])**2 + (whl[i,1] - whl[i+1,1])**2)
        else:
            temp[i]=-1
    
    for i in range(5,whl.shape[0]-5):
        if whl[i,0] > 0:
            t = temp[i-5:i+5]
            t = t[t>=0]
            if(len(t)>0):
                speed[i] = np.mean(t)
    return speed*39.0625

# better way of measuring the speed: linearized!
def dist(v,p):
    return np.sqrt((v[:,0]-p[0])**2 + (v[:,1]-p[1])**2)
def dist_clean(trial):
    trial[:,0] = np.interp(range(len(trial[:,0])), np.where(trial[:,0]>-1)[0],trial[trial[:,0]>-1,0])
    trial[:,1] = np.interp(range(len(trial[:,1])), np.where(trial[:,1]>-1)[0],trial[trial[:,1]>-1,1])
    di = dist(trial,(101,116))
    return nd.gaussian_filter1d(di,3)
def sp_ln(trial):
    sp=np.zeros(len(trial))
    for i in range(1,len(trial)-1):
        sp[i]=np.abs(trial[i+1]-trial[i-1])/2*39.0625
    return sp

def sn(h,k): #squared norm
    y = h - k
    return y.T[0]**2 + y.T[1]**2

def K(x,z,sig): #triweight kernel
    
    nss = 9*sig**2 #9 sigma squared
    snp = 1 - sn(x,z) / nss #1 - squared norm points / 9 sigma squared
    snp[snp<0] = 0
    
    return 4 / (np.pi * nss) * snp**3

def occmap(whl, speed=None,spf=None,sigma_ker=4.2): #range: 0 to 200
    if spf:
        whl = np.array(whl[speed > spf,:])
    occ = np.zeros((55,55))
    for i in range(55): #could be optimized...
        for j in range(55):
            occ[i,j] = np.sum(K(np.array((2 + i*4, 2 + j*4)),whl,sigma_ker))
    occ = occ / np.sum(occ) * whl.shape[0] * 0.0256  #try to be coherent with the time - in seconds!
    occ[occ<np.max(occ)/1000]=0 #cut out bins with low coverage, < 0.1% of peak
    return occ

def ratemap(occ, whl, spkl, speed, spf, sigma_ker=4.2, sigma_gauss=1):
    rate = np.zeros((55,55))
    if len(spkl) > 0:
        spkl = np.array(spkl[speed[spkl]>spf])
        spkp = np.array(whl[spkl,:])
        for i in range(55): #could be optimized...
            for j in range(55):
                if(occ[i,j] > 0): #avoid problems at locations with low coverage...
                    rate[i,j] = np.sum(K(np.array((2 + i*4, 2 + j*4)),spkp,sigma_ker)) / occ[i,j]
        if np.mean(rate[occ>0]) > 0:
            rate = rate/np.mean(rate[occ>0])*len(spkl)/len(whl) #be coherent with the real mean firing rate
        if sigma_gauss > 0:
            rate = nd.gaussian_filter(rate,sigma=sigma_gauss)
    return rate

# [(0, 'start'),
#  (1, 'centrebegin'),
#  (2, 'centreend'),
#  (3, 'goalbegin'),
#  (4, 'goalend'),
#  (5, 'startarm'),
#  (6, 'goalarm'),
#  (7, 'control'),
#  (8, 'lightarm'),
#  (9, 'ruletype'),
#  (10, 'errortrial')]

def select_tist(whl,timestamps,start=-1,goal=-1,rule=-1,err=-1):
    # select indexes that respect the given conditions (not all have to be given, if not present or -1 then it's excluded)
    index=np.zeros(len(whl),dtype=bool)
    c=conditions = np.array([start,goal,rule,err])
    for line in timestamps:
        cline=np.array([line[5],line[6],line[9],line[10]])
        if np.sum(c>=0)==0 or np.all(c[c>=0] == cline[c>=0]):
            index[line[0]:line[4]]=True
    return index

def occmap_tist(whl, speed,spf, timestamps,
                start=-1,goal=-1,rule=-1,err=-1,
                sigma_ker=4.2): #range: 0 to 200
    #True only when conditions are respected
    index_tist=select_tist(whl,timestamps,start,goal,rule,err)
    #speed filter
    index_tist[speed < spf] = False
    
    whl = np.array(whl[index_tist,:])
    occ = np.zeros((55,55))
    for i in range(55): #could be optimized...
        for j in range(55):
            occ[i,j] = np.sum(K(np.array((2 + i*4, 2 + j*4)),whl,sigma_ker))
    #occ = occ / np.sum(occ) * whl.shape[0] * 0.0256  #try to be coherent with the time - in seconds!
    occ[occ<np.max(occ)/1000]=0 #cut out bins with low coverage, < 0.1% of peak
    return occ

def ratemap_tist(occ, whl, spkl, speed, spf, timestamps,
                 start=-1,goal=-1,rule=-1,err=-1,
                 sigma_ker=4.2, sigma_gauss=1):
    
    rate = np.zeros((55,55))
    #True only when conditions are respected
    index_tist=select_tist(whl,timestamps,start,goal,rule,err)
    #speed filter
    index_tist[speed < spf] = False

    if len(spkl) > 0:
        spkl = np.array(spkl)
        spkl = spkl[index_tist[spkl]]
        spkp = np.array(whl[spkl,:])
        
        for i in range(55): #could be optimized...
            for j in range(55):
                if(occ[i,j] > 0): #avoid problems at locations with low coverage...
                    rate[i,j] = np.sum(K(np.array((2 + i*4, 2 + j*4)),spkp,sigma_ker)) / occ[i,j]
           
        if sigma_gauss > 0:
            rate = nd.gaussian_filter(rate,sigma=sigma_gauss)
            
    return rate, len(spkl)

def occmap_inds(whl, speed,spf, inds,
                sigma_ker=4.2):
    index_tist=inds
    #speed filter
    index_tist[speed < spf] = False
    
    whl = np.array(whl[index_tist,:])
    occ = np.zeros((55,55))
    for i in range(55): #could be optimized...
        for j in range(55):
            occ[i,j] = np.sum(K(np.array((2 + i*4, 2 + j*4)),whl,sigma_ker))
    #occ = occ / np.sum(occ) * whl.shape[0] * 0.0256  #try to be coherent with the time - in seconds!
    occ[occ<np.max(occ)/2000]=0 #cut out bins with low coverage, < 0.1% of peak
    return occ

def ratemap_inds(occ, whl, spkl, speed, spf, inds,
                 sigma_ker=4.2, sigma_gauss=1):
    
    rate = np.zeros((55,55))
    index_tist=inds
    #speed filter
    index_tist[speed < spf] = False

    if len(spkl) > 0:
        spkl = np.array(spkl)
        spkl = spkl[index_tist[spkl]]
        spkp = np.array(whl[spkl,:])
        
        for i in range(55): #could be optimized...
            for j in range(55):
                if(occ[i,j] > 0):
                    rate[i,j] = np.sum(K(np.array((2 + i*4, 2 + j*4)),spkp,sigma_ker)) / occ[i,j]

        if sigma_gauss > 0:
            rate = nd.gaussian_filter(rate,sigma=sigma_gauss)
            
    return rate, len(spkl)


from scipy.stats import circmean
def heading(whl):
    temp = np.zeros(whl.shape[0])
    heading = np.zeros(whl.shape[0])
    for i in range(whl.shape[0]-1):
        if whl[i,0] > 0 and whl[i+1,0] > 0 and whl[i,1] > 0 and whl[i+1,1] > 0:
            temp[i] = np.arctan2(whl[i+1,1] - whl[i,1], whl[i+1,0] - whl[i,0]) + np.pi
        else:
            temp[i]=-1000
    
    for i in range(5,whl.shape[0]-5):
        t = temp[i-5:i+5]
        t = t[t>-1000]
        if(len(t)>0):
            heading[i] = circmean(t)
    return heading
def sparsity(rm,occ):
    om = np.array(occ/np.sum(occ))
    if np.sum(rm*rm*om) > 0:
        sp = 1 - np.sum(rm*om)**2 / np.sum(rm*rm*om)
    else:
        sp = 0
    #penalize low firing rate
    if np.max(rm) < 2:
        sp = sp * np.log(0.61 + np.max(rm))
    return sp

def spat_info(rm,occ):
    om = np.array(occ/np.sum(occ))
    rm=np.array(rm)
    r=np.mean(rm[om>0])
    si=-1
    if r>0:
        ind = rm > 0
        si=np.sum( om[ind] * (rm[ind]/r) * np.log2(rm[ind]/r))
    return si