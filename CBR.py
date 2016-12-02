import random
import numpy as np
from collections import defaultdict
from time import time
import pp
def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))
def dsigmoid(z):
    return np.exp(-z)/(1+np.exp(-z))**2
def reverseR(R):
    Rr=defaultdict(dict)
    for u in R:
        for i in R[u]:
            Rr[i][u] = R[u][i]
    return Rr
def normalize(T):
    for u in T:
        All = sum(T[u].values())
        if All>0:
            for v in T[u]:
                T[u][v] /= All
    return T 
def average(Rr,c):
    All = 0.0;
    sumn = 0;
    for i in c:
        All += sum(Rr[i].values())
        sumn += len(Rr[i].values())
    if sumn>0:
        return All/sumn
    else:
        return 0
    
def rmse(U,V,R,v2c):
    error = 0.0
    nums = 0
    for u in R:
        for i in R[u]:
            cid = v2c[i]
            if cid<len(U):
                error += 25*(sigmoid(U[cid][u].dot(V[i]))- R[u][i])**2
                nums += 1
    return np.sqrt(error/nums)
def mae(U,V,R,v2c):
    ap = 0.0
    nums = 0
    for u in R:
        for i in R[u]:
            cid = v2c[i]
            if cid<len(U):
                ap += abs(sigmoid(U[cid][u].dot(V[i]))-R[u][i])/R[u][i]
                nums += 1
    return ap/nums
def CBR(R,T,clists, N,M,K, lambdaU,lambdaV,lambdaT):
    def CircleCon3(T,clists):
        nc = defaultdict(dict)
        res = defaultdict(dict)
        for i in range(len(clists)):
            res[i] = defaultdict(dict)
        for u in T:
            for v in T[u]:
                All = 0
                for ci,c in enumerate(clists):
                    if v in R:
                        nc[ci]=(len(set(R[v].keys())&set(c)))
                        All += nc[ci]
                if All>0:
                    for ci,c in enumerate(clists):
                        if nc[ci]>0:
                            res[ci][u][v] = 1.0 * nc[ci]/All
        return res
    def costL(U,V,*args):
        vid,R,T,Rr=args
        cost=0.0
        for u in R:
            for i in (set(vid)&set(R[u])):
                cost += 0.5 * (R[u][i] - sigmoid(U[u].dot(V[i])))**2
        cost += lambdaU/2 * np.linalg.norm(U)
        cost += lambdaV/2 * np.linalg.norm(V)
        for u in T:
            e = np.copy(U[u]) #1xK
            for v in T[u]:
                e -= T[u][v]*U[v]
            cost += lambdaT/2 * e.dot(e.T)
        return cost

    def gradient(U,V, *args):
        vid,R,T,Rr=args
        dU = np.zeros(U.shape)
        dV = np.zeros(V.shape)
        for u in R:
            for i in set(vid)&set(R[u]):
                tmp = U[u].dot(V[i])
                dU[u] += V[i]*dsigmoid(tmp)*(sigmoid(tmp)-R[u][i]) 
            dU[u] += lambdaU * U[u]
            e = np.copy(U[u]) #1xK
            for v in T[u]:
                e -= T[u][v] * U[v]
            dU[u] += lambdaT * e
            e2 = np.zeros_like(U[u])
            for v in T[u]:
                if v in T and u in T[v]:
                    e = np.copy(U[v])
                    for w in T[v]:
                        e-= T[v][w] * U[w]
                    e2 += T[v][u] * e 
            dU[u] -= lambdaT * e2
        for i in vid:
            for u in Rr[i]:
                tmp = U[u].dot(V[i])
                dV[i] += U[u] * dsigmoid(tmp) * (sigmoid(tmp)-R[u][i])
            dV[i] += lambdaV * V[i]
        return dU,dV

    def f(x, *args):
        x=x.reshape(N+M,K)
        u, v = x[:N,], x[N:,]
        return costL(u, v, *args)

    def gradf(x, *args):
        x=x.reshape(N+M,K)
        u, v = x[:N,], x[N:,]
        gu,gv=gradient(u,v,*args)
        x_ = np.vstack((gu,gv)).ravel()
        return x_

    def optim(x0,vid,cid):
        from scipy import optimize
        x0=np.vstack(x0).ravel()
        Sc = normalize(Se[cid])
        args=vid,R,Sc,Rr
        x = optimize.fmin_cg(f, x0, fprime=gradf, args=args)
        x=x.reshape(N+M,K)
        u, v = x[:N,], x[N:,]
        return  u,v

    def train(U,V,vid,cid):
        Sc = normalize(Se[cid])
        args=vid,R,Sc,Rr
        res=[]
        steps=10**3
        rate = 0.1
        pregradU = 0
        pregradV = 0
        tol=1e-3
        momentum = 0.8
        stage = max(steps/100 , 1)
        for step in xrange(steps):
            dU,dV = gradient(U,V,*args)
            dU = dU + momentum*pregradU
            dV = dV + momentum*pregradV
            pregradU = dU
            pregradV = dV
            if not step%stage and rate>0.001:
                rate = 0.95*rate
            U -= rate * dU
            V -= rate * dV
            e = costL(U,V,*args)
            res.append(e)
            if not step%(stage*5):
                print step,e
            if step>100 and abs(sum(res[-10:])-sum(res[-20:-10]))<tol:
                print "====================" 
                print "stop in %d step"%(step)
                print "error is ",e
                print "====================" 
                break
        return U, V

    Rr = reverseR(R)
    Se = CircleCon3(T,clists)
    Uembd = defaultdict(dict)
    Vembd = np.random.normal(0,0.01,size=(M,K))
    for cid,c in enumerate(clists):
        Uc = np.random.normal(0,0.01,size=(N,K))
        x0=Uc,Vembd
        # Uc,Vembd = optim(x0,c,cid)
        #Uc,Vembd = optim(x0,c,cid)
        Uc,Vembd = train(Uc,Vembd,c,cid)
        Uembd[cid] = Uc
    return Uembd,Vembd

def test(R,T,C,N,M,K,max_r,lambdaU,lambdaV,lambdaT,R_test):
    def lookR_hat():
        R_hat=np.zeros((N,M))
        for u in R:
            for v in R[u]:
                cid = v2c[v]
                if cid<len(Uembd):
                    ub = Uembd[cid][u]
                    vb = Vembd[v]
                    R_hat[u][v] = max_r*max(0,(ub.dot(vb)+bias[cid]))
                else:
                    R_hat[u][v] = 0
        print "R_hat",R_hat 
        return R_hat
    print 'N:%d, M:%d, K:%d,lambdaU:%s, lambdaV:%s,lambdaT:%s' \
            %(N,M,K,lambdaU,lambdaV,lambdaT)
    start = time()
    Uembd,Vembd = CBR(R,T,C,N,M,K,lambdaU,lambdaV,lambdaT)
    # print "u",Uembd
    # print "v",Vembd  
    Rr = reverseR(R)
    v2c = defaultdict(int)
    for cid,c in enumerate(C):
        for v in c:
            v2c[v] = cid

    # print 'R_hat:\n', R_hat
    print "=================RESULT======================="
    print 'K:%d,lambdaU:%s, lambdaV:%s,lambdaT:%s' \
            %(K,lambdaU,lambdaV,lambdaT)
    print "time",time()-start
    print "rmse",rmse(Uembd,Vembd,R_test,v2c)
    print "mae",mae(Uembd,Vembd,R_test,v2c)
    return Uembd,Vembd,rmse(Uembd,Vembd,R_test,v2c),mae(Uembd,Vembd,R_test,v2c)

def t_yelp(limitu,limiti):
    #data from: http://www.trustlet.org/wiki/Epinions_datasets
    N,M = limitu,limiti
    max_r = 5.0
    cNum = 8
    R=defaultdict(dict)
    T=defaultdict(dict)
    R_test=defaultdict(dict)
    print 'get T'
    for line in open('./yelp_data/users.txt','r'):
        u = int(line.split(':')[0])
        uf = line.split(':')[1][1:-1].split(',')
        if len(uf)>1:
            for x in line.split(':')[1][1:-1].split(',')[:-1]:
                v = int(x)
                if u < limitu and v<limitu:
                    T[u][v] = 1.0
    print 'get R'
    k = 0
    for line in open('./yelp_data/ratings-train.txt','r'):
        u,i,r = [int(x) for x in line.split('::')[:3]]
        if u<limitu and i<limiti:
            R[u][i] = r/max_r

    print 'get R_test'
    for line in open('./yelp_data/ratings-test.txt','r'):
        u,i,r = [int(x) for x in line.split('::')[:3]]
        if u<limitu and i<limiti:
            R_test[u][i] = r/max_r
    print "get Circle"
    C = [[] for i in range(cNum)]
    for line in open('./yelp_data/items-class.txt','r'):
        i,ci = [int(x) for x in line.split(' ')]
        if i<limiti:
            C[ci].append(i)
    lambdaU_,lambdaV_,lambdaT_,K_=0.2, 0.2, 0.1, 4
    U_,V_,rmse_,mae_ = test(R,T,C,N,M,K_,max_r,lambdaU_,lambdaV_,lambdaT_,R_test)
    job_server = pp.Server(5)
    jobs = []
    for lambdaU in [0.2,0.5,1]:
        for lambdaV in [0.2,0.5,1]:
            for lambdaT in [0.1,0.5,1]:
                # test(R,T,N,M,K,max_r,lambdaU,lambdaV,lambdaT,R_test)
                jobs.append(job_server.submit(test,(R,T,C,N,M,K_,max_r,lambdaU,lambdaV,lambdaT,R),(mae,rmse,average,circleRec,reverseR,normalize,sigmoid,dsigmoid),("numpy as np","from collections import defaultdict","random","from time import time")))
    job_server.wait()
    for job in jobs:
            U,V,rmse1,mae1 = job()
            if mae1+rmse1<mae_+rmse_:
                U_ = U
                V_ = V
                lambdaU_ = lambdaU
                lambdaV_ = lambdaV
                lambdaT_ = lambdaT
                mae_,rmse_ = mae1,rmse1
    print "jobs finish"
    jobs = []
    for K in [1,2,3,4,5]:
        jobs.append(job_server.submit(test,(R,T,C,N,M,K,max_r,lambdaU_,lambdaV_,lambdaT_,R_test),(mae,rmse,average,circleRec,reverseR,normalize,sigmoid,dsigmoid),("numpy as np","from collections import defaultdict","random","from time import time")))
    job_server.wait()
    for job in jobs:
            U,V,rmse1,mae1 = job()
            if mae1+rmse1<mae_+rmse_:
                K_ = K
                U_ = U
                V_ = V
                lambdaU_ = lambdaU
                lambdaV_ = lambdaV
                mae_,rmse_ = mae1,rmse1
    print "=========all finish=============="
    print "rmse-test",rmse_
    print "map-test",mae_

if __name__ == "__main__":
   t_yelp(1000,20000)
   # t_toy()