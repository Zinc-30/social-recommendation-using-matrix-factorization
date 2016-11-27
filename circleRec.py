import numpy as np
import scipy.sparse as sp

def socialMF(R,S,C,N,M,K,lambdaU,lambdaV,lambdaT,R_test,ul,il):
    def sigmoid(z):
        return 1.0 / (1+np.exp(-z))
    def dsigmoid(z):
        return np.exp(-z)/np.power((1+np.exp(-z)),2)
    def rmse(U,V,R):
        keylist = np.array(R.todok().keys()).T
        utl = keylist[0]
        itl = keylist[1]
        error = (get_csrmat(sigmoid(U.dot(V.T)),utl,itl)-R).power(2).sum()/R.nnz
        return 5*np.sqrt(error)
    def mae(U,V,R):
        keylist = np.array(R.todok().keys()).T
        utl = keylist[0]
        itl = keylist[1]
        error = abs(get_csrmat(sigmoid(U.dot(V.T)),utl,itl)-R).sum()/R.nnz
        return error
    def get_csrmat(mat,ul,il):
        indx = ul*mat.shape[1]+il
        return sp.csr_matrix((np.take(np.array(mat),indx),(ul,il)),shape=(N,M))
    def costL(U,V,Sc):
        tmp = U.dot(V.T)
        Rx = get_csrmat(sigmoid(tmp),ul,il)
        cost = 0.5*((R - Rx).power(2)).sum()+0.5*lambdaU*np.linalg.norm(U)**2+0.5*lambdaV*np.linalg.norm(V)**2
        cost += 0.5*lambdaT*np.power(U-S.dot(U),2).sum()
        return cost
    def gradient(U,V):
        dU = lambdaU*U
        tmp = U.dot(V.T)
        Rv = get_csrmat(dsigmoid(tmp),ul,il)
        Rx = get_csrmat(sigmoid(tmp),ul,il)
        dU += Rv.multiply((Rx-R)).dot(V)
        dU += lambdaT*(U-S.dot(U))-lambdaT*S.T.dot((U-S.dot(U)))
        dV = lambdaV*V
        dV += (Rv.multiply((Rx-R))).T.dot(U)
        # print dU,dV
        if np.max(dU)>1:
            dU = dU/np.max(dU)
        if np.max(dV)>1:
            dV = dV/np.max(dV)
        return dU,dV

    def train(U,V):
        res=[]
        steps=10**3
        rate = 0.1
        pregradU = 0
        pregradV = 0
        tol=1e-3
        momentum = 0.9
        stage = max(steps/100 , 1)
        for step in xrange(steps):
            dU,dV = gradient(U,V)
            dU = dU + momentum*pregradU
            dV = dV + momentum*pregradV
            pregradU = dU
            pregradV = dV
            if not step%stage and rate>0.0001:
                rate = 0.95*rate
            U -= rate * dU
            V -= rate * dV
            e = costL(U,V)
            res.append(e)
            if not step%stage:
                print step,e
            if step>100 and abs(sum(res[-3:])-sum(res[-13:-10]))<tol:
                print "====================" 
                print "stop in %d step"%(step)
                print "error is ",e
                print "====================" 
                break
        return U, V
    U = np.random.normal(0,0.1,size=(N,K))
    V = np.random.normal(0,0.1,size=(M,K))
    for i in range(8):
        print C[i]
    # U,V = train(U,V)
    print "=================RESULT======================="
    print 'K:%d,lambdaU:%s, lambdaV:%s,lambdaT:%s' \
            %(K,lambdaU,lambdaV,lambdaT)
    print "rmse",rmse(U,V,R_test)
    print "mae",mae(U,V,R_test)
    return 0
def t_yelp(limitu,limiti):
    #data from: http://www.trustlet.org/wiki/Epinions_datasets
    def getdata():
        N,M = limitu,limiti
        max_r = 5.0
        cNum = 8
        T=sp.dok_matrix((N,N))
        print 'get T'
        for line in open('./yelp_data/users.txt','r'):
            u = int(line.split(':')[0])
            uf = line.split(':')[1][1:-1].split(',')
            if len(uf)>1:
                for x in line.split(':')[1][1:-1].split(',')[:-1]:
                    v = int(x)
                    if u<limitu and v<limitu:
                        T[u,v] = 1.0
        T = T.tocsr()
        print 'get R_test'
        utl,itl,rtl = [],[],[]
        for line in open('./yelp_data/ratings-test.txt','r'):
            u,i,r = [int(x) for x in line.split('::')[:3]]
            if u<limitu and i<limiti:
                utl.append(u)
                itl.append(i)
                rtl.append(r/5.0)
        utl,itl = np.array(utl),np.array(itl)
        R_test = sp.csr_matrix((rtl,(utl,itl)),shape=(N,M))
        print 'get R'
        ul,il,rl = [],[],[]
        for line in open('./yelp_data/ratings-train.txt','r'):
            u,i,r = [int(x) for x in line.split('::')[:3]]
            if u<limitu and i<limiti:
                ul.append(u)
                il.append(i)
                rl.append(r/5.0)
        ul,il = np.array(ul),np.array(il)
        R = sp.csr_matrix((rl,(ul,il)),shape=(N,M))
        print "get Circle"
        C = [[] for i in range(cNum)]
        for line in open('./yelp_data/items-class.txt','r'):
            i,ci = [int(x) for x in line.split(' ')]
            if i<limiti:
                C[ci].append(i)
        C = np.array(C)
        return R,T,C,N,M,R_test,ul,il
    R,T,C,N,M,R_test,ul,il = getdata()
    lambdaU,lambdaV,lambdaT,K = 1, 1, 0.4, 2
    socialMF(R,T,C,N,M,K,lambdaU,lambdaV,lambdaT,R_test,ul,il)

if __name__ == "__main__":
#   t_epinion()
   t_yelp(100,2000)