import numpy as np
import scipy.special
from scipy.spatial.distance import pdist as pairwise_dist, squareform

import pennylane as qml

from scipy.spatial.distance import hamming as dH
from scipy.linalg import logm, sqrtm

# Importing standard Qiskit libraries
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, transpile, Aer, IBMQ, execute
from qiskit.circuit import Parameter, ParameterVector
from qiskit.providers.aer import QasmSimulator
from qiskit.circuit.library.standard_gates import RYGate, XXPlusYYGate

inf = 9999999

# ---------------------- Quantum Circuit section ------------------------

def Hadamard_barrier(n):
    # create Hadamard barrier on n qubits
    qc = QuantumCircuit(n)
    for i in range(n):
        qc.h(i)
    return(qc)


def Uz(alpha,n):
    # Create the Ising Hamiltonian circuit in eq. (4)
    # inputs: alpha is a lower triangular numpy matrix, with diagonal elements b_k and lower elements J_ij
    qc = QuantumCircuit(n)
    for i in range(n):
        qc.rz(alpha[i,i],i)

    for i in range(n):
        j=0
        while j < i:
            qc.rzz( alpha[i,j], i, j)
            j+=1
    return(qc)


def Uf(Gamma, Delta, Sigma, n):
    # Create Uf circuit as in eq. (3)
    # inputs are n-dim numpy vectors
    qc = QuantumCircuit(n)
    for k in range(n):
        qc.rx(Gamma[k],k)
        qc.ry(Delta[k],k)
        qc.rz(Sigma[k],k)
    return(qc)


def QCIBM(alpha, Gamma, Delta, Sigma, n):
    # create QCIBM circuit according to parameters
    qc = QuantumCircuit(n)
    qc.compose(Hadamard_barrier(n), qubits = range(n), inplace = True)
    qc.compose(Uz(alpha, n), qubits = range(n), inplace = True)
    qc.compose(Uf(Gamma, Delta, Sigma, n), qubits = range(n), inplace = True)
    qc.measure_all()

    return(qc)


def sample_circuit(backend, nsamples = 1024, qc = QuantumCircuit(0) ):
    # sample from (parametrized) quantum circuit
    nq = qc.num_qubits
    if nq == 0: # create circuit from scratch
        print('Please, provide a quantum circuit')

    res = backend.run(qc, seed_simulator=2703, shots=nsamples, memory = True).result()
    rawdata = res.get_memory()
    xsamples = np.zeros((len(rawdata),nq))
    for i in range(len(rawdata)):
        xsamples[i] = str_to_array(rawdata[i])

    counts = res.get_counts()
    phat = np.zeros(2**nq)
    for j in range(2**nq):
        binstr = binstr_from_int(j,nq)
        if(binstr in counts.keys()):
            phat[j] = counts[binstr]/nsamples
        else:
            phat[j] = 0

    return(xsamples,phat)


# ----------------- Loss Section ----------------------------------

def TV_dist(p,q):
    # compute Total Variation distance between discrete distributions p, q
    TV = 0
    for x in range(len(p)):
        TV += np.abs(p[x] - q[x])
    return(TV/2)

def KL_Hypercube_Loss(psamples, qsamples, kernel_params):
    # compute kernel KL Loss with linear parametrized kernel
    Sigmap = estimate_covariance_embedding(psamples, kernel_params) # alternatively one can use the Graam matrix for computing D_KL
    Sigmaq = estimate_covariance_embedding(qsamples, kernel_params)

    #return(entropy_relative(Sigmap,Sigmaq))
    return(entropy_relative(Sigmap,Sigmaq))

def KL_Hypercube_grad(qc, theta, theta_t, psamples, qsamples, backend, kernel_params):
    # compute gradient of the MMD Loss wrt circuit parameters
    nq = qc.num_qubits
    nabla = np.zeros(len(theta_t))
    N = len(psamples)
    M = len(qsamples)
    P = 1024
    Q = 1024

    sigmap = estimate_covariance_embedding(psamples, kernel_params )
    sigmaq = estimate_covariance_embedding(qsamples, kernel_params)

    for k in range(len(theta_t)):

        theta_plus = np.copy(theta_t)
        theta_plus[k] += np.pi/2
        theta_minus = np.copy(theta_t)
        theta_minus[k] -= np.pi/2

        asamples, phat = sample_circuit(backend, nsamples = P, qc = qc.bind_parameters({theta: theta_minus}) )
        bsamples, phat = sample_circuit(backend, nsamples = Q, qc = qc.bind_parameters({theta: theta_plus}) )

        sigmaminus = estimate_covariance_embedding(asamples, kernel_params)
        sigmaplus = estimate_covariance_embedding(bsamples,  kernel_params)

        Delta = sigmaplus - sigmaminus

        #nabla[k] = 0.5*(entropy_cross(Delta,sigmaq) - entropy_cross(Delta,sigmap))
        nabla[k] = 0.5*np.trace( (nsym(sigmap) - nsym(sigmaq))@Delta )

    return(nabla)

def R2_Hypercube_Loss(psamples, qsamples, kernel_params):
    # compute kernel KL Loss with linear parametrized kernel
    Sigmap = estimate_covariance_embedding(psamples, kernel_params) # alternatively one can use the Graam matrix for computing D_KL
    Sigmaq = estimate_covariance_embedding(qsamples, kernel_params)

    #return(entropy_relative(Sigmap,Sigmaq))
    return(renyi2_relative(Sigmap,Sigmaq))

def R2_Hypercube_grad(qc, theta, theta_t, psamples, qsamples, backend, kernel_params):
    # compute gradient of the MMD Loss wrt circuit parameters
    nq = qc.num_qubits
    nabla = np.zeros(len(theta_t))
    N = len(psamples)
    M = len(qsamples)
    P = 1024
    Q = 1024

    sigmap = estimate_covariance_embedding(psamples, kernel_params)
    sigmaq = estimate_covariance_embedding(qsamples, kernel_params)
    if( len(sigmaq) > np.linalg.matrix_rank(sigmaq, tol=10**(-11)) ):
        print('Warning: sigmaq is not full rank in R2 grad')
        sigmaq = sigmaq + 10**(-10)*np.eye(len(sigmaq)) # stabilize
    sigmaqinv = np.linalg.inv(sigmaq)
    denom = 2* np.trace( sigmap@sigmap@sigmaqinv)

    for k in range(len(theta_t)):

        theta_plus = np.copy(theta_t)
        theta_plus[k] += np.pi/2
        theta_minus = np.copy(theta_t)
        theta_minus[k] -= np.pi/2

        asamples, phat = sample_circuit(backend, nsamples = P, qc = qc.bind_parameters({theta: theta_minus}) )
        bsamples, phat = sample_circuit(backend, nsamples = Q, qc = qc.bind_parameters({theta: theta_plus}) )

        sigmaminus = estimate_covariance_embedding(asamples, kernel_params)
        sigmaplus = estimate_covariance_embedding(bsamples,  kernel_params)

        Delta = sigmaplus - sigmaminus

        nabla[k] = np.trace( anticomm(Delta,sigmap)@sigmaqinv )

    return(nabla)


def KL_Gauss_Loss(psamples, qsamples, kernel_params, Ksqrt=[], proj_samples=[]):
    # compute kernel KL Loss with linear parametrized kernel

    if(len(Ksqrt) == 0 and len(proj_samples)==0):
        rng = np.random.default_rng(27031995)
        proj_nsamples = 256
        nq = psamples.shape[1]
        proj_samples = np.zeros((proj_nsamples,nq))
        for i in range(nq):
            proj_samples[:,i] =  rng.integers(0, 2, proj_nsamples)
        K = compute_graham_gauss(proj_samples, kernel_params)
        delta = 10**(-10)
        #Ksqrt = np.real(sqrtm(np.linalg.inv(0.5*(K+K.T) + delta * np.eye(K.shape[0]))))
        Ksqrt = negsqrtm(K+ delta * np.eye(K.shape[0]))
    elif(len(proj_samples) == 0 or len(Ksqrt) == 0):
        print('Warining: only one of the following has been provided while both are needed: Ksqrt, Ksamples')

    #print('K = ', K)
    #print(np.linalg.eig(0.5*(K+K.T) + delta * np.eye(K.shape[0])))
    #print('sqrt(K)^-1' , Ksqrt)

    Sigmap = estimate_covariance_embedding_projection(psamples,proj_samples, Ksqrt, kernel_params)
    Sigmaq = estimate_covariance_embedding_projection(qsamples,proj_samples, Ksqrt, kernel_params)

    return(entropy_relative(Sigmap,Sigmaq))



def KL_Gauss_grad(qc, theta, theta_t, psamples, qsamples, backend, kernel_params, Ksqrt=[], proj_samples=[]):
    # compute gradient of the MMD Loss wrt circuit parameters
    nq = qc.num_qubits
    nabla = np.zeros(len(theta_t))
    N = len(psamples)
    M = len(qsamples)
    P = 1024
    Q = 1024

    if(len(Ksqrt) == 0 and len(proj_samples)==0):
        rng = np.random.default_rng(27031995)
        proj_nsamples = 256
        nq = psamples.shape[1]
        proj_samples = np.zeros((proj_nsamples,nq))
        for i in range(nq):
            proj_samples[:,i] =  rng.integers(0, 2, proj_nsamples)
        K = compute_graham_gauss(proj_samples, kernel_params)
        delta = 10**(-10)
        #Ksqrt = np.real(sqrtm(np.linalg.inv(0.5*(K+K.T) + delta * np.eye(K.shape[0]))))
        Ksqrt = negsqrtm(K+ delta * np.eye(K.shape[0]))
    elif(len(proj_samples) == 0 or len(Ksqrt) == 0):
        print('Warining: only one of the following has been provided while both are needed: Ksqrt, Ksamples')

    sigmap = estimate_covariance_embedding_projection(psamples,proj_samples, Ksqrt, kernel_params)
    sigmaq = estimate_covariance_embedding_projection(qsamples,proj_samples, Ksqrt, kernel_params)

    for k in range(len(theta_t)):

        theta_plus = np.copy(theta_t)
        theta_plus[k] += np.pi/2
        theta_minus = np.copy(theta_t)
        theta_minus[k] -= np.pi/2

        asamples, phat = sample_circuit(backend, nsamples = P, qc = qc.bind_parameters({theta: theta_minus}) )
        bsamples, phat = sample_circuit(backend, nsamples = Q, qc = qc.bind_parameters({theta: theta_plus}) )

        sigmaminus = estimate_covariance_embedding_projection(asamples,proj_samples, Ksqrt, kernel_params)
        sigmaplus =estimate_covariance_embedding_projection(bsamples,proj_samples, Ksqrt, kernel_params)

        Delta = sigmaplus - sigmaminus


        #nabla[k] = 0.5*(entropy_cross(Delta,sigmaq) - entropy_cross(Delta,sigmap))
        nabla[k] = np.real( 0.5*np.trace( (nsym(sigmap) - nsym(sigmaq))@Delta ) )

    return(nabla)


def MMD_Gauss_Loss(psamples, qsamples, kernel_bandwidth):
    # compute MMD Loss with Gaussian kernel
    Lp, Lq, Lpq = 0, 0, 0
    N = len(psamples)
    M = len(qsamples)
    for i in range(N):
        for j in range(N):
            Lp += k_Gauss(psamples[i], psamples[j], kernel_bandwidth)

    for i in range(M):
        for j in range(M):
            Lq += k_Gauss(qsamples[i], qsamples[j], kernel_bandwidth)

    for i in range(N):
        for j in range(M):
            Lpq += k_Gauss(psamples[i], qsamples[j], kernel_bandwidth)

    return( Lp/(N*(N-1)) + Lq/(M*(M-1)) - 2*Lpq/(N*M) )

def MMD_Gauss_grad(qc, theta, theta_t, psamples, qsamples, backend, kernel_bandwidth):
    # compute gradient of the MMD Loss wrt circuit parameters
    nq = qc.num_qubits
    nabla = np.zeros(len(theta_t))
    N = len(psamples)
    M = len(qsamples)
    P = 1024
    Q = 1024
    for k in range(len(theta_t)):

        theta_plus = np.copy(theta_t)
        theta_plus[k] += np.pi/2
        theta_minus = np.copy(theta_t)
        theta_minus[k] -= np.pi/2


        asamples, phat = sample_circuit(backend, nsamples = P, qc = qc.bind_parameters({theta: theta_minus}) )
        bsamples, phat = sample_circuit(backend, nsamples = Q, qc = qc.bind_parameters({theta: theta_plus}) )

        dLk = 0
        dLk -= 1/(P*N) * k_Gauss(asamples, psamples, kernel_bandwidth)
        dLk += 1/(Q*N) * k_Gauss(bsamples, psamples, kernel_bandwidth)
        dLk += 1/(P*M) * k_Gauss(asamples, qsamples, kernel_bandwidth)
        dLk -= 1/(Q*M) * k_Gauss(bsamples, qsamples, kernel_bandwidth)

        nabla[k] = dLk

    return(nabla)

# ------------------- Gradient Descent utilities ------------------------------

def adam(f, g, thetapar, theta0, backend, qc, sampleshots, ysamples, batch_size=1 ,p_target = [] ,tol = 10**(-8), maxiter= 200, alpha=0.001, beta1=0.9, beta2=0.999):
    TV_history = []
    Loss_history = []
    epsilon = 10**(-8)
    m = 0
    v = 0
    t = 0
    theta = theta0
    xsamples, phat = sample_circuit( backend, sampleshots, qc.bind_parameters({thetapar: theta}))
    ft = f(theta, xsamples)
    if len(p_target) > 0: # if we only have samples (e.g. real data, we may not have p_target)
        TV_history.append(TV_dist(phat,p_target))
    Loss_history.append(ft)
    ft_ = ft -2*tol
    num_batches = int(np.ceil(np.divide(ysamples.shape[0], batch_size)))
    #while abs(ft - ft_) > tol and t < maxiter:
    while abs(ft)>tol and t < maxiter:
        t = t+1
        print(' -- Adam Optimization -- Step t = ',t)
        print(' - Current Loss = ',ft)

        for b in range(num_batches):
            batch_inds = np.arange(b*batch_size, min((b+1)*batch_size, ysamples.shape[0]))
            gt = g(theta,xsamples[batch_inds], ysamples[batch_inds])
            m = beta1*m + (1-beta1)*gt
            v = beta2*v + (1 - beta2)*(gt**2)
            mhat = m/(1-beta1**t)
            vhat = v/(1-beta2**t)
            theta = theta - alpha*np.divide(mhat,(np.sqrt(v) + epsilon ))

        xsamples, phat = sample_circuit( backend, sampleshots, qc.bind_parameters({thetapar: theta}))
        ft_ = ft
        ft = f(theta, xsamples)
        if len(p_target) > 0: # if we only have samples (e.g. real data, we may not have p_target)
            TV_history.append(TV_dist(phat,p_target))
        Loss_history.append(ft)
    return(theta, np.array(Loss_history), np.array(TV_history))

# ------------------------- Kernel Utilities ----------------------------------

def estimate_covariance_embedding(samples, phi_params):
    d = len(phi_params)
    if(d-1 != samples.shape[1]):
        print('Estimate covariance embedding ERROR: Wrong Dimension!')
        return 0
    eta = np.diag(np.sqrt(phi_params))
    sigma = np.zeros((d,d))
    for i in range(len(samples)):
        cvec = np.append( 2*samples[i]-1, 1 )
        sigma += eta @ np.outer(cvec,cvec) @ eta

    return sigma/len(samples)

def estimate_covariance_embedding_projection(xsamples,samples, Ksqrt, ker_params):

    Sigmap = np.zeros((samples.shape[0],samples.shape[0]))

    for i in range(xsamples.shape[0]):
        ki = compute_kernel_vector(samples,xsamples[i], ker_params)
        Sigmap += np.outer(ki,ki)

    delta = 10**(-12)
    Sigmap = 1/(2*xsamples.shape[0]) * Ksqrt @ (Sigmap + Sigmap.conj().T + delta*np.eye(Sigmap.shape[0])) @ Ksqrt


    return(Sigmap)


def compute_graham_gauss(samples, ker_params):
    c = len(ker_params)
    S = squareform(pairwise_dist(samples))
    K = np.zeros(S.shape)
    for i in range(c):
        K += np.exp(-S**2/(2*ker_params[i]))


    return(K/c)

def compute_kernel_vector(X, x, kernel_params):
    Y = (x.reshape((X.shape[1],1)) @ np.ones((1,X.shape[0]))).T
    e = np.diag((X-Y)@(X-Y).T)
    k = np.zeros(e.shape[0])
    for c in range(len(kernel_params)):
        k += np.exp(-e/(2*kernel_params[c]))

    return(k/c)


def k_Gauss(x,y,bandwidth):
    # compute Gaussian kernel with given bandwidth
    c = len(bandwidth)

    if len(x.shape) != len(y.shape):
        print('Error in kernel input')
        k = -1

    if len(x.shape) == 1:
        k = 0
        d = (x-y).T @ (x-y)
        for i in range(c):
            k += np.exp(-d/(2*bandwidth[i]))
    else:
        k = 0
        mul = np.ones(x.shape[0]).reshape((x.shape[0],1))
        for p in range(y.shape[0]):
            kp = np.zeros(x.shape[0])
            d = np.diag((x-mul@y[p].reshape((1,len(y[p])))) @ (x-mul@y[p].reshape((1,len(y[p])))).T)
            for i in range(c):
                kp += np.exp(-d/(2*bandwidth[i]))
            k += np.sum(kp)
    return(k)

# -------------------------- Entropies ---------------------------------------
'''
def entropy_relative(rho, sigma, tol=1e-12):

    rvals, rvecs = np.linalg.eig(rho)
    if any(abs(np.imag(rvals)) >= tol):
        raise ValueError("Input rho has non-real eigenvalues.")
    rvals = np.real(rvals)
    svals, svecs = np.linalg.eig(sigma)
    if any(abs(np.imag(svals)) >= tol):
        raise ValueError("Input sigma has non-real eigenvalues.")
    svals = np.real(svals)
    # Calculate inner products of eigenvectors and return +inf if kernel
    # of sigma overlaps with support of rho.
    P = abs(rvecs @ np.conj(svecs)) ** 2
    print(P)
    if (rvals >= tol) @ (P >= tol) @ (svals < tol):
        return inf
    # Avoid -inf from log(0) -- these terms will be multiplied by zero later
    # anyway
    svals[abs(svals) < tol] = 1
    nzrvals = rvals[abs(rvals) >= tol]
    # Calculate S
    S = nzrvals @ np.log(nzrvals) - rvals @ P @ np.log(svals)
    # the relative entropy is guaranteed to be >= 0, so we clamp the
    # calculated value to 0 to avoid small violations of the lower bound.
    if( S < 0):
        print('Negative divergence: S = ', S)
    return max(0, S)
    '''

def renyi2_relative(rho, sigma, sigmainv = np.array([]) ,tol=1e-12):
    rvals, rvecs = np.linalg.eigh(rho)
    rvals = np.real(rvals)
    svals, svecs = np.linalg.eigh(sigma)
    svals = np.real(svals)

    if any(svals < tol):
        print('Renyi2 entropy: sigma is not full rank')
        return(inf)

    if any(rvals<tol):
        T = sigma@rvecs[:,rvals>tol]
        flags = []
        [flags.append(np.linalg.norm(T[:,i]) < tol) for i in range(np.sum(rvals>tol))]
        if any(flags):
            return(inf)

    if(len(sigmainv)!=len(sigma)): # if inverse is not provided
        sigmainv = np.linalg.inv(sigma)

    return(np.real(np.log(np.trace(rho@rho@sigmainv))))

def entropy_relative(rho, sigma, tol=1e-14):
    rvals, rvecs = np.linalg.eigh(rho)
    rvals = np.real(rvals)
    svals, svecs = np.linalg.eigh(sigma)
    svals = np.real(svals)

    if any(rvals<tol):
        T = sigma@rvecs[:,rvals>tol]
        flags = []
        [flags.append(np.linalg.norm(T[:,i]) < tol) for i in range(np.sum(rvals>tol))]
        if any(flags):
            print(inf)

    H = -rvals[rvals > tol]@np.log(rvals[rvals > tol])
    rvals[rvals < tol] = 0
    svals[svals < tol] = 1

    IP = abs(np.conj(svecs).T@rvecs)**2
    Hcross = -rvals@IP@np.log(svals)
    return(np.real(Hcross - H -np.trace(rho) + np.trace(sigma)))

'''
def entropy_cross(rho, sigma, tol=1e-12):

    rvals, rvecs = np.linalg.eig(rho)
    if any(abs(np.imag(rvals)) >= tol):
        raise ValueError("Input rho has non-real eigenvalues.")
    rvals = np.real(rvals)
    svals, svecs = np.linalg.eig(sigma)
    if any(abs(np.imag(svals)) >= tol):
        raise ValueError("Input sigma has non-real eigenvalues.")
    svals = np.real(svals)
    # Calculate inner products of eigenvectors and return +inf if kernel
    # of sigma overlaps with support of rho.
    P = abs(rvecs @ np.conj(svecs)) ** 2
    if (rvals >= tol) @ (P >= tol) @ (svals < tol):
        return inf
    # Avoid -inf from log(0) -- these terms will be multiplied by zero later
    # anyway
    svals[abs(svals) < tol] = 1
    nzrvals = rvals[abs(rvals) >= tol]
    # Calculate S
    H = - rvals @ P @ np.log(svals)
    # the relative entropy is guaranteed to be >= 0, so we clamp the
    # calculated value to 0 to avoid small violations of the lower bound.
    return H
'''

'''
def entropy_cross(rho, sigma, tol=1e-12):

    rvals, rvecs = np.linalg.eig(rho)
    if any(abs(np.imag(rvals)) >= tol):
        raise ValueError("Input rho has non-real eigenvalues.")
    rvals = np.real(rvals)
    svals, svecs = np.linalg.eig(sigma)
    if any(abs(np.imag(svals)) >= tol):
        raise ValueError("Input sigma has non-real eigenvalues.")
    svals = np.real(svals)
    rvals = rvals[rvals > tol]
    svals = svals[svals > tol]
    H = 0
    for j in range(len(rvals)):
        for k in range(len(svals)):
            if(rvals[j] > tol and svals[k] > tol):
                H += abs(rvecs[j].dot(np.conj(svecs[k])))**2 * rvals[j]*np.log(svals[k])

    return(-H)
'''
def entropy_cross(rho,sigma, tol = 10**(-12)):
    rvals, rvecs = np.linalg.eig(rho)
    rvals = np.real(rvals)
    svals, svecs = np.linalg.eig(sigma)
    svals = np.real(svals)

    rvals[rvals < tol] = 0
    svals[svals < tol] = 1

    IP = abs(rvecs@np.conj(svecs).T)**2
    Hcross = -rvals@IP@np.log(svals)
    return(Hcross)


# ------------------- General Purpose Utilities -------------------------------

def int_to_binarray(num, n):
    br = np.binary_repr(num, width=n)
    br_array = np.zeros(n)
    for i in range(n):
        br_array[i] = br[i]
    return(br_array)


def target_pdf(x, s, p):
    # compute target pdf at point x
    T = len(s)
    nq = len(x)
    pi_x = 0
    for k in range(T):
        d = nq*dH(x,s[k])
        pi_x += p**(nq - d) * (1-p)**(d)

    return(pi_x/T)

def compute_target_table(s,p):
    nq = len(s[0])
    spdim = 2**nq
    pvec = np.zeros(spdim)
    x = np.zeros((spdim,nq))
    for j in range(spdim):
        x[j] = int_to_binarray(j,nq)
        pvec[j] = target_pdf(x[j],s,p)
    return([x,pvec])

def sample_target_pdf(n, s, p):
    # sample n vectors from target pdf
    nq = len(s[0])
    spdim = 2**nq
    [x,pvec] = compute_target_table(s,p)
    #print('\pi(x) = ', pvec)
    x_int = np.random.choice(len(pvec), n, p=pvec)
    samples = np.zeros((n,nq))
    for i in range(n):
        samples[i] = x[x_int[i]]
    return(samples, pvec)



def decompose_params(theta, n):
    # decompose theta to get all parameters
    nj = int(n*(n+1)/2)
    if(len(theta) == n*(n+7)/2):
        b = theta[:n]
        J = theta[n:nj]
        Gamma = theta[nj:nj+n]
        Delta = theta[nj+n:nj+2*n]
        Sigma = theta[nj+2*n:nj+3*n]
        alpha = np.diag(b)
        alpha[np.tril_indices(n,-1)] = J

    elif(len(theta) == n*(n+3)/2):
        b = theta[:n]
        J = theta[n:nj]
        Gamma = theta[nj:nj+n]
        Delta = np.zeros(n)
        Sigma = np.zeros(n)
        alpha = np.diag(b)
        alpha[np.tril_indices(n,-1)] = J

    else:
        print('WARNING! Theta length is not correct!')
        return(0)

    return(alpha, Gamma, Delta, Sigma)



def decompose_params_QAOA(theta, n):
    # decompose theta to get all parameters
    if(len(theta) != n*(n+3)/2):
        print('WARNING! Theta length is not correct!')
        return(0)

    nj = int(n*(n+1)/2)
    b = theta[:n]
    J = theta[n:nj]
    Gamma = -theta[nj:nj+n]
    Delta = np.zeros(n)
    Sigma = np.zeros(n)

    alpha = np.diag(b)
    alpha[np.tril_indices(n,-1)] = J

    return(alpha, Gamma, Delta, Sigma)

def compose_params(alpha, Gamma, Delta, Sigma):
    # compose parameters to get thteta
    n = len(Gamma)
    b = np.diag(alpha)
    J = alpha[np.tril_indices(n,-1)]
    theta = np.concatenate((b,J, Gamma, Delta, Sigma))

    return(theta)

def compose_params_QAOA(alpha, Gamma):
    # compose parameters to get thteta
    n = len(Gamma)
    b = np.diag(alpha)
    J = alpha[np.tril_indices(n,-1)]
    theta = np.concatenate((b,J, -Gamma))

    return(theta)

def str_to_array(s):
    # convert binary string into array form
    ba = np.zeros(len(s))
    for i in range(len(s)):
        if s[i] == '1':
            ba[i] = 1

    return(ba)

def nsym(A):

    return( A + A.T - np.multiply(np.eye(A.shape[0]),A))


def negsqrtm(X):
    # returns the negative square root of a positive (symmetric) matrix
    tol = 10**(-10)
    eigvals, eigvecs = np.linalg.eig(X)
    eigvals = np.real(eigvals)
    eigvals[eigvals > tol] = 1/np.sqrt(eigvals[eigvals > tol])
    eigvals[eigvals < tol] = 0
    X1 = eigvecs@np.diag(eigvals)@np.linalg.inv(eigvecs)
    return(X1)

def anticomm(A,B):
    return(A@B + B@A)

binstr_from_int = lambda x, n: format(x, 'b').zfill(n)
