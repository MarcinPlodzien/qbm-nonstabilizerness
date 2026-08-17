"""Independent O(8^N) reference for M_n. Slow on purpose: no FWHT, no tricks."""
import itertools, numpy as np

I2=np.eye(2,dtype=complex); X=np.array([[0,1],[1,0]],dtype=complex)
Y=np.array([[0,-1j],[1j,0]]); Z=np.diag([1,-1]).astype(complex)
P1=[I2,X,Y,Z]

def sre_bruteforce(psi, N, n=2):
    d=2**N; tot=0.0
    for combo in itertools.product(range(4), repeat=N):
        P=np.array([[1.0+0j]])
        for c in combo: P=np.kron(P,P1[c])
        ev = np.vdot(psi, P@psi).real          # <P> is real for hermitian P
        tot += abs(ev)**(2*n)
    return np.log2(tot/d)/(1-n)
