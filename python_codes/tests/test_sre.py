import sys, os; sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np, jax, jax.numpy as jnp
from utils.sre import stabilizer_renyi_entropy, half_chain_entropy, page_entropy
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _bruteforce_reference import sre_bruteforce
from utils.cliffords import load_clifford_group

rng=np.random.default_rng(7)
print("=== A. FWHT vs BRUTE FORCE O(8^N), random complex states ===")
print(f"{'N':>2} {'M2 brute':>12} {'M2 fwht':>12} {'|diff|':>10}")
worst=0
for N in (2,3,4,5):
    for trial in range(3):
        psi=rng.normal(size=2**N)+1j*rng.normal(size=2**N); psi/=np.linalg.norm(psi)
        b=sre_bruteforce(psi,N); f=float(stabilizer_renyi_entropy(jnp.asarray(psi),N))
        worst=max(worst,abs(b-f))
        if trial==0: print(f"{N:>2} {b:>12.8f} {f:>12.8f} {abs(b-f):>10.2e}")
print(f"  worst |diff| over 12 random states = {worst:.2e}   {'PASS' if worst<1e-9 else '*** FAIL ***'}")

print("\n=== B. n = 3 as well (not just M2) ===")
for N in (3,4):
    psi=rng.normal(size=2**N)+1j*rng.normal(size=2**N); psi/=np.linalg.norm(psi)
    b=sre_bruteforce(psi,N,n=3); f=float(stabilizer_renyi_entropy(jnp.asarray(psi),N,3))
    print(f"  N={N}  M3 brute={b:.8f}  fwht={f:.8f}  diff={abs(b-f):.1e}  {'PASS' if abs(b-f)<1e-9 else 'FAIL'}")

print("\n=== C. analytic anchors ===")
for N in (2,4,6,8):
    plus=jnp.ones(2**N,dtype=jnp.complex128)/np.sqrt(2**N)
    T=np.diag([1,np.exp(1j*np.pi/4)])
    t=(np.ones(2**N,dtype=complex)/np.sqrt(2**N)).reshape((2,)*N)
    for q in range(N): t=np.moveaxis(np.tensordot(T,np.moveaxis(t,q,0),axes=([1],[0])),0,q)
    m0=float(stabilizer_renyi_entropy(plus,N)); mT=float(stabilizer_renyi_entropy(jnp.asarray(t.reshape(-1)),N))
    print(f"  N={N}: M2(|+>^N)={m0:+.2e} (want 0)   M2(T|+>^N)={mT:.6f} (want {0.415*N:.6f})  "
          f"{'PASS' if abs(m0)<1e-9 and abs(mT-0.415*N)<2e-3 else 'FAIL'}")

print("\n=== D. Clifford invariance, N=6, 200 random Cliffords ===")
G=load_clifford_group(); N=6
t=(np.ones(2**N,dtype=complex)/np.sqrt(2**N)).reshape((2,)*N)
T=np.diag([1,np.exp(1j*np.pi/4)])
for q in range(N): t=np.moveaxis(np.tensordot(T,np.moveaxis(t,q,0),axes=([1],[0])),0,q)
psi=t.reshape(-1); m_before=float(stabilizer_renyi_entropy(jnp.asarray(psi),N))
def ap2(psi,U,a,b,N):
    t=psi.reshape((2,)*N); t=np.moveaxis(t,[a,b],[0,1]).reshape(4,-1)
    t=(U@t).reshape((2,2)+(2,)*(N-2)); return np.moveaxis(t,[0,1],[a,b]).reshape(-1)
for _ in range(200):
    a=rng.integers(N-1); psi=ap2(psi,G[rng.integers(len(G))],a,a+1,N)
m_after=float(stabilizer_renyi_entropy(jnp.asarray(psi),N))
print(f"  M2 before={m_before:.10f}  after={m_after:.10f}  drift={abs(m_before-m_after):.2e}  "
      f"{'PASS' if abs(m_before-m_after)<1e-8 else 'FAIL'}")

print("\n=== E. Haar-random states: M2 = log2((2^N+3)/4) exactly ===")
for N in (6,8):
    ms=[]
    for _ in range(8):
        p=rng.normal(size=2**N)+1j*rng.normal(size=2**N); p/=np.linalg.norm(p)
        ms.append(float(stabilizer_renyi_entropy(jnp.asarray(p),N)))
    print(f"  N={N}: <M2> = {np.mean(ms):.3f} +- {np.std(ms):.3f}   log2((2^N+3)/4) = {np.log2((2**N+3)/4):.3f}")
