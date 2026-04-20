import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
alpha = 10
n = 2
tmax = 50
def ODEs(t, y):
    p1, p2 = y
    dp1_dt = alpha / (1 + p2**n) - p1
    dp2_dt = alpha / (1 + p1**n) - p2
    return [dp1_dt, dp2_dt]
def deterministic(p11, p21):
    sol = solve_ivp(ODEs, [0, tmax], [p11, p21],
                    t_eval=np.linspace(0, tmax, 1000))
    return sol.t, sol.y[0], sol.y[1]
def gillespie(NP1_0, NP2_0):
    NP1 = NP1_0
    NP2 = NP2_0
    t = 0
    time = [t]
    P1 = [NP1]
    P2 = [NP2]
    while t < tmax:
        a1 = alpha / (1 + NP2**n)   
        a2 = NP1                    
        a3 = alpha / (1 + NP1**n)  
        a4 = NP2                    
        a0 = a1 + a2 + a3 + a4
        if a0 == 0:
            break
        r1, r2 = np.random.rand(), np.random.rand()
        tau = (1 / a0) * np.log(1 / r1)
        t += tau
        if r2 * a0 < a1:
            NP1 += 1
        elif r2 * a0 < a1 + a2:
            NP1 -= 1
        elif r2 * a0 < a1 + a2 + a3:
            NP2 += 1
        else:
            NP2 -= 1
        NP1 = max(NP1, 0)
        NP2 = max(NP2, 0)
        time.append(t)
        P1.append(NP1)
        P2.append(NP2)
    return np.array(time), np.array(P1), np.array(P2)
p11, p21 = 8, 2
t1, p1d1 , p2d1 = deterministic(p11, p21)
t1s1, p1s1, p2s1 = gillespie(p11, p21)
p11, p21 = 2, 8
td1, p1d2, p2d2 = deterministic(p11, p21)
ts2, p1s2, p2s2 = gillespie(p11, p21)
plt.figure(figsize=(10, 5))
plt.plot(t1, p1d1 , '--', label='p1 deterministic')
plt.plot(t1, p2d1, '--', label='p2 deterministic')
plt.step(t1s1, p1s1, where='post', label='gillespie P1')
plt.step(t1s1, p2s1, where='post', label='gillespie P2')
plt.title('Case 1: NP1(0) > NP2(0)')
plt.xlabel('Time')
plt.ylabel('Molecule Count')
plt.legend()
plt.grid()
plt.figure(figsize=(10, 5))
plt.plot(td1, p1d2, '--', label='p1 deterministic')
plt.plot(td1, p2d2, '--', label='p2 deterministic')
plt.step(ts2, p1s2, where='post', label='gillespie P1')
plt.step(ts2, p2s2, where='post', label='gillespie P2')
plt.title('Case 2: NP1(0) < NP2(0)')
plt.xlabel('Time')
plt.ylabel(' Molecule Count')
plt.legend()
plt.grid()
plt.show()
