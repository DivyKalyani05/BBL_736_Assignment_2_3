import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

alpha = 10
n = 2
t_max = 50


def ODEs(t, y):
    p1, p2 = y
    dp1_dt = alpha / (1 + p2**n) - p1
    dp2_dt = alpha / (1 + p1**n) - p2
    return [dp1_dt, dp2_dt]


def solve_deterministic(p1_0, p2_0):
    sol = solve_ivp(ODEs, [0, t_max], [p1_0, p2_0],
                    t_eval=np.linspace(0, t_max, 1000))
    return sol.t, sol.y[0], sol.y[1]


def gillespie(NP1_0, NP2_0):
    NP1 = NP1_0
    NP2 = NP2_0

    t = 0

    time = [t]
    P1 = [NP1]
    P2 = [NP2]

    while t < t_max:

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



p1_0, p2_0 = 8, 2
t_det1, p1_det1, p2_det1 = solve_deterministic(p1_0, p2_0)
t_sto1, P1_sto1, P2_sto1 = gillespie(p1_0, p2_0)


p1_0, p2_0 = 2, 8
t_det2, p1_det2, p2_det2 = solve_deterministic(p1_0, p2_0)
t_sto2, P1_sto2, P2_sto2 = gillespie(p1_0, p2_0)


plt.figure(figsize=(10, 5))

plt.plot(t_det1, p1_det1, '--', label='p1 deterministic')
plt.plot(t_det1, p2_det1, '--', label='p2 deterministic')

plt.step(t_sto1, P1_sto1, where='post', label='NP1 stochastic')
plt.step(t_sto1, P2_sto1, where='post', label='NP2 stochastic')

plt.title('Case 1: NP1(0) > NP2(0)')
plt.xlabel('Time')
plt.ylabel('Expression Level / Molecule Count')
plt.legend()
plt.grid()


plt.figure(figsize=(10, 5))

plt.plot(t_det2, p1_det2, '--', label='p1 deterministic')
plt.plot(t_det2, p2_det2, '--', label='p2 deterministic')

plt.step(t_sto2, P1_sto2, where='post', label='NP1 stochastic')
plt.step(t_sto2, P2_sto2, where='post', label='NP2 stochastic')

plt.title('Case 2: NP1(0) < NP2(0)')
plt.xlabel('Time')
plt.ylabel('Expression Level / Molecule Count')
plt.legend()
plt.grid()

plt.show()
