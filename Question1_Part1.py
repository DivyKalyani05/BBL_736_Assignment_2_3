import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


alpha = 10
n = 2

def ODES(t, y):
    p1, p2 = y
    dp1_dt = alpha / (1 + p2**n) - p1
    dp2_dt = alpha / (1 + p1**n) - p2
    return [dp1_dt, dp2_dt]


t_st = 0
t_end = 10
dt = 1   # big to show oscillation
points = np.arange(t_st, t_end, dt)

# 2 required conditions
y0_1 = np.array([5, 1])   # p1 > p2
y0_2 = np.array([1, 5])   # p1 < p2


#ODE Solver
sol1 = solve_ivp(ODES, [t_st, t_end], y0_1, t_eval=points)
sol2 = solve_ivp(ODES, [t_st, t_end], y0_2, t_eval=points)

#euler
def euler(y0):
    y = np.zeros((len(points), 2))
    y[0] = y0
    
    for i in range(len(points) - 1):
        p1, p2 = y[i]
        
        dp1 = alpha / (1 + p2**n) - p1
        dp2 = alpha / (1 + p1**n) - p2
        
        y[i+1, 0] = p1 + dt * dp1
        y[i+1, 1] = p2 + dt * dp2
    
    return y

euler1 = euler(y0_1)
euler2 = euler(y0_2)

# RK4 
def RK4(y0):
    y = np.zeros((len(points), 2))
    y[0] = y0
    
    for i in range(len(points) - 1):
        p = y[i]
        
        k1 = np.array(ODES(0, p))
        k2 = np.array(ODES(0, p + dt * k1 / 2))
        k3 = np.array(ODES(0, p + dt * k2 / 2))
        k4 = np.array(ODES(0, p + dt * k3))
        
        y[i+1] = p + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
    
    return y

rk4_1 = RK4(y0_1)
rk4_2 = RK4(y0_2)




plt.figure(figsize=(10,5))

plt.plot(sol1.t, sol1.y[0], label='p1 (ODE)')
plt.plot(sol1.t, sol1.y[1], '--', label='p2 (ODE)')

plt.plot(points, euler1[:,0], ':', label='p1 (Euler)')
plt.plot(points, euler1[:,1], ':', label='p2 (Euler)')

plt.plot(points, rk4_1[:,0], '-.', label='p1 (RK4)')
plt.plot(points, rk4_1[:,1], '-.', label='p2 (RK4)')

plt.xlabel('Time')
plt.ylabel('Proteins')
plt.title('p1(0) > p2(0)')
plt.legend()
plt.show()


plt.figure(figsize=(10,5))

plt.plot(sol2.t, sol2.y[0], label='p1 (ODE)')
plt.plot(sol2.t, sol2.y[1], '--', label='p2 (ODE)')

plt.plot(points, euler2[:,0], ':', label='p1 (Euler)')
plt.plot(points, euler2[:,1], ':', label='p2 (Euler)')

plt.plot(points, rk4_2[:,0], '-.', label='p1 (RK4)')
plt.plot(points, rk4_2[:,1], '-.', label='p2 (RK4)')

plt.xlabel('Time')
plt.ylabel('Proteins')
plt.title('p1(0) < p2(0)')
plt.legend()
plt.show()




p1_vals = np.linspace(0, 10, 20)
p2_vals = np.linspace(0, 10, 20)

P1, P2 = np.meshgrid(p1_vals, p2_vals)


dP1 = np.zeros_like(P1)
dP2 = np.zeros_like(P2)

for i in range(P1.shape[0]):
    for j in range(P1.shape[1]):
        dp = ODES(0, [P1[i, j], P2[i, j]])
        dP1[i, j] = dp[0]
        dP2[i, j] = dp[1]

mag = np.sqrt(dP1**2 + dP2**2)


mag[mag == 0] = 1

dP1_norm = dP1 / mag
dP2_norm = dP2 / mag

plt.figure(figsize=(7,7))


plt.quiver(P1, P2, dP1_norm, dP2_norm)


plt.plot(sol1.y[0], sol1.y[1], label='Trajectory IC1')
plt.plot(sol2.y[0], sol2.y[1], label='Trajectory IC2')


plt.xlabel('p1')
plt.ylabel('p2')
plt.title('Phase Portrait')

plt.xlim(0,10)
plt.ylim(0,10)

plt.legend()
plt.show()
