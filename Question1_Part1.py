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
