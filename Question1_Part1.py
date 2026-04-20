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
ts, te = 0, 10
dt = 1   # big dt to show in graph
points = np.arange(ts, te + dt, dt)
y01 = np.array([5, 1])   # p1 > p2
y02 = np.array([1, 5])   # p1 < p2
def euler(y0):
    y = np.zeros((len(points), 2))
    y[0] = y0
    for i in range(len(points) - 1):
        p1, p2 = y[i]
        deriv = ODES(0, [p1, p2])
        y[i+1] = y[i] + dt * np.array(deriv)
    return y
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
sol1 = solve_ivp(ODES, [ts, te], y01, t_eval=points)
euler1 = euler(y01)
rk4_1 = RK4(y01)
sol2 = solve_ivp(ODES, [ts, te], y02, t_eval=points)
euler2 = euler(y02)
rk4_2 = RK4(y02)

plt.figure(figsize=(10, 5))
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
plt.figure(figsize=(10, 5))
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
plt.figure(figsize=(7, 7))
p_range = np.linspace(0, 10, 100)
nc1 = alpha / (1 + p_range**n) 
nc2 = alpha / (1 + p_range**n) 
plt.plot(nc1, p_range, 'g-', label='dp1/dt = 0') 
plt.plot(p_range, nc2, 'r-', label='dp2/dt = 0') 
p1_vals, p2_vals = np.meshgrid(np.linspace(0, 10, 20), np.linspace(0, 10, 20))
dp1 = alpha / (1 + p2_vals**n) - p1_vals
dp2 = alpha / (1 + p1_vals**n) - p2_vals
mag = np.sqrt(dp1**2 + dp2**2)
mag[mag == 0] = 1
plt.quiver(p1_vals, p2_vals, dp1/mag, dp2/mag, color='gray', alpha=0.3)
plt.plot(sol1.y[0], sol1.y[1], 'k-', label='Trajectory IC1')
plt.plot(sol2.y[0], sol2.y[1], 'C1-', label='Trajectory IC2')
plt.plot([0.1, 9.9], [9.9, 0.1], 'bo', label='Stable Node') 
plt.plot(2, 2, 'gx', markersize=10, label='Saddle Point')     
plt.xlabel('p1')
plt.ylabel('p2')
plt.title('Phase Portrait with Nullclines and Fixed Points')
plt.xlim(0, 10)
plt.ylim(0, 10)
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.show()
