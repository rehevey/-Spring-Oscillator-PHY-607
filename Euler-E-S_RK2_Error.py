import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

# Setup
omega_sq = 1
x0, v0 = 0, 2
dt = 0.01
t_max = 1000
n_steps = int(t_max / dt)
t_euler = np.linspace(0, t_max, n_steps)

# Euler explicit
x_expl = np.zeros(n_steps)
v_expl = np.zeros(n_steps)
Ee = np.zeros(n_steps)
x_expl[0], v_expl[0] = x0, v0
Ee[0] = 0.5*v_expl[0]**2 + 0.5*omega_sq*x_expl[0]**2

for i in range(n_steps - 1):
    a = -omega_sq * x_expl[i]
    x_expl[i+1] = x_expl[i] + v_expl[i] * dt
    v_expl[i+1] = v_expl[i] + a * dt
    Ee[i+1] = 0.5*v_expl[i+1]**2 + 0.5*omega_sq*x_expl[i+1]**2

plt.plot(t_euler, x_expl)
plt.ylabel("Position")
plt.xlabel("Time")
plt.title('Euler Explicit', fontsize=20)
plt.show()

# Euler symplectic
x_sympl = np.zeros(n_steps)
v_sympl = np.zeros(n_steps)
Es = np.zeros(n_steps)
x_sympl[0], v_sympl[0] = x0, v0
Es[0] = 0.5*v_sympl[0]**2 + 0.5*omega_sq*x_sympl[0]**2

for i in range(n_steps - 1):
    a = -omega_sq * x_sympl[i]
    v_sympl[i+1] = v_sympl[i] + a * dt
    x_sympl[i+1] = x_sympl[i] + v_sympl[i+1] * dt
    Es[i+1] = 0.5*v_sympl[i+1]**2 + 0.5*omega_sq*x_sympl[i+1]**2

plt.plot(t_euler, x_sympl)
plt.ylabel("Position")
plt.xlabel("Time")
plt.title('Euler Symplectic', fontsize=20)
plt.show()

# RK2 -- midpoint method 

x_rk2 = np.zeros(n_steps)
v_rk2 = np.zeros(n_steps)
E_rk2 = np.zeros(n_steps)
x_rk2[0], v_rk2[0] = x0, v0
E_rk2[0] = 0.5*v_rk2[0]**2 + 0.5*omega_sq*x_rk2[0]**2

def f(x, v):
    return v, -omega_sq*x    # Returns (dx/dt, dv/dt) = (v, a)

for i in range(n_steps - 1):
    # slope evaluation at the start
    k1x, k1v = f(x_rk2[i], v_rk2[i])

    #use slope to estimate midpoint 
    x_mid = x_rk2[i] + 0.5*dt*k1x
    v_mid = v_rk2[i] + 0.5*dt*k1v

    #evaluate the midpoint slope
    k2x, k2v = f(x_mid, v_mid)

    #next step 
    x_rk2[i+1] = x_rk2[i] + dt*k2x
    v_rk2[i+1] = v_rk2[i] + dt*k2v

    E_rk2[i+1] = 0.5*v_rk2[i+1]**2 + 0.5*omega_sq*x_rk2[i+1]**2 

plt.plot(t_euler, x_rk2)
plt.ylabel("Position")
plt.xlabel("Time")
plt.title('RK2 (Midpoint Method)', fontsize=20)
plt.show()

# Energy comparison

plt.plot(t_euler, Ee, label='Explicit Euler')
plt.plot(t_euler, Es, label='Symplectic Euler')
plt.plot(t_euler, E_rk2, label='RK2')
plt.ylabel("Energy")
plt.xlabel("Time")
plt.title('Energy Conservation Comparison', fontsize=20)
plt.legend()
plt.show()

# Exact solution

def exact_solution(t):
    return 2*np.sin(t)   # since omega=1, x0=0, v0=2 → x(t) = 2*sin(t)

# Running for errors 


def run_explicit(dt):
    n = int(t_max/dt)
    x, v = np.zeros(n), np.zeros(n)
    x[0], v[0] = x0, v0
    for i in range(n-1):
        a = -omega_sq*x[i]
        x[i+1] = x[i] + v[i]*dt
        v[i+1] = v[i] + a*dt
    return x[-1]

def run_symplectic(dt):
    n = int(t_max/dt)
    x, v = np.zeros(n), np.zeros(n)
    x[0], v[0] = x0, v0
    for i in range(n-1):
        a = -omega_sq*x[i]
        v[i+1] = v[i] + a*dt
        x[i+1] = x[i] + v[i+1]*dt
    return x[-1]

def run_rk2(dt):
    n = int(t_max/dt)
    x, v = np.zeros(n), np.zeros(n)
    x[0], v[0] = x0, v0
    for i in range(n-1):
        k1x, k1v = f(x[i], v[i])
        x_mid = x[i] + 0.5*dt*k1x
        v_mid = v[i] + 0.5*dt*k1v
        k2x, k2v = f(x_mid, v_mid)
        x[i+1] = x[i] + dt*k2x
        v[i+1] = v[i] + dt*k2v
    return x[-1]

dt_values = np.array([0.5, 0.1, 0.05, 0.01, 0.005, 0.001]) #time range
x_exact = 2*np.sin(t_max)   # exact answer at t_max

errors_expl = []
errors_sympl = []
errors_rk2 = []

for dt in dt_values:
    errors_expl.append(abs(run_explicit(dt) - x_exact))
    errors_sympl.append(abs(run_symplectic(dt) - x_exact))
    errors_rk2.append(abs(run_rk2(dt) - x_exact))

# Error graph - log scale

plt.plot(dt_values, errors_expl, 'o-', label='Explicit Euler')
plt.plot(dt_values, errors_sympl, 'o-', label='Symplectic Euler')
plt.plot(dt_values, errors_rk2, 'o-', label='RK2')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Time step (dt)')
plt.ylabel('Error at t_max')
plt.title('Change in Error Based on Time Step', fontsize=20)
plt.legend()
plt.show()
