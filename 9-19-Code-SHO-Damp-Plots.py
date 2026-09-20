import numpy as np 
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

t = np.linspace(0,15,1000)
omega_sq = 1
y = [0,2] #y[0]=x and y[1]=v

def harmonic(t,y):
    solution = [y[1],-omega_sq*y[0]]
    return solution
sho = solve_ivp(harmonic, [0,1000], y0 = y, t_eval = t) 

#simple harminic oscillator plot
plt.plot(t,sho.y[0])
plt.ylabel("Position")
plt.xlabel("Time")
plt.title('SHO', fontsize = 20)
plt.show()

t = np.linspace(0,15,1000)
y = [0,1]
gamma = 1
omega_sqr = 100

def sho(t,y):
    solution = (y[1],(-gamma*y[1]-omega_sqr*y[0]))
    return solution
solution = solve_ivp(sho, [0,1000], y0 = y, t_eval = t)

#damped plot
plt.plot(t,solution.y[0])
plt.ylabel("Position")
plt.xlabel("Time")
plt.title('Damped Oscillator', fontsize = 20)
plt.show() 

#both plot
plt.plot(t, sho.y[0], label='SHO')
plt.plot(t, solution.y[0], label='Damped Oscillator')
plt.ylabel("Position")
plt.xlabel("Time")
plt.title('SHO vs Damped Oscillator', fontsize=20)
plt.legend()
plt.show()
