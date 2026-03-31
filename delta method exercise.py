import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-5, 3, 1000) #start, finish, number of points

y = x**2 +2*x + 2
def f(my_x):
    my_y =  my_x**2 + 2*my_x + 2
    return my_y

y = f(x)
print('y at x=1 is:', f(1))
print('y at x=1.1 is:', f(1.1))

m = (f(1.1)-f(1)/(1.1-1))

b = f(1) - m*1

line_y = m*x + b

fig, ax = plt.subplots()
plt.axvline(x=0, color='gray')
plt.axhline(y=0, color='gray')
plt.scatter(1,5 ,color='blue')
plt.scatter(1.1,5.41 ,color='red', zorder=3)
plt.ylim(-10, 30)
plt.plot(x,line_y, color='green')
_=ax.plot(x,y)

plt.show()