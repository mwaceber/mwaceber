import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-10, 10, 1000) # start, finish, number of points

y = x**2 + 2*x + 2

def f(my_x):
    my_y = my_x**2 + 2*my_x + 2
    return my_y

y = f(x)

m = (37-10)/(5-2)
b = 37 - m*5
line_y = m*x + b

fig, ax = plt.subplots()
plt.axvline(x= 0, color='gray')
plt.axhline(y= 0, color='gray')
plt.scatter(2,10)
plt.scatter(5,37, color= 'red', zorder=3)
plt.ylim(-10, 150)
plt.plot(x,line_y, color= 'orange')
_=ax.plot(x,y)

plt.show()

#Delta method uses the difference between two points to
# estimate the slope

