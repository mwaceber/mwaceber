import torch as tr

x = tr.tensor(0.).requires_grad_()
y = tr.tensor(0.).requires_grad_()

#print("x:", x)
#print("y:", y)

def f(x, y):
    return x**2 - y**2

z = f(x, y) 
#print("z:"z)
z.backward()

print("dz/dx:", x.grad)
print("dz/dy:", y.grad)