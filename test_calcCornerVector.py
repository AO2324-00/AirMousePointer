import numpy as np
import vector
from matplotlib import pyplot as plt

O = vector.Vector3D({'x': 1, 'y':0, 'z':0})
V = vector.Vector3D({'x': -1, 'y':0, 'z':0})
P = vector.Vector3D({'x': 0, 'y':-1, 'z':0})

t = np.arange( -10, 10, 1)
#O = vector.Vector3D({'x': 1, 'y':1, 'z':1})
#V = vector.Vector3D({'x': -1, 'y':-1, 'z':-1})
#P = vector.Vector3D({'x': 0, 'y':-1, 'z':0})

corner = vector.calcCornerVector(P, O, V)
print(corner)

PC = vector.calcVector3D(P, corner)
VC = vector.calcVector3D(V, corner)
print(vector.calcDotProduct(PC.parseArray(), VC.parseArray()))

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter([O.x, P.x], [O.y, P.y], [O.z, P.z], s=10, c="blue")
ax.scatter([corner.x], [corner.y], [corner.z], s=40, c="red")
X = O.x+V.x*t
Y = O.y+V.y*t
Z = O.z+V.z*t
ax.plot(X, Y, Z, color='blue')
X = P.x+(corner.x-P.x)*t
Y = P.y+(corner.y-P.y)*t
Z = P.z+(corner.z-P.z)*t
ax.plot(X, Y, Z, color='red', linestyle="dashed")
plt.show()