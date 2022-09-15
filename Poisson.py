from __future__ import print_function
from fenics import *
import matplotlib.pyplot as plt
import numpy as np

"""
-Laplace(u) = f    
u = u_D  

U_d = 3 + x^2 + 1y^2
f = -1
"""

mesh = UnitSquareMesh(8, 8) # Criamos uma malha 8x8 triangular
A = FunctionSpace(mesh, 'P', 1) # Criamos a função do espaço
U_d = Expression('3 + x[0]*x[0] + 1*x[1]*x[1]', degree=2) # 0) Definimos a condição de contorno 
def Contorno(x, contorno): # 1)
    return contorno
BC = DirichletBC(A, U_d, Contorno) # 2)
# Forma variacional do problema
u = TrialFunction(A); v = TestFunction(A); f = Constant(-1.0)
g = dot(grad(u), grad(v))*dx
K = f*v*dx
u = Function(A); solve(g == K, u, BC) # Aqui Calculamos a solução
plot(u); plot(mesh) # Visualização a solução (distribuição de calor) sobre a malha
vtkfile = File('Sol_Poisson/solution.pvd'); vtkfile << u # Salvamos no formato VTK pois o visualizaremos a solução no ParaView
erro = errornorm(U_d, u, 'L2') # Erro (pela norma L²) 
ValorVertice_U_d = U_d.compute_vertex_values(mesh); ValorVertice_u = u.compute_vertex_values(mesh) # Valores dos vértices
ErroMaximo = np.max(np.abs(ValorVertice_U_d - ValorVertice_u)) # Erro maximo

print('error_L2  =', erro)
print('error_max =', ErroMaximo)

plt.show()


