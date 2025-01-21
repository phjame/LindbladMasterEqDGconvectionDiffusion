"""
FEniCS program for a convection-diffusion system representing a master eq.
for a density matrix in transformed position coordinates:
F{w}(x,y,t)=(2pi)^d rho(x-y/2,x+y/2,t). 
The complex-valued function has been decomposed into real and imaginary
parts, 
F{w} = R+iI = (R,I) = u(x,y,t) =(u1,u2)
obtaining then a system of two equations. The convective part
is mostly related to the Hamiltonian transport (except for a term of QFP),
and the diffusive terms are related to the Quantum Fokker-Planck operator.
For simplicity in this first study we will deal with a 1D case (d=1).
In a first trial the potential V under consideration will be harmonic. 
In this case, the system has the following gradient form:

 R_t + (0,y).grad(R) =  I(dV) - Ry^2 + \partial_x^2 R + \partial_{xy} I
 I_t + (0,y).grad(I) = -R(dV) - Iy^2 + \partial_x^2 I - \partial_{xy} R

or in divergence form as
 
 R_t + div(R(0,y) - A grad(R) - B grad(I) ) = R(1-y)^2 + IdV
 I_t + div(I(0,y) - A grad(I) + B grad(R) ) = I(1-y)^2 - RdV

defining the diffusion matrices
    1 0
A =(   ),
    0 0

    0  1/2
B =(      ).
    1/2 0

and where grad=(\partial_x,\partial_y). 
Notice that properly speaking there's not a source term as the RHS terms depend on (R,I).
"""
import time
import os
import math
import numpy
from dolfin import *
from fenics import *
from matplotlib import pyplot
import matplotlib.pyplot as plt
# get file name
fileName = os.path.splitext(__file__)[0]

parameters['form_compiler']['cpp_optimize'] = True
parameters['form_compiler']['optimize'] = True
parameters["ghost_mode"] = "shared_facet"
#parameters["plotting_backend"] = "matplotlib"
epst = 1.0
epsd = 1.0
# Parameters
DiffA = Constant(((1.0*epst, 0.0,),
                   (0.0, 0.0)))
# Diffusion matrices A and B
DiffB = Constant(((0.0, 0.5*epsd,),
                    (0.5*epsd, 0.0)))
Diff = Constant(1)
t_end = 50.0 
Nx = 128 
Ny = Nx
#Problem parameters
Lx = 100#Constant(1)
Kfreq =1/Lx# Constant(1/Lx)
J =1# Constant(1)
Xmax = Lx
Ymax = (J+0.25)*pi/Kfreq
dt=(2*Ymax/Ny)/(Ymax) #CFL condition is: dt <= min_i {dx_i/|u_i|} 
#dt = 0.01
# Create mesh and define function space
#mesh = RectangleMesh(Point(-Lx,-(J+0.25)*pi/Kfreq),Point(Lx,(J+0.25)*pi/Kfreq),40,40)#, 'crossed')
mesh = RectangleMesh(Point(-Xmax,-Ymax),Point(Xmax,Ymax),Nx,Ny)#, 'crossed')
vals = numpy.empty((Nx+1)*(Ny+1), dtype=float)

# Define function spaces
V = VectorFunctionSpace(mesh, "DG", 1)
# substitute initial condition with the related eigenstate for benchmarking
ic= Expression(("(2*sqrt(pi)/(1*L))*exp(-(pow(x[0]/L,2) +pow(x[1]/(2*L),2) ))","0"), domain=mesh, degree=1, L=Lx)
ic1= Expression(("(2*sqrt(pi)/(1*L))*exp(-(pow(x[0]/L,2) +pow(x[1]/(2*L),2) ))"), domain=mesh, degree=1, L=Lx)
ic2= Expression("0", domain=mesh, degree=1, L=Lx)

b = Expression(("0.0","x[1]"), domain=mesh, degree=1) #transport vector
# BC adequate for our transformed master equation problem
u1_ss = Expression("(1/(sqrt(6*pi)))*exp(-(pow(x[0]/sqrt(6),2) +pow(sqrt(5)*x[1]/2,2) ))*cos(x[0]*x[1]/sqrt(6))", degree=4, domain=mesh)
u2_ss = Expression("(1/(sqrt(6*pi)))*exp(-(pow(x[0]/sqrt(6),2) +pow(sqrt(5)*x[1]/2,2) ))*sin(x[0]*x[1]/sqrt(6))", degree=4, domain=mesh)
u1_D = Expression("0", degree=4, domain=mesh)
u2_D = Expression("0", degree=4, domain=mesh)
bc=DirichletBC(V.sub(0),u1_D, DomainBoundary(), method="geometric")
bc=DirichletBC(V.sub(1),u2_D, DomainBoundary(), method="geometric")

#Expressions useful for some source terms
eta2 = Expression("x[1]*x[1]", degree=2) # eta^2: exponential decay representing Fourier transformed diffusion
# V = x^2/2 # normalized harmonic potential
# deltaV = V(x+y/2) - V(x-y/2)
# deltaV = 0.5[(x+y/2)^2 - (x-y/2)^2]=0.5[(x+y/2) + (x-y/2)][(x+y/2) - (x-y/2)] =xy
#deltaV = Expression("x[0]*x[1]",degree=2) #Quadratic potential as above
#deltaV = Expression("x[1]", degree=1) # V(x)=x linear potential
deltaV = Expression("pow(x[0]+0.5*x[1],4)-pow(x[0]-0.5*x[1],4)", degree=4) # V(x)=x^4 quartic potential

# Define unknown and test function(s)
v = TestFunction(V)#W)
v_1, v_2 = split(v) #TestFunctions(V) 
#v_1, v_2 = TestFunctions(V)
uaux1 = TrialFunction(V)#W)
uaux2 = TrialFunction(V)#W)
u = TrialFunction(V)#W)
#u = Function(W)
uaux1_1, uaux1_2 = split(uaux1)
uaux2_1, uaux2_2 = split(uaux2)
u_1, u_2 = split(u)
#u_2 = TrialFunction(V)

u0 = Function(V)#W)
u0 = project(ic, V)
u0_1, u0_2 = u0.split(u0)

#projecterror1_L1 = assemble(abs(ic1 - u0_1)*dx(mesh))
#print('Projection L1error-u1 = ', projecterror1_L1)
#projecterror2_L1 = assemble(abs(ic2 - u0_2)*dx(mesh))
#print('Projection L1error-u2 = ', projecterror2_L1)
projecterror1_L2 = errornorm(ic1, u0_1, 'L2')
print('Projection L2error-u1 = ', projecterror1_L2)
projecterror2_L2 = errornorm(ic2, u0_2, 'L2')
print('Projection L2error-u2 = ', projecterror2_L2)
#error1_L1 = assemble(abs(u1_D - u0_1)*dx(mesh))
#print('Initial L1error-u1 = ', error1_L1)
#error2_L1 = assemble(abs(u2_D - u0_2)*dx(mesh))
#print('Initial L1error-u2 = ', error2_L1)
error1_L2 = errornorm(u1_ss, u0_1, 'L2')
print('Initial L2error-u1 = ', error1_L2)
error2_L2 = errornorm(u2_ss, u0_2, 'L2')
print('Initial L2error-u2 = ', error2_L2)

# STABILIZATION
h = CellDiameter(mesh)
n = FacetNormal(mesh)
alpha = Constant(1e0) # penalty constant...
# Theta zero makes it explicit FWD Euler
theta = Constant(0.5) #constant for theta-method in time.

# ( dot(v, n) + |dot(v, n)| )/2.0
bn = (dot(b, n) + abs(dot(b, n)))/2.0

def a(u,v) :
        # Bilinear form
        a_int = dot(grad(v[0]), (DiffA)*grad(u[0]) - b*u[0])*dx + dot(grad(v[1]), (DiffA)*grad(u[1]) - b*u[1])*dx \
        	+ dot(grad(v[0]), (DiffB)*grad(u[1]))*dx - dot(grad(v[1]), (DiffB)*grad(u[0]))*dx \
	- inner(u[1]*deltaV, v[0])*dx + inner(u[0]*eta2, v[0])*dx \
	+ inner(u[0]*deltaV, v[1])*dx + inner(u[1]*eta2, v[1])*dx
        
        a_fac = (alpha/avg(h))*dot(DiffA*jump(u[0], n), jump(v[0], n))*dS \
        	+ (alpha/avg(h))*dot(DiffB*jump(u[1], n), jump(v[0], n))*dS \
                - dot(DiffA*avg(grad(u[0])), jump(v[0], n))*dS \
                - dot(DiffB*avg(grad(u[1])), jump(v[0], n))*dS \
                - dot(DiffA*jump(u[0], n), avg(grad(v[0])))*dS \
                - dot(DiffB*jump(u[1], n), avg(grad(v[0])))*dS \
        	+ (alpha/avg(h))*dot(DiffA*jump(u[1], n), jump(v[1], n))*dS \
                - (alpha/avg(h))*dot(DiffB*jump(u[0], n), jump(v[1], n))*dS \
                - dot(DiffA*avg(grad(u[1])), jump(v[1], n))*dS \
                + dot(DiffB*avg(grad(u[0])), jump(v[1], n))*dS \
                - dot(DiffA*jump(u[1], n), avg(grad(v[1])))*dS \
                + dot(DiffB*jump(u[0], n), avg(grad(v[1])))*dS

#CHECK THE FLUXES        
        a_vel =  dot(jump(v[0]), bn('+')*u[0]('+') - bn('-')*u[0]('-') )*dS  + dot(v[0], bn*u[0])*ds \
        	+dot(jump(v[1]), bn('+')*u[1]('+') - bn('-')*u[1]('-') )*dS  + dot(v[1], bn*u[1])*ds
        
        a = a_int + a_fac + a_vel
        return a
#################### Define variational forms
a0=a(u0,v)

A = (1/dt)*inner(u[0], v[0])*dx - (1/dt)*inner(u0[0],v[0])*dx + theta*a(u,v) \
  + (1/dt)*inner(u[1], v[1])*dx - (1/dt)*inner(u0[1],v[1])*dx + (1-theta)*a(u0,v)

F = A

# Create files for storing results 
file = File("results_%s/u.pvd" % (fileName))

u = Function(V)#W)
#u_2 = Function(V)
ffc_options = {"optimize": True, "quadrature_degree": 8}
problem = LinearVariationalProblem(lhs(F),rhs(F), u, [bc], form_compiler_parameters=ffc_options)
solver  = LinearVariationalSolver(problem)


u.assign(u0)
#u_2.assign(u0_2)
u.rename("u", "u")
#u_2.rename("u_2", "u_2")
#u_1, u_2 = split(u)


# Time-stepping
t = 0.0

#file.write(u, 0)
u_1, u_2 = u.split(u)
file << u_2

while t < t_end:

	print("t =", t, "end t=", t_end, "dt =", dt)

	# Compute
	solver.solve()
	#plot(u)
	# Save to file
	#file << u
	u_1, u_2 = u.split(u)
	file << u_2
#	error1_L1 = assemble(abs(u1_D - u_1)*dx(mesh))
#	print('L1error-u1 = ', error1_L1)
#	error2_L1 = assemble(abs(u2_D - u_2)*dx(mesh))
#	print('L1error-u2 = ', error2_L1)
	error1_L2 = errornorm(u1_ss, u_1, 'L2')
	print('L2error-u1 = ', error1_L2)
	error2_L2 = errornorm(u2_ss, u_2, 'L2')
	print('L2error-u2 = ', error2_L2)
	# Move to next time step
	u0.assign(u)
	#u0_2.assign(u_2)
	t += dt

#plt.show()
#u_1, u_2 = u.split(u)
#error1_L2 = errornorm(u1_D, u_1, 'L2')
#error2_L2 = errornorm(u2_D, u_2, 'L2')
#print('L2error-u1 = ', error1_L2)
#print('L2error-u2 = ', error2_L2)
