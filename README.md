# LindbladMasterEqDGconvectionDiffusion
DG for Lindblad master eq. as convection-diffusion
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
A =(1 0; 0 0)
B = (0  1/2; 1/2 0)

and where grad=(\partial_x,\partial_y). 
Notice that properly speaking there's not a source term as the RHS terms depend on (R,I).
"""
