import taichi as ti
import math

ti.init(arch=ti.cpu)

# global control
paused = True
damping_toggle = ti.field(ti.i32, ())
curser = ti.Vector.field(2, ti.f32, ())
picking = ti.field(ti.i32,())
using_auto_diff = False

# procedurally setting up the cantilever
init_x, init_y = 0.1, 0.6
N_x = 20
N_y = 4
# N_x = 2
# N_y = 2
N = N_x*N_y
N_edges = (N_x-1)*N_y + N_x*(N_y - 1) + (N_x-1) * \
    (N_y-1)  # horizontal + vertical + diagonal springs
N_triangles = 2 * (N_x-1) * (N_y-1)
dx = 1/32
curser_radius = dx/2

# physical quantities
m = 1
g = 9.8
YoungsModulus = ti.field(ti.f32, ())
PoissonsRatio = ti.field(ti.f32, ())
LameMu = ti.field(ti.f32, ())
LameLa = ti.field(ti.f32, ())

# time-step size (for simulation, 16.7ms)
h = 16.7e-3
# substepping
substepping = 100
# time-step size (for time integration)
dh = h/substepping

# simulation components
x = ti.Vector.field(2, ti.f32, N, needs_grad=True)
v = ti.Vector.field(2, ti.f32, N)
total_energy = ti.field(ti.f32, (), needs_grad=True)
grad = ti.Vector.field(2, ti.f32, N)
elements_Dm_inv = ti.Matrix.field(2, 2, ti.f32, N_triangles)
elements_V0 = ti.field(ti.f32, N_triangles)

# geometric components
triangles = ti.Vector.field(3, ti.i32, N_triangles)
edges = ti.Vector.field(2, ti.i32, N_edges)


def ij_2_index(i, j): return i * N_y + j


# -----------------------meshing and init----------------------------
@ti.kernel
def meshing():
    # setting up triangles
    for i,j in ti.ndrange(N_x - 1, N_y - 1):
        # triangle id
        tid = (i * (N_y - 1) + j) * 2
        triangles[tid][0] = ij_2_index(i, j)
        triangles[tid][1] = ij_2_index(i + 1, j)
        triangles[tid][2] = ij_2_index(i, j + 1)

        tid = (i * (N_y - 1) + j) * 2 + 1
        triangles[tid][0] = ij_2_index(i, j + 1)
        triangles[tid][1] = ij_2_index(i + 1, j + 1)
        triangles[tid][2] = ij_2_index(i + 1, j)

    # setting up edges
    # edge id
    eid_base = 0

    # horizontal edges
    for i in range(N_x-1):
        for j in range(N_y):
            eid = eid_base+i*N_y+j
            edges[eid] = [ij_2_index(i, j), ij_2_index(i+1, j)]

    eid_base += (N_x-1)*N_y
    # vertical edges
    for i in range(N_x):
        for j in range(N_y-1):
            eid = eid_base+i*(N_y-1)+j
            edges[eid] = [ij_2_index(i, j), ij_2_index(i, j+1)]

    eid_base += N_x*(N_y-1)
    # diagonal edges
    for i in range(N_x-1):
        for j in range(N_y-1):
            eid = eid_base+i*(N_y-1)+j
            edges[eid] = [ij_2_index(i+1, j), ij_2_index(i, j+1)]

@ti.kernel
def initialize():
    YoungsModulus[None] = 1e6
    paused = True
    # init position and velocity
    for i, j in ti.ndrange(N_x, N_y):
        index = ij_2_index(i, j)
        x[index] = ti.Vector([init_x + i * dx, init_y + j * dx])
        v[index] = ti.Vector([0.0, 0.0])

@ti.func
def compute_D(i):
    a = triangles[i][0]
    b = triangles[i][1]
    c = triangles[i][2]
    return ti.Matrix.cols([x[b] - x[a], x[c] - x[a]])

@ti.kernel
def initialize_elements():
    for i in range(N_triangles):
        Dm = compute_D(i)
        elements_Dm_inv[i] = Dm.inverse()
        elements_V0[i] = ti.abs(Dm.determinant())/2

# ----------------------core-----------------------------
@ti.func
def compute_R_2D(F):
    R, S = ti.polar_decompose(F, ti.f32)
    return R

@ti.kernel
def compute_gradient():
    # clear gradient
    for i in grad:
        grad[i] = ti.Vector([0, 0])

    # gradient of elastic potential
    for i in range(N_triangles):
        Ds = compute_D(i)
        F = Ds@elements_Dm_inv[i]
        # co-rotated linear elasticity
        R = compute_R_2D(F)
        Eye = ti.Matrix.cols([[1.0, 0.0], [0.0, 1.0]])
        # first Piola-Kirchhoff tensor
        P = 2*LameMu[None]*(F-R) + LameLa[None]*((R.transpose())@F-Eye).trace()*R
        #assemble to gradient
        H = elements_V0[i] * P @ (elements_Dm_inv[i].transpose())
        a,b,c = triangles[i][0],triangles[i][1],triangles[i][2]
        gb = ti.Vector([H[0,0], H[1, 0]])
        gc = ti.Vector([H[0,1], H[1, 1]])
        ga = -gb-gc
        grad[a] += ga
        grad[b] += gb
        grad[c] += gc     

@ti.kernel
def compute_total_energy():
    for i in range(N_triangles):
        Ds = compute_D(i)
        F = Ds @ elements_Dm_inv[i]
        # co-rotated linear elasticity
        R = compute_R_2D(F)
        Eye = ti.Matrix.cols([[1.0, 0.0], [0.0, 1.0]])
        element_energy_density = LameMu[None]*((F-R)@(F-R).transpose()).trace() + 0.5*LameLa[None]*(R.transpose()@F-Eye).trace()**2

        total_energy[None] += element_energy_density * elements_V0[i]   

@ti.kernel
def update():
    # perform time integration
    for i in range(N):
        # symplectic integration
        # elastic force + gravitation force, divding mass to get the acceleration
        if using_auto_diff:
            acc = -x.grad[i]/m - ti.Vector([0.0, g])
            v[i] += dh*acc
        else:
            acc = -grad[i]/m - ti.Vector([0.0, g])
            v[i] += dh*acc
        x[i] += dh*v[i]

    # explicit damping (ether drag)
    for i in v:
        if damping_toggle[None]:
            v[i] *= ti.exp(-dh*5)

    # enforce boundary condition
    for i in range(N):
        if picking[None]:           
            r = x[i]-curser[None]
            if r.norm() < curser_radius:
                x[i] = curser[None]
                v[i] = ti.Vector([0.0, 0.0])
                pass

    for j in range(N_y):
        ind = ij_2_index(0, j)
        v[ind] = ti.Vector([0, 0])
        x[ind] = ti.Vector([init_x, init_y + j * dx])  # rest pose attached to the wall

    for i in range(N):
        if x[i][0] < init_x:
            x[i][0] = init_x
            v[i][0] = 0


@ti.kernel
def updateLameCoeff():
    E = YoungsModulus[None]
    nu = PoissonsRatio[None]
    LameLa[None] = E*nu / ((1+nu)*(1-2*nu))
    LameMu[None] = E / (2*(1+nu))