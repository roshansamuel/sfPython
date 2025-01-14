import sys
import h5py
import time
import numpy as np
from scipy import interpolate

device = "cpu"

import numba as nb  #comment this line if device is not cpu

if device == "gpu":
   import cupy as cp

print()

# If computeVSF is true, code will compute *only* velocity SF
# If set to false, code will compute *only* scalar SF
# Both are not computed together
computeVSF = False

# Name of scalar whose SF must be computed if computeVSF is False
sfScalar = 'T'

# Default times
startTime = 0.0
stopTime = float('Inf')

argList = sys.argv[1:]
if argList:
    # If only two arguments are there, it is start and end times
    if len(argList) == 2:
        startTime = float(argList[0])
        stopTime = float(argList[1])
        computeVSF = True
    elif len(argList) == 3:
        startTime = float(argList[0])
        stopTime = float(argList[1])
        computeVSF = False
        sfScalar = argList[2]
    else:
        print("Argument error")
        exit()
else:
    print("Argument error")
    exit()


# read the data file ###############
def hdf5_reader(filename,dataset):
    file_read = h5py.File(filename, 'r')
    dataset_read = file_read["/"+dataset]
    V = dataset_read[...]

    return V

if device == "gpu":
    # select GPU device
    dev1 = cp.cuda.Device(1)

    dev1.use()

############ Calculate domain params ####

dataDir = "../data/2d_data/1_1e9_pySaras/"

Lx, Lz = 2.0, 1.0
Nx, Nz = 1024, 512
lStr, lEnd = 0.4, 2.2

dx = Lx/Nx
dz = Lz/Nz

nx = Nx
nz = Nz

X = np.linspace(0.0, Lx, Nx, endpoint=False)
Z = np.linspace(0.0, Lz, Nz, endpoint=False)

pSkip = 4096

#############################

# define cpu arrays
S_array_cpu = np.zeros([nx, nz])
S_u_r_array_cpu = np.zeros([nx, nz])

# define gpu arrays
if device == "gpu":
    S_array = cp.asarray(S_array_cpu)
    S_u_r_array = cp.asarray(S_u_r_array_cpu)


#############################
def interpolateData(f, xO, zO, xI, zI):
    intFunct = interpolate.interp1d(zI, f, kind='cubic', axis=1)
    f = intFunct(zO)
    intFunct = interpolate.interp1d(xI, f, kind='cubic', axis=0)
    f = intFunct(xO)

    return f


def periodicBC(f):
    f[0,:], f[-1,:] = f[-2,:], f[1,:]


# WARNING: parallel=True may have to be disabled below on some systems
@nb.jit(nopython=True, parallel=True)
def vel_str_function_cpu(Vx, Vz, Ix, Iz, l_cap_x, l_cap_z, S_array_cpu, S_u_r_array_cpu):
    N = len(l_cap_x)

    for m in range(N):
        u1, u2 = Vx[0:Nx-Ix[m], 0:Nz-Iz[m]], Vx[Ix[m]:Nx, Iz[m]:Nz]
        w1, w2 = Vz[0:Nx-Ix[m], 0:Nz-Iz[m]], Vz[Ix[m]:Nx, Iz[m]:Nz]

        del_u, del_w = u2[:, :] - u1[:, :], w2[:, :] - w1[:, :]
        diff_magnitude_sqr = (del_u)**2 + (del_w)**2

        S_array_cpu[Ix[m], Iz[m]] = np.mean(diff_magnitude_sqr[:, :])
        S_u_r_array_cpu[Ix[m], Iz[m]] = np.mean((del_u[:, :]*l_cap_x[m] + del_w[:, :]*l_cap_z[m])**2)

        if (not m%pSkip): print(m, N, Ix[m]*dx, Iz[m]*dz)

    return

# WARNING: parallel=True may have to be disabled below on some systems
@nb.jit(nopython=True, parallel=True)
def tmp_str_function_cpu(Vx, Vz, T, Ix, Iz, l_cap_x, l_cap_z, S_array_cpu):
    N = len(l_cap_x)

    for m in range(N):
        t1, t2 = T[0:Nx-Ix[m], 0:Nz-Iz[m]], T[Ix[m]:Nx, Iz[m]:Nz]

        del_t = t2[:, :] - t1[:, :]
        diff_temp_sqr = (del_t)**2

        S_array_cpu[Ix[m], Iz[m]] = np.mean(diff_temp_sqr[:, :])

        if (not m%pSkip): print(m, N, Ix[m]*dx, Iz[m]*dz)

    return


def vel_str_function_gpu(Vx, Vz, Ix, Iz, l_cap_x, l_cap_z, S_array, S_u_r_array):
    N = len(l_cap_x)

    # copy data from cpu to gpu
    Vx, Vz = cp.asarray(Vx), cp.asarray(Vz)

    cp._core.set_routine_accelerators(['cub', 'cutensor'])

    for m in range(N):
        u1, u2 = Vx[0:Nx-Ix[m], 0:Nz-Iz[m]], Vx[Ix[m]:Nx, Iz[m]:Nz]
        w1, w2 = Vz[0:Nx-Ix[m], 0:Nz-Iz[m]], Vz[Ix[m]:Nx, Iz[m]:Nz]

        del_u, del_w = u2[:, :] - u1[:, :], w2[:, :] - w1[:, :]
        diff_magnitude_sqr = (del_u)**2 + (del_w)**2

        S_array[Ix[m], Iz[m]] = cp.mean(diff_magnitude_sqr[:, :])
        S_u_r_array[Ix[m], Iz[m]] = cp.mean((del_u[:, :]*l_cap_x[m] + del_w[:, :]*l_cap_z[m])**2)

        if (not m%pSkip): print(m, N, Ix[m]*dx, Iz[m]*dz)

    return


def tmp_str_function_gpu(Vx, Vz, T, Ix, Iz, l_cap_x, l_cap_z, S_array):
    N = len(l_cap_x)

    # copy data from cpu to gpu
    T = cp.asarray(T)

    cp._core.set_routine_accelerators(['cub', 'cutensor'])

    for m in range(N):
        t1, t2 = T[0:Nx-Ix[m], 0:Nz-Iz[m]], T[Ix[m]:Nx, Iz[m]:Nz]

        del_t = t2[:, :] - t1[:, :]
        diff_temp_sqr = (del_t)**2

        S_array[Ix[m], Iz[m]] = cp.mean(diff_temp_sqr[:, :])

        if (not m%pSkip): print(m, N, Ix[m]*dx, Iz[m]*dz)

    return


## pre-process
ix = np.arange(0, nx)
iz = np.arange(0, nz)

imgx, imgz = np.meshgrid(ix, iz, indexing='ij')
lTemp = np.sqrt((imgx*dx)**2 + (imgz*dz)**2)

index = np.where((lTemp >= lStr) & (lTemp <= lEnd))

Ix = imgx[index].flatten()
Iz = imgz[index].flatten()
l = lTemp[index].flatten()

l_cap_x = np.zeros_like(l)
l_cap_z = np.zeros_like(l)
l_cap_x[1:], l_cap_z[1:] = ((dx*Ix[1:])/l[1:]), ((dx*Iz[1:])/l[1:])


tList = np.loadtxt(dataDir + "timeList.dat", comments='#')

for i in range(tList.shape[0]):
    tVal = tList[i]
    if tVal > startTime and tVal < stopTime:
        fileName = dataDir + "Soln_{0:09.4f}.h5".format(tVal)
        print("\nReading from file ", fileName)
        Vx = np.pad(hdf5_reader(fileName, "Vx"), 1)
        Vz = np.pad(hdf5_reader(fileName, "Vz"), 1)
        if not computeVSF:
            T = np.pad(hdf5_reader(fileName, sfScalar), 1)

        xI = np.pad(hdf5_reader(fileName, "X"), (1, 1), 'constant', constant_values=(0, Lx))
        zI = np.pad(hdf5_reader(fileName, "Z"), (1, 1), 'constant', constant_values=(0, Lz))

        # Periodic BC
        xI[0], xI[-1] = -xI[1], Lx + xI[1]
        periodicBC(Vx)
        periodicBC(Vz)
        if not computeVSF:
            periodicBC(T)

        print("\tInterpolating data")
        Vx = interpolateData(Vx, X, Z, xI, zI)
        Vz = interpolateData(Vz, X, Z, xI, zI)
        if not computeVSF:
            T = interpolateData(T, X, Z, xI, zI)

        print("\tComputing Structure Function")
        ## compute str_function
        t_str_func_start = time.time()
        print()

        if device == "gpu":
            if computeVSF:
                vel_str_function_gpu(Vx, Vz, Ix, Iz, l_cap_x, l_cap_z, S_array, S_u_r_array)
            else:
                tmp_str_function_gpu(Vx, Vz, T, Ix, Iz, l_cap_x, l_cap_z, S_array)
        else:
            if computeVSF:
                vel_str_function_cpu(Vx, Vz, Ix, Iz, l_cap_x, l_cap_z, S_array_cpu, S_u_r_array_cpu)
            else:
                tmp_str_function_cpu(Vx, Vz, T, Ix, Iz, l_cap_x, l_cap_z, S_array_cpu)

        t_str_func_end = time.time()
        print("str func compute time = ", t_str_func_end-t_str_func_start)

        if device == "gpu":
            S_array_cpu = cp.asnumpy(S_array)
            if computeVSF:
                S_u_r_array_cpu = cp.asnumpy(S_u_r_array)

        ## save file
        if computeVSF:
            fileName = dataDir + "V_SF2_{0:09.4f}.h5".format(tVal)
        else:
            fileName = dataDir + sfScalar + "_SF2_{0:09.4f}.h5".format(tVal)

        hf = h5py.File(fileName, 'w')
        hf.create_dataset("S", data=S_array_cpu)
        if computeVSF:
            hf.create_dataset("S_u_r", data=S_u_r_array_cpu)
        hf.close()

