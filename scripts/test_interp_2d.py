import pytools as pt
import numpy as np
import time
import sys
try:
   from collections.abc import Iterable
except ImportError:
   from collections import Iterable
import matplotlib.pyplot as plt
import matplotlib as mpl
import cProfile
from pstats import SortKey

rng = np.random.default_rng(12345)

t0 = time.time()
N = 1000#int(np.sqrt(800))
# coords = (rng.random((N, 3))-0.5) *2e8
# coords = np.linspace([10*6371e3,-1e6,-1e6],[2e8,-1e6,-1e6],N)
# coords = np.linspace([47140909.09090909,-1e6,-1e6],[52236212.12121212,-1e6,-1e6],100)
delta = 60e6
xmin = 45.0e6
xcoords = np.linspace(xmin,xmin+delta,N)
dx = xcoords[1]-xcoords[0]
ymin = -37.51e6 - 1e7*0
ycoords = np.linspace(ymin,delta+ymin,N)
dy = ycoords[1]-ycoords[0]
# print(max(xcoords)/6371e3,max(ycoords)/6371e3)

# print(coords)
N = ycoords*xcoords
X,Y,Z = np.meshgrid(xcoords, ycoords,np.array([-0.25e6]))




# fn = '/home/mjalho/Documents/XO-ms/EGI/sus/EGI_lmn_0001266.vlsv'
# f = pt.vlsvfile.VlsvReader(fn)

# w = pt.vlsvfile.VlsvWriter(f, "./EGI_interpolationtest.vlsv", copy_meshes=['SpatialGrid'])
# w.copy_variables_list(f,['vg_b_vol','CellID'])
# crds = f.read_variable('vg_coordinates_cell_center')
# r2 = np.sum(crds**2,axis=1)
# r = np.sqrt(r2)
# w.write(data=r2,name='r2',tag='VARIABLE', mesh='SpatialGrid')
# w.write(data=r,name='r',tag='VARIABLE', mesh='SpatialGrid')

name = "r2"
operator = "pass"
f=pt.vlsvfile.VlsvReader("/wrk-vakka/group/spacephysics/vlasiator/3D/miscellanous/EGI_interpolationtest.vlsv")

f.read_variable_to_cache('vg_b_vol')
f.read_variable_to_cache('r2')
asd =f.read_variable('vg_b_vol', cellids=[])
print(asd)

whole_variable = f.read_variable(name,cellids=[1,2],operator=operator)
if isinstance(whole_variable[0], Iterable):
   value_length=len(whole_variable[0])
else:
   value_length=1

ncoords = np.prod(X.shape)

coords = np.hstack((np.reshape(X,(ncoords))[:,np.newaxis],
                    np.reshape(Y,(ncoords))[:,np.newaxis],
                    np.reshape(Z,(ncoords))[:,np.newaxis]))

baseline = np.sum(coords**2,axis=1)
baseline = np.reshape(baseline,X.shape)

cids = f.get_cellid(coords)

cids = np.array(list(set(cids)),dtype=np.int64)#np.array(f.get_unique_cellids(coords),dtype=np.int64)
refs = f.get_amr_level(cids)
# cids = np.array(list(set(cids)),dtype=np.int64)
# print(cids)
cid_coords = f.read_variable("vg_coordinates",cids)
# print(coords.shape, cids.shape)

cid_coords_ll = f.read_variable("vg_coordinates_cell_lowcorner",cids)
cid_coords_lh = cid_coords_ll + f.read_variable("vg_dxs",cids)*np.array([0,1,0])[np.newaxis,:]
cid_coords_hl = cid_coords_ll + f.read_variable("vg_dxs",cids)*np.array([1,0,0])[np.newaxis,:]
cid_coords_hh = cid_coords_ll + f.read_variable("vg_dxs",cids)*np.array([1,1,0])[np.newaxis,:]

cid_coords_corners = np.vstack((cid_coords_ll,cid_coords_hl,cid_coords_lh,cid_coords_hh))
# cProfile.run('f.read_variable_to_cache("vg_regular_interp_neighbors")',sort=SortKey.CUMULATIVE)
# f.read_variable_to_cache("vg_regular_interp_neighbors")
print(time.time()-t0,"seconds for test init")
# sys.exit()
# f.read_variable_to_cache('vg_coordinates')
# print(f.get_amr_level(cids))
t0 = time.time()
# res2 = f.read_interpolated_variable_irregular(name,coords[0,:],operator=operator)
# cProfile.run("f.read_interpolated_variable(name,f.read_variable('vg_coordinates'))", sort=SortKey.CUMULATIVE)
# print("")
# res2 = f.read_interpolated_variable_irregular(name,np.array([7.50000000e+07,  2.86211111e+07, -2.50000000e+05]),operator=operator)
# cProfile.run("res2 = f.read_interpolated_variable_irregular(name,np.array([7.50000000e+07,  2.86211111e+07, -2.50000000e+05]),operator=operator)")
# print(res2)
print(time.time()-t0,"seconds for jit (if enabled)")
# sys.exit()
# print("Redoing the one point")
# res2 = f.read_interpolated_variable_irregular(name,coords[0,:],operator=operator)
# print(res2)
t0 = time.time()
# try:
# print("doing the grid")
# res2 = f.read_interpolated_variable(name,coords,operator=operator)
# print("doing that one point")
# res2 = f.read_interpolated_variable_irregular(name,np.array([7.50000000e+07,  2.86211111e+07, -2.50000000e+05]),operator=operator)
cProfile.run("res2 = f.read_interpolated_variable(name,coords,operator=operator)",sort=SortKey.CUMULATIVE)
cProfile.run("res2 = f.read_interpolated_variable(name,coords,operator=operator)",sort=SortKey.CUMULATIVE)

# except Exception as e:
#    print(e)
#    res2 = np.zeros_like(coords[:,0])   
print(time.time()-t0,"seconds for batch interp")
res = np.reshape(res2[:],X.shape)
# cProfile.run("res2 = f.read_interpolated_variable(name,coords,operator=operator)")

# t0 = time.time()
# # try:
# res2 = f.read_interpolated_variable(name,coords,operator=operator)
# # except Exception as e:
# #    print(e)
# #    res2 = np.zeros_like(coords[:,0])   
# print(time.time()-t0,"seconds for a second pass of batch interp (duals cached?)")
# res = np.reshape(res2,X.shape)

import matplotlib as mpl
plt.figure(figsize=[20,20])
font = {'family' : 'normal',
        'size'   : 34}
plt.rc('font', **font)
plt.imshow(np.abs(res[:,:,0]-baseline[:,:,0])/baseline[:,:,0],
           origin="lower", 
           extent=(min(xcoords)-0.5*dx,max(xcoords)+0.5*dx,min(ycoords)-0.5*dy,max(ycoords)+0.5*dy),
           norm=mpl.colors.LogNorm(vmin=1e-5,vmax=1e0), cmap="glasgow")
plt.colorbar()
#plt.imshow(np.log10(res_ids[:,:,0]),origin="lower", extent=(min(xcoords),max(xcoords),min(ycoords),max(ycoords)))

# plt.contour(X[:,:,0],Y[:,:,0],refs[:,:,0])
plt.scatter(cid_coords[:,0],cid_coords[:,1],label="vg_centers",c='gray')
plt.scatter(cid_coords_corners[:,0],cid_coords_corners[:,1],label="vg_corners", marker="+")
#plt.scatter(ptsdel[:,0],ptsdel[:,1],marker="x",label="intp_pts")
plt.legend()
plt.title("abs. relative diff $|r_\mathrm{intp}^2 - r^2|*r^{-2}$")

plt.savefig("try_2db.png", dpi=300)

plt.figure(figsize=[20,20])
font = {'family' : 'normal',
        'size'   : 34}
plt.rc('font', **font)

#plt.imshow(np.log10(res_ids[:,:,0]),origin="lower", extent=(min(xcoords),max(xcoords),min(ycoords),max(ycoords)))

# plt.contour(X[:,:,0],Y[:,:,0],refs[:,:,0])

cid_num_verts = [len(f._VlsvReader__cell_vertices.get(c,[])) for c in cids]
plt.scatter(cid_coords[:,0],cid_coords[:,1],label="vg_centers",c = cid_num_verts)
plt.scatter(cid_coords_corners[:,0],cid_coords_corners[:,1],label="vg_corners", marker="+")
plt.colorbar()
#plt.scatter(ptsdel[:,0],ptsdel[:,1],marker="x",label="intp_pts")
plt.legend()
plt.title("abs. relative diff $|r_\mathrm{intp}^2 - r^2|*r^{-2}$")

plt.savefig("try_2db_nverts.png", dpi=300)
sys.exit()
t0 = time.time()
res2 = f.read_interpolated_variable_irregular(name,coords,operator=operator,method="RBF")
print(time.time()-t0,"seconds for irregular interp (RBF)")

# res = np.array([f.read_interpolated_variable_irregular(name,c,operator=operator) for c in coords])

res = np.reshape(res2,X.shape)
refs = np.reshape(refs,X.shape)
res_ids = np.reshape(f.get_cellid(coords),X.shape)
#ptsdel = f._VlsvReader__irregular_cells_delaunay.get_points()

plt.figure(figsize=[20,20])
font = {'family' : 'normal',
        'size'   : 34}
plt.rc('font', **font)
#plt.pcolor(X[:,:,0],Y[:,:,0],res[:,:,0])

plt.imshow(np.abs(res[:,:,0]-baseline[:,:,0])/baseline[:,:,0],
           origin="lower", 
           extent=(min(xcoords),max(xcoords),min(ycoords),max(ycoords)),
           norm=mpl.colors.LogNorm(vmin=1e-6,vmax=1e-3),cmap="lipari")
plt.colorbar()
#plt.imshow(np.log10(res_ids[:,:,0]),origin="lower", extent=(min(xcoords),max(xcoords),min(ycoords),max(ycoords)))

# plt.contour(X[:,:,0],Y[:,:,0],refs[:,:,0])
plt.scatter(cid_coords[:,0],cid_coords[:,1],label="vg_centers",c='gray')
plt.scatter(cid_coords_corners[:,0],cid_coords_corners[:,1],label="vg_corners", marker="+")
#plt.scatter(ptsdel[:,0],ptsdel[:,1],marker="x",label="intp_pts")
plt.legend()
plt.title("abs. relative diff $|r_\mathrm{intp}^2 - r^2|*r^{-2}$")

plt.savefig("try_2db_cleaned.png", dpi=300)


sys.exit()
res = np.zeros((N,value_length))
t0 = time.time()
cids = [f.get_cellid(c) for c in coords]
print(time.time()-t0,"seconds for list comprehension cids")
# print("list comp cids ",cids)
t0 = time.time()
cids = f.get_cellid(coords)
print(time.time()-t0,"seconds for batch cids")
# print("batch cids ",cids)
#sys.exit()
t0 = time.time()
res_new = f.read_interpolated_variable(name,coords,operator=operator)
print(time.time()-t0,"seconds for batch interp")
print(res_new.shape)

t0 = time.time()
if isinstance(whole_variable[0], Iterable) and N > 1:
   for i,coord in enumerate(coords):
      res[i,:] = f.read_interpolated_variable(name,coord,operator=operator)
elif N > 1:
   for i,coord in enumerate(coords):
      res[i] = f.read_interpolated_variable(name,coord,operator=operator)
else:
   res = f.read_interpolated_variable(name,coords,operator=operator)
print(time.time()-t0,"seconds for pointwise interp")
res=res.squeeze()

if True:
   t0 = time.time()
   res_v = np.zeros((N,value_length))
   if isinstance(whole_variable[0], Iterable) and N > 1:
      for i,coord in enumerate(coords):
         res_v[i,:] = f.read_interpolated_variable_irregular(name,coord,operator=operator)
   elif N > 1:
      for i,coord in enumerate(coords):
         res_v[i] = f.read_interpolated_variable_irregular(name,coord,operator=operator)
   else:
      res_v = f.read_interpolated_variable_irregular(name,coords,operator=operator)
   res_v=res_v.squeeze()
   print(time.time()-t0,"seconds for vertex interp")

if True:
   t0 = time.time()
   res_new2 = f.read_interpolated_variable_irregular(name,coords,operator=operator)
   print(time.time()-t0,"seconds for batch vertex interp")

# print(res)
# print(res_new)

#print(res,res_new, res_new-res)

print("Column-wise sum of absolute differences", np.sum(np.abs((res_v-res_new2)), axis = 0))
# print("Column-wise sum of absolute differences", np.sum(np.abs((res_v-res)), axis = 0))

plt.step(coords[:,0]/1e6,
         f.read_variable(name, f.get_cellid(coords), operator=operator)*1e9,
         where="mid",
         label="Bx nearest neighour")

# plt.plot(coords[:,0]/1e6,res*1e9, label="Bx regular intepr")
plt.plot(coords[:,0]/1e6,res_v*1e9, label="Bx vertex\&delaunay")
plt.plot(coords[:,0]/1e6,res_new2*1e9, label="Bx batch+")
plt.plot(coords[:,0]/1e6,(res_new2-res_v)*1e9, label="diff")
plt.plot(coords[:,0]/1e6,f.get_amr_level(cids), label="reflevel")
#plt.yscale("symlog", linthresh=1e-8)
plt.ylabel("nT or reflevel")
plt.xlabel("x/Mm")
plt.legend()
plt.grid()
plt.savefig("try.png")
