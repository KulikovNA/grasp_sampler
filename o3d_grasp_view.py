#!/usr/bin/env python3
# ---------------------------------------------------------
#   grasp_sampler · pinch · DEBUG (2025-07-31)
# ---------------------------------------------------------
import numpy as np, open3d as o3d, numpy.linalg as npl
from scipy.spatial import cKDTree
from graspnetAPI import GraspGroup
from collections import Counter, defaultdict
# ─────────── FILES ───────────
NPZ_PATH, MESH_PATH = "antipodal_pairs.npz", "test.stl"
VIS_TOP_N = 200
# ─────────── PARAMS ───────────
W_MAX, MIN_DIST = 0.7, 0.40          # м
FINGER_H, D_MAX = 0.10, 0.20         # м
WIDTH_PAD, CENTER_PAD = 0.10, 0.10   # м
MU               = 2.0              # friction
CLEAR_FREE       = 2.0               # м
HPR_VIEWS, HPR_R = 120, 0.30
UP = np.array([0,0,1], np.float32)
# ─── quick ON/OFF of filters ───
CHECK_CONE   = True
CHECK_OUTER  = True
CHECK_FINGER = False
# ---------------------------------------------------------

def fib_dirs(n:int):
    i = np.arange(n, dtype=np.float32); φ = (1+5**0.5)/2
    z = 1-2*i/(n-1); r = np.sqrt(np.maximum(0,1-z*z))
    th = 2*np.pi*i/φ
    return np.stack([r*np.cos(th), r*np.sin(th), z],1)

def hpr_outer(pts, nrms, mesh):
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    aabb= mesh.get_axis_aligned_bounding_box()
    c   = aabb.get_center(); R = npl.norm(aabb.get_extent())*HPR_R
    keep= np.zeros(len(pts),bool)
    for d in fib_dirs(HPR_VIEWS):
        cam=c+d*R; _,idx=pcd.hidden_point_removal(cam, R*3); keep[idx]=True
    return pts[keep], nrms[keep]

def build_scene(mesh):
    sc = o3d.t.geometry.RaycastingScene()
    sc.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh)); return sc

def ray_dist(sc,o,v,tmax):
    o,v = np.asarray(o,np.float32), np.asarray(v,np.float32)
    v /= npl.norm(v)+1e-9
    if hasattr(sc,"cast_ray"):
        t = sc.cast_ray(o3d.core.Tensor(o), o3d.core.Tensor(v), t_max=tmax)['t_hit'].item()
        return t if t<np.inf else None
    ray = np.hstack([o,v])[None,:].astype(np.float32)
    t   = sc.cast_rays(o3d.core.Tensor(ray))['t_hit'].numpy()[0]
    return t if t<=tmax else None

def in_cone(n,v):
    return np.dot(n,-v)/(npl.norm(n)*npl.norm(v)) >= 1/np.sqrt(1+MU**2)

# ---------- build pairs ----------
def build_pairs(pts,nrms,sc):
    kd = cKDTree(pts)
    idx_pairs,pt_pairs=[],[]
    drop,sample = Counter(),defaultdict(list)

    for i,(p1,n1) in enumerate(zip(pts,nrms)):
        for j in kd.query_ball_point(p1, r=W_MAX):
            if j<=i: continue
            p2,n2 = pts[j], nrms[j]
            dvec  = p2-p1; dist = npl.norm(dvec)
            if dist<MIN_DIST: drop['too_narrow']+=1; continue

            jaw = dvec/dist            # направление между точками

            # ------ (1) friction cone ------
            if CHECK_CONE and not(in_cone(n1,jaw) and in_cone(n2,-jaw)):
                if len(sample['cone'])<30:       # лог первых 30 углов
                    a1 = np.degrees(np.arccos(
                        np.clip(np.dot(n1, jaw)/ (npl.norm(n1)+1e-9),-1,1)))
                    a2 = np.degrees(np.arccos(
                        np.clip(np.dot(n2,-jaw)/ (npl.norm(n2)+1e-9),-1,1)))
                    sample['cone'].append(f"{i:4},{j:4}  θ1={a1:5.1f}° θ2={a2:5.1f}°")
                drop['force_closure']+=1; continue

            # half-width depth
            if max(np.dot(n1,jaw),np.dot(n2,-jaw))*dist*0.5 > D_MAX:
                drop['half_depth']+=1; continue

            # внутренняя стенка
            if ray_dist(sc, p1-n1*1e-4, -n1, D_MAX) is None and \
               ray_dist(sc, p2-n2*1e-4, -n2, D_MAX) is None:
                drop['no_inner']+=1; continue

            approach = -(n1+n2)
            if npl.norm(approach)<1e-6: continue
            approach/=npl.norm(approach)

            # ------ (2) outer free ------
            if CHECK_OUTER and ray_dist(sc,(p1+p2)/2+approach*1e-4,
                                        approach, CLEAR_FREE) is not None:
                drop['outer_blocked']+=1; continue

            # ------ (3) finger collision ------
            binorm = np.cross(approach, jaw)
            if npl.norm(binorm)<1e-6: binorm=np.cross(jaw,UP)
            binorm/=npl.norm(binorm)
            L = dist-1e-4
            if CHECK_FINGER and any(ray_dist(sc, p1+binorm*o+jaw*1e-4,jaw,L) is not None
                                    for o in (0, FINGER_H/2,-FINGER_H/2)):
                drop['finger_col']+=1; continue

            idx_pairs.append((i,j)); pt_pairs.append((p1,p2))

    return np.asarray(idx_pairs,int), np.asarray(pt_pairs,float), drop, sample

# ---------- GraspGroup ----------
def to_gg(pt_pairs,nrms,idx_pairs):
    G=np.zeros((len(idx_pairs),17),np.float32)
    for k,(i,j) in enumerate(idx_pairs):
        p1,p2 = pt_pairs[k]; d=p2-p1; w0=npl.norm(d)
        jaw   = d/(w0+1e-9)
        approach = -(nrms[i]+nrms[j]); approach/=npl.norm(approach)+1e-9
        binorm=np.cross(approach,jaw); binorm/=npl.norm(binorm)+1e-9
        center=(p1+p2)/2 - approach*(D_MAX*0.5 + CENTER_PAD)
        R=np.stack([approach, jaw, binorm],1)
        G[k,:4]=[1,w0+2*WIDTH_PAD,FINGER_H,D_MAX]
        G[k,4:13]=R.reshape(-1); G[k,13:16]=center; G[k,16]=-1
    return GraspGroup(G)

# ---------- MAIN ----------
def main():
    d=np.load(NPZ_PATH); pts,nrms=d['points'],d['normals']
    mesh=o3d.io.read_triangle_mesh(MESH_PATH); mesh.compute_vertex_normals()
    scene=build_scene(mesh)

    pts_o,nrms_o = hpr_outer(pts,nrms,mesh)
    idx,pp,drop,sample = build_pairs(pts_o,nrms_o,scene)
    print(f"outer {len(pts_o)}/{len(pts)}   pairs {len(idx)}   drop {dict(drop)}")

    if sample['cone']:
        print("\n--- first bad cone pairs (θ1 / θ2) ---")
        print(*sample['cone'][:10], sep='\n')

    gg_raw   = to_gg(pp,nrms_o,idx)
    gg_filt  = gg_raw.nms(0.3, np.deg2rad(90))
    print("after NMS :", len(gg_filt))

    if len(gg_filt)==0:
        print("[!] NO grasps survived – try:  MU↓  |  WIDTH_PAD↑  |  turn off filters")
        return

    geoms=[mesh]
    pc=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts_o))
    pc.paint_uniform_color([.45,.45,.45]); geoms.append(pc)
    geoms += gg_filt[:VIS_TOP_N].to_open3d_geometry_list()
    o3d.visualization.draw_geometries(geoms, window_name="pinch – debug")

if __name__=="__main__":
    main()
