# blender_antipodal.py
import bpy, addon_utils, bmesh
import numpy as np
from mathutils import Vector, kdtree
from pathlib import Path
from mathutils.bvhtree import BVHTree

# ─── ПАРАМЕТРЫ ───────────────────────────────────────────────
SRC_PATH          = "/home/nikita/diplom/testblend/test.stl"
OUT_NPZ           = "/home/nikita/diplom/testblend/antipodal_pairs.npz"
SCALE             = 1.0
NUM_SAMPLES       = 2000
MAX_GRIP_WIDTH    = 1.0
MAX_GRIP_DEPTH    = 0.3
NORMAL_COS_THRESH = 0.5
RESET_SCENE       = True
FREE_CLEAR   = 0.05   # свободное пространство наружу 
SOLID_THICK  = 0.01   # толщина материала за лицом 
EPS = 1e-4

FREE_CLEAR_K  = 0.05   # 5% диагонали bbox
SOLID_THICK_K = 0.005  # 0.5% диагонали bbox
OCC_RAYS      = 8      # кол-во лучей в полушарии вокруг нормали
OCC_MAX_HIT   = 0.3    # доля пересечений допустимая


def build_bvh(obj):
    dg = bpy.context.evaluated_depsgraph_get()
    return BVHTree.FromObject(obj, dg)

def ray_free(p, d, bvh, max_dist):
    """True, если по направлению d от точки p нет пересечений до max_dist."""
    hit = bvh.ray_cast(Vector(p) + Vector(d)*1e-4, Vector(d), max_dist)
    return hit[0] is None

def ray_dist(bvh, orig, dir_vec, max_dist=1e6):
    hit = bvh.ray_cast(orig, dir_vec, max_dist)
    if hit[0] is None:
        return None
    return (hit[0] - orig).length

def hemi_dirs(n, k=8):
    # равномерно по полушарию вокруг n
    n = n / (np.linalg.norm(n) + 1e-9)
    # простая сетка: случайные направления + проекция на полушарие
    dirs = np.random.randn(k, 3)
    dirs -= (dirs @ n)[:, None] * n[None, :]  # ортогональ
    dirs = dirs / (np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-9)
    dirs = (dirs + n) / 2.0
    return dirs / (np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-9)

def occlusion_score(p, n, bvh, reach):
    dirs = hemi_dirs(n, OCC_RAYS)
    hits = 0
    for d in dirs:
        hit = bvh.ray_cast(Vector(p)+Vector(d)*1e-4, Vector(d), reach)
        if hit[0] is not None:
            hits += 1
    return hits / OCC_RAYS

def sample_mesh_filtered(obj, num_samples: int):
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bmesh.ops.triangulate(bm, faces=bm.faces)
    faces = list(bm.faces)

    M = obj.matrix_world.copy()
    R = M.to_3x3()

    # bbox-диагональ
    bb = [M @ Vector(c) for c in obj.bound_box]
    diag = max((bb[i] - bb[j]).length for i in range(8) for j in range(8))
    free_clear  = FREE_CLEAR_K  * diag
    solid_thick = SOLID_THICK_K * diag

    bvh = build_bvh(obj)

    outer = []
    for f in faces:
        c_w = M @ f.calc_center_median()
        n_w = (R @ f.normal).normalized()

        d_out = ray_dist(bvh, c_w + n_w*EPS,  n_w,  diag)
        d_in  = ray_dist(bvh, c_w - n_w*EPS, -n_w, diag)

        cond_out = (d_out is None) or (d_out > free_clear)
        cond_in  = (d_in is not None) and (d_in < solid_thick)

        if cond_out and cond_in:
            outer.append((f, n_w))

    if not outer:
        print("WARN outer=0, fallback all faces")
        outer = [(f, (R @ f.normal).normalized()) for f in faces]

    areas = np.array([f.calc_area() for f,_ in outer], dtype=np.float64)
    areas /= areas.sum()

    pts  = np.empty((num_samples, 3), dtype=np.float32)
    nrms = np.empty((num_samples, 3), dtype=np.float32)

    idxs = np.random.choice(len(outer), size=num_samples, p=areas)
    kept = []
    for i, fi in enumerate(idxs):
        f, n_w = outer[fi]
        u, v = np.random.rand(2)
        if u+v > 1: u, v = 1-u, 1-v
        w = 1 - u - v
        v0, v1, v2 = (vt.co for vt in f.verts)
        p_w = M @ (v0*u + v1*v + v2*w)
        p_np = np.array((p_w.x, p_w.y, p_w.z), np.float32)
        n_np = np.array((n_w.x, n_w.y, n_w.z), np.float32)

        # окклюзия
        occ = occlusion_score(p_np, n_np, bvh, free_clear*2)
        if occ > OCC_MAX_HIT:
            continue

        pts[len(kept)]  = p_np
        nrms[len(kept)] = n_np
        kept.append(i)

    pts  = pts[:len(kept)]
    nrms = nrms[:len(kept)]
    bm.free()
    return pts, nrms
# ─── BLENDER UTILS ───────────────────────────────────────────
def reset_and_enable_addons():
    if RESET_SCENE:
        bpy.ops.wm.read_factory_settings(use_empty=True)
    for mod in ("object_fracture_cell", "io_mesh_stl", "io_scene_obj", "io_mesh_ply"):
        try:
            addon_utils.enable(mod, default_set=True, persistent=True)
        except Exception as exc:
            print(f"[WARN] cannot enable addon {mod}: {exc}")

def import_mesh(src: Path, scale: float | None):
    ext = src.suffix.lower()
    before = {o.name for o in bpy.data.objects}
    if ext == ".stl":
        bpy.ops.import_mesh.stl(filepath=str(src), global_scale=scale or 1.0)
    elif ext == ".ply":
        bpy.ops.import_mesh.ply(filepath=str(src))
        if scale:
            bpy.context.view_layer.objects.active.scale = (scale,)*3
    else:
        bpy.ops.import_scene.obj(filepath=str(src))
        if scale:
            bpy.context.view_layer.objects.active.scale = (scale,)*3
    after = {o.name for o in bpy.data.objects}
    new_names = list(after - before)
    if not new_names:
        raise RuntimeError("Импорт не создал объектов")
    obj = bpy.data.objects[new_names[0]]
    bpy.context.view_layer.objects.active = obj
    try:
        for o in bpy.context.selected_objects:
            o.select_set(False)
    except AttributeError:
        pass
    obj.select_set(True)
    obj.name = src.stem
    return obj

# ─── АНТИПОДАЛЬНЫЕ ПАРЫ ──────────────────────────────────────
def line_free(p1, p2, bvh):
    p1_vec = Vector(p1)
    p2_vec = Vector(p2)
    direction = p2_vec - p1_vec
    distance = direction.length
    if distance < 1e-6:
        return False
    direction.normalize()
    # небольшое смещение, чтобы не попадать сразу в поверхность
    hit = bvh.ray_cast(p1_vec + direction * 1e-4, direction, distance - 2e-4)
    return hit[0] is None


def find_antipodal_pairs(points, normals, max_width, max_depth, cos_thr, bvh):
    N = len(points)
    tree = kdtree.KDTree(N)
    for i, p in enumerate(points):
        tree.insert(Vector(p), i)
    tree.balance()

    pairs_idx = []
    pairs_pts = []

    for i in range(N):
        p1 = points[i]; n1 = normals[i]
        for (_, j, dist) in tree.find_range(Vector(p1), max_width):
            if j <= i or dist < 0.005:
                continue

            p2 = points[j]; n2 = normals[j]
            gdir = (p2 - p1) / dist

            if np.dot(n1,  gdir) < cos_thr:  continue
            if np.dot(n2, -gdir) < cos_thr:  continue

            # важная проверка на коллизии (линия свободна!)
            if not line_free(p1, p2, bvh):
                continue

            pairs_idx.append((i, j))
            pairs_pts.append((p1, p2))

    return np.array(pairs_idx, dtype=np.int32), np.array(pairs_pts, dtype=np.float32)



# ── MAIN ────────────────────────────────────────
def main():
    reset_and_enable_addons()
    obj = import_mesh(Path(SRC_PATH), SCALE)

    pts, nrms = sample_mesh_filtered(obj, NUM_SAMPLES)

    # 1) BVH
    bvh = build_bvh(obj)

    # 2) Проверка ориентации нормалей (автофлип при необходимости)
    center = pts.mean(axis=0)
    dotc = np.einsum('ij,ij->i', nrms, pts - center)
    print("Средний dot нормалей с направлением наружу:", dotc)
    if (dotc > 0).sum() < (dotc <= 0).sum():
        nrms = -nrms
        dotc = -dotc

    # 3) Мягкие маски
    outward_mask = dotc > 0.0
    # “наружу” луч свободен хотя бы в одном из направлений ±n
    exposed_mask = np.array(
        [ray_free(p, n,  bvh, MAX_GRIP_WIDTH*2) or
         ray_free(p, -n, bvh, MAX_GRIP_WIDTH*2)
         for p, n in zip(pts, nrms)],
        dtype=bool
    )

    mask = outward_mask & exposed_mask
    # DEBUG
    print("pts total:", len(pts),
          "outward:", mask.sum(), "exposed:", exposed_mask.sum())

    pts_f  = pts[mask]
    nrms_f = nrms[mask]

    min_bounds = pts.min(axis=0)
    max_bounds = pts.max(axis=0)
    print("Размеры объекта:", max_bounds - min_bounds)

    avg_normal_dot = np.mean(np.einsum('ij,ij->i', nrms, (pts - pts.mean(axis=0))))
    print("Средний dot нормалей с направлением наружу:", avg_normal_dot)

    # временно без масок!
    idx_pairs, pt_pairs = find_antipodal_pairs(
    pts, nrms,
    MAX_GRIP_WIDTH,
    MAX_GRIP_DEPTH,
    NORMAL_COS_THRESH,
    bvh
        )
    print("Найдено пар (без масок):", len(idx_pairs))
    print("Найдено пар:", len(idx_pairs))

    np.savez(OUT_NPZ,
         pts_all=pts,
         nrms_all=nrms,
         mask_outward=outward_mask,
         mask_exposed=exposed_mask,
         keep_mask=mask,          # если используешь
         pts_keep=pts_f,
         nrms_keep=nrms_f)
    print("Saved:", OUT_NPZ)

if __name__ == "__main__":
    main()
