#!/usr/bin/env python3
# ---------------------------------------------------------------------------
#  single_grasp_manual.py            2025‑07‑30
#  ‣ один захват по двум точкам       (pinch / antipodal)
#  ‣ с дополнительным запасом по ширине и по подъёму
# ---------------------------------------------------------------------------
import sys, numpy as np, open3d as o3d
from graspnetAPI import GraspGroup

# ──────────── файлы и исходные данные ───────────────────────────────────────
NPZ_PATH, MESH_PATH = "antipodal_pairs.npz", "test.stl"
IDX_P1, IDX_P2      = 395, 1483               # ← ваши точки

# ──────────── параметрические ручки ─────────────────────────────────────────
FINGER_H   = 0.050     # толщина губок             H,  м
DEPTH      = 0.100     # глубина захода            D,  м
WIDTH_PAD  = 0.010     # с каждой стороны!         Δw, м
EXTRA_SHIFT= 0.60      # доля D дополн. подъёма    0…1
UP         = np.array([0,0,1], np.float32)
# ---------------------------------------------------------------------------


def build_row(p1, p2, n1, n2):
    """Формируем 17‑float строку GraspNet."""
    base_w = np.linalg.norm(p2 - p1)
    width  = base_w + 2*WIDTH_PAD            # • с запасом

    jaw    = (p2 - p1) / (base_w + 1e-9)     # Y‑ось (между губками)
    appr   = -(n1 + n2);  appr /= np.linalg.norm(appr)+1e-9   # X‑ось
    binm   = np.cross(appr, jaw); binm/=np.linalg.norm(binm)+1e-9 # Z‑ось

    # центр: половина глубины + EXTRA_SHIFT·D
    shift  = appr * (DEPTH*0.5 + EXTRA_SHIFT*DEPTH)
    center = (p1 + p2)/2 - shift

    R = np.stack([appr, jaw, binm], 1).astype(np.float32)

    row                 = np.zeros(17, np.float32)
    row[:4]             = [1.0, width, FINGER_H, DEPTH]
    row[4:13]           = R.reshape(-1)
    row[13:16]          = center
    row[16]             = -1
    return row, center, R, width


def main():
    data = np.load(NPZ_PATH)
    pts, nrms = data["points"], data["normals"]

    for idx in (IDX_P1, IDX_P2):
        if not 0 <= idx < len(pts):
            sys.exit(f"[ERR] индекс {idx} вне диапазона точек")

    p1, p2 = pts[IDX_P1], pts[IDX_P2]
    n1, n2 = nrms[IDX_P1], nrms[IDX_P2]

    row, center, R, width = build_row(p1, p2, n1, n2)
    gg = GraspGroup(row[None])

    # ---- консоль -----------------------------------------------------------
    print("\n=== SINGLE GRASP (with pads) ===")
    print(f"indices        : {IDX_P1}, {IDX_P2}")
    print(f"p1, p2 (m)     : {p1},\n                 {p2}")
    print(f"base width (m) : {np.linalg.norm(p2-p1):.4f}")
    print(f"WIDTH_PAD      : {WIDTH_PAD:.3f}  → final width = {width:.4f} m")
    print(f"DEPTH, EXTRA   : {DEPTH:.3f}, shift +{EXTRA_SHIFT*DEPTH:.3f} m")
    print(f"center xyz     : {center}")
    print("R (A,J,B cols) :\n", R, "\n")

    # ---- визуализация ------------------------------------------------------
    mesh = o3d.io.read_triangle_mesh(MESH_PATH); mesh.compute_vertex_normals()
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    cloud.paint_uniform_color([.6,.6,.6])

    pin_r = 0.012 * (pts.max(0) - pts.min(0)).max()
    pins  = []
    for p in (p1, p2):
        s = o3d.geometry.TriangleMesh.create_sphere(pin_r); s.translate(p)
        s.paint_uniform_color([1,0,0]); pins.append(s)

    geoms = [mesh, cloud] + pins + gg.to_open3d_geometry_list()
    o3d.visualization.draw_geometries(geoms,
        window_name="Single grasp (+ width / depth pads)")

if __name__ == "__main__":
    main()
