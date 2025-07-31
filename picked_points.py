#!/usr/bin/env python3
# -------------------------------------------------------------------------
#  pick_points.py                 минимальный “пинкер” для Open3D
#  • грузит cloud из NPZ‑файла (ключи: points | pts_keep | pts_all)
#  • Shift+ЛКМ – добавить / убрать точку,  q – выйти
#  • после закрытия печатает индексы и XYZ,
#    сохраняет всё в picked_points.json
# -------------------------------------------------------------------------
import json, sys, pathlib
import numpy as np
import open3d as o3d

NPZ_PATH  = "antipodal_pairs.npz"          # где лежат точки
OUT_JSON  = "picked_points.json"           # сюда запишутся результаты

def load_cloud(npz_path: str) -> np.ndarray:
    """Берём первое подходящее поле с точками."""
    data = np.load(npz_path)
    for k in ("pts_keep", "points", "pts_all"):
        if k in data:
            return data[k]
    sys.exit(f"[ERR] в {npz_path} нет массива точек (keys: {data.files})")

def main() -> None:
    pts = load_cloud(NPZ_PATH)
    print(f"[INFO] cloud loaded: {len(pts)} points")

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(pts)
    cloud.paint_uniform_color([0.6, 0.6, 0.6])

    print("Shift + ЛКМ – добавить/убрать точку,  q – выйти")
    o3d.visualization.draw_geometries_with_editing([cloud])

    # ----------------------------------------------------------
    #  Open3D пишет выбор в picked_points.json (если что‑то есть)
    # ----------------------------------------------------------
    try:
        sel = json.load(open(OUT_JSON))["selected"]
    except Exception:
        print("\n[INFO] файл picked_points.json не создан – точки не выбраны.")
        return

    if not sel:
        print("\n[INFO] список selected пуст – ничего не выбрано.")
        return

    sel = np.asarray(sel, int)
    xyz = pts[sel]

    result = {"selected": sel.tolist(),
              "xyz":      xyz.tolist()}
    json.dump(result, open(OUT_JSON, "w"), indent=2)

    print("\n=== picked_points.json ===")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
