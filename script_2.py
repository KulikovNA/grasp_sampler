# blender_antipodal.py
#
#  ▸ Что делает скрипт
#    1. Сбрасывает сцену Blender и включает импортер STL/OBJ/PLY.
#    2. Загружает указанный файл (SRC_PATH) как Object‑меш.
#    3. Равномерно по площади треугольников (!) сэмплирует NUM_SAMPLES точек:
#         • pts  — координаты в мировой системе (world space)
#         • nrms — нормали в world space
#       ⟶  на этой стадии НЕ отсекаем отверстия / внутренние стенки.
#    4. Для справки оценивает габаритную диагональ модели (diag) и
#       записывает «разумные» прикидки max_width = 0.3·diag,
#       max_depth = 0.1·diag (их будете менять под свой захват).
#    5. Сохраняет всё в NPZ, чтобы потом второй скрипт (Open3D)
#       строил антиподальные пары и превращал их в GraspGroup.

import bpy, addon_utils, bmesh
import numpy as np
from mathutils import Vector
from pathlib import Path

# --- Параметры ----------------------------------------------------------------
SRC_PATH    = "/home/nikita/diplom/testblend/test.stl"      # что грузим
OUT_NPZ     = "/home/nikita/diplom/testblend/antipodal_pairs.npz"
SCALE       = 1.0        # допмасштаб к модели (1.0 = как есть)
NUM_SAMPLES = 2000       # сколько точек сэмплировать

# ───────────────────────────────────────────────────────────────────────────────
# 1. Сброс сцены и включение нужных аддонов
def reset_and_enable_addons():
    bpy.ops.wm.read_factory_settings(use_empty=True)        # «чистый» Blender
    for mod in ("io_mesh_stl", "io_scene_obj", "io_mesh_ply"):
        try:
            addon_utils.enable(mod, default_set=True, persistent=True)
        except Exception as e:
            print("[WARN] не включён аддон", mod, ":", e)

# 2. Импорт модели (STL/OBJ/PLY) ------------------------------------------------
def import_mesh(src: Path, scale: float | None):
    before = set(bpy.data.objects.keys())                   # объекты ДО импорта
    if src.suffix.lower() == ".stl":
        bpy.ops.import_mesh.stl(filepath=str(src), global_scale=scale or 1.0)
    elif src.suffix.lower() == ".ply":
        bpy.ops.import_mesh.ply(filepath=str(src))
    else:  # obj, fbx, …
        bpy.ops.import_scene.obj(filepath=str(src))
    # получаем единственный новый объект
    obj_name = list(set(bpy.data.objects.keys()) - before)[0]
    obj = bpy.data.objects[obj_name]
    obj.name = src.stem
    return obj

# 3. Равномерный сэмплинг точек на поверхности ----------------------------------
def sample_mesh(obj, n):
    """
    Возвращает:
      pts  — (N,3) мировых координат
      nrms — (N,3) нормалей в world‑space
    """
    # --- перевели BMesh меша в триангулированный вид
    bm = bmesh.new(); bm.from_mesh(obj.data)
    bmesh.ops.triangulate(bm, faces=bm.faces)
    faces = list(bm.faces)
    
    # --- вероятность выбрать лицо ∝ площади (равномерно по поверхности)
    areas = np.array([f.calc_area() for f in faces], dtype=np.float64)
    areas /= areas.sum()

    # --- матрицы перехода в мировую систему
    M = obj.matrix_world.copy()   # 4×4
    R = M.to_3x3()                # только вращение/масштаб

    pts  = np.empty((n, 3), np.float32)
    nrms = np.empty((n, 3), np.float32)

    # для каждого сэмпла выбираем лицо по area‑PDF
    idxs = np.random.choice(len(faces), size=n, p=areas)
    for i, fi in enumerate(idxs):
        f = faces[fi]

        # барицентрические координаты (u,v,w) → случайная точка внутри треуг.
        u, v = np.random.rand(2)
        if u + v > 1.0:                 # чтобы точка была внутри
            u, v = 1 - u, 1 - v
        w = 1 - u - v
        v0, v1, v2 = (vt.co for vt in f.verts)   # локальные координаты
        p_loc = v0 * u + v1 * v + v2 * w         # Vector
        p_w   = M @ p_loc                        # world‑space Vector

        n_w   = (R @ f.normal).normalized()      # world‑space нормаль

        pts[i]  = (p_w.x, p_w.y, p_w.z)
        nrms[i] = (n_w.x, n_w.y, n_w.z)

    bm.free()
    return pts, nrms

# 4. Основная программа ---------------------------------------------------------
def main():
    reset_and_enable_addons()
    obj = import_mesh(Path(SRC_PATH), SCALE)

    # Сэмплируем точки
    pts, nrms = sample_mesh(obj, NUM_SAMPLES)

    # --- прикидываем масштаб детали через диагональ bbox
    bbox_world = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    diag = max((bbox_world[i] - bbox_world[j]).length
               for i in range(8) for j in range(8))

    max_width = 0.4 * diag   # «ширина губок» по умолчанию (30 % диагонали)
    max_depth = 0.2 * diag   # «глубина захода»      (10 % диагонали)

    # Сохраняем для последующей стадии (Open3D+GraspNetAPI)
    np.savez(OUT_NPZ,
             points     = pts,
             normals    = nrms,
             max_width  = max_width,
             max_depth  = max_depth)
    print(f"Saved: {OUT_NPZ}   samples: {len(pts)}")

if __name__ == "__main__":
    main()
