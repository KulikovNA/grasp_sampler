# grasp_sampler


В script_2 поменять пути для test.stl 

Запустить script_2 чтобы сгенерировать точки на каждом треугольнике меша. Они будут сохранены в `antipodal_pairs.npz`

```bash 
# конда среда с o3d и graspnetAPI
conda activate sbg38 

python script_2 
```

Запустить `picked_points.py` чтобы выбрать точки. Точки выбираются через `shift+ЛКМ` 

```bash 
python picked_points.py
```

И терминала скопировать номера точек и вставить в `single_grasp_from_picks.py`
Отрисуется захват в o3d 

```bash
python single_grasp_from_picks.py
```

Неудачное построение захватов по `antipodal_pairs.npz`

```bash
python o3d_grasp_view.py
```