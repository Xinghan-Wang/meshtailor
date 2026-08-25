"""Blender bpy: import mesh, mark seam edges, UV-unwrap along seams, export.

Run with:
  blender.exe --background --python unwrap_blender.py -- <input.obj> <seam.json> <output.obj>
"""
import json
import sys

import bpy

argv = sys.argv
args = argv[argv.index("--") + 1:] if "--" in argv else []
input_obj, seam_json, output_obj = args[0], args[1], args[2]

# clean scene
bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)
for block in list(bpy.data.meshes):
    bpy.data.meshes.remove(block)

bpy.ops.wm.obj_import(filepath=input_obj)
obj = bpy.context.active_object
mesh = obj.data
print(f"imported: V={len(mesh.vertices)} E={len(mesh.edges)} F={len(mesh.polygons)}")

with open(seam_json) as f:
    seam = [tuple(e) for e in json.load(f)]

edge_map = {}
for i, e in enumerate(mesh.edges):
    a, b = sorted([e.vertices[0], e.vertices[1]])
    edge_map[(a, b)] = i

n_marked = 0
for (v0, v1) in seam:
    key = (min(v0, v1), max(v0, v1))
    ei = edge_map.get(key)
    if ei is not None:
        mesh.edges[ei].use_seam = True
        n_marked += 1
print(f"seams marked: {n_marked}/{len(seam)}")

bpy.context.view_layer.objects.active = obj
obj.select_set(True)
bpy.ops.object.mode_set(mode="EDIT")
bpy.ops.mesh.select_all(action="SELECT")
unwrapped = False
for method in ("ANGLE_BASED", "CONFORMAL"):
    try:
        bpy.ops.uv.unwrap(method=method)
        print(f"unwrap method={method} OK")
        unwrapped = True
        break
    except Exception as e:
        print(f"unwrap method={method} failed: {e}")
if not unwrapped:
    bpy.ops.uv.unwrap()
    print("unwrap (default) OK")
bpy.ops.object.mode_set(mode="OBJECT")

uv_layer = obj.data.uv_layers.active
if uv_layer:
    print(f"UV layer: {uv_layer.name} uv_count={len(uv_layer.data)}")
else:
    print("NO UV layer!")

bpy.ops.wm.obj_export(filepath=output_obj, export_uv=True)
print(f"exported -> {output_obj}")
