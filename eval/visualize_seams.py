"""Blender script: visualize GT seams (blue) vs generated seams (red) as tubes.

Run:
  blender --background --python visualize_seams.py -- <mesh.obj> <gen_seam.json> <gt_seam.json> <output.blend>
"""
import bpy
import json
import sys
from mathutils import Vector

argv = sys.argv
args = argv[argv.index("--") + 1:] if "--" in argv else []
mesh_path, gen_seam_path, gt_seam_path, output_path = args[0], args[1], args[2], args[3]

# clear scene
bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete()
for block in list(bpy.data.meshes):
    bpy.data.meshes.remove(block)
for block in list(bpy.data.materials):
    bpy.data.materials.remove(block)
for block in list(bpy.data.curves):
    bpy.data.curves.remove(block)

# read mesh.obj
verts = []
faces = []
with open(mesh_path) as f:
    for line in f:
        t = line.split()
        if not t:
            continue
        if t[0] == "v":
            verts.append([float(t[1]), float(t[2]), float(t[3])])
        elif t[0] == "f":
            faces.append([int(t[1].split("/")[0]) - 1,
                          int(t[2].split("/")[0]) - 1,
                          int(t[3].split("/")[0]) - 1])

# create base mesh (gray)
mesh = bpy.data.meshes.new("BaseMesh")
mesh.from_pydata(verts, [], [tuple(f) for f in faces])
mesh.update()
obj = bpy.data.objects.new("GarmentMesh", mesh)
bpy.context.collection.objects.link(obj)

mat_gray = bpy.data.materials.new("GrayMat")
mat_gray.use_nodes = True
mat_gray.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.75, 0.75, 0.78, 1)
obj.data.materials.append(mat_gray)


def create_seam_tubes(name, seam_edges, color, radius=0.004):
    """Create a single curve object with all seam edges as poly splines,
    beveled into thin tubes for visibility."""
    curve_data = bpy.data.curves.new(name, type="CURVE")
    curve_data.dimensions = "3D"
    curve_data.bevel_depth = radius
    curve_data.bevel_resolution = 3
    curve_data.use_fill_caps = True

    for e in seam_edges:
        v1 = Vector(verts[int(e[0])])
        v2 = Vector(verts[int(e[1])])
        spline = curve_data.splines.new("POLY")
        spline.points.add(1)
        spline.points[0].co = (v1.x, v1.y, v1.z, 1)
        spline.points[1].co = (v2.x, v2.y, v2.z, 1)

    obj = bpy.data.objects.new(name, curve_data)
    bpy.context.collection.objects.link(obj)

    mat = bpy.data.materials.new(name + "_Mat")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*color, 1)
    bsdf.inputs["Roughness"].default_value = 0.4
    # emissive for extra visibility
    if "Emission" in bsdf.inputs:
        bsdf.inputs["Emission"].default_value = (*color, 1)
        bsdf.inputs["Emission Strength"].default_value = 0.5
    obj.data.materials.append(mat)
    return obj


# read seam edges
gen_seam = json.loads(open(gen_seam_path).read())
gt_seam = json.loads(open(gt_seam_path).read())

# GT seams - blue
create_seam_tubes("GT_Seams_Blue", gt_seam, (0.0, 0.3, 1.0))

# Generated seams - red
create_seam_tubes("Gen_Seams_Red", gen_seam, (1.0, 0.15, 0.0))

# shade smooth on base mesh
for poly in mesh.polygons:
    poly.use_smooth = True

# save
bpy.ops.wm.save_as_mainfile(filepath=output_path)
print(f"Saved: {output_path}")
