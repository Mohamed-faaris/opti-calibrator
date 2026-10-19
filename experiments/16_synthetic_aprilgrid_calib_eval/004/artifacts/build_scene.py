import bpy, os
from mathutils import Vector, Euler

out = r"/home/mfk01/Projects/econ-systems/demos/a2-01/experiment/16_synthetic_aprilgrid_calib_eval/004/artifacts"
os.makedirs(out, exist_ok=True)

bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)

scene = bpy.context.scene
available_engines = {item.identifier for item in scene.render.bl_rna.properties["engine"].enum_items}
scene.render.engine = "BLENDER_EEVEE_NEXT" if "BLENDER_EEVEE_NEXT" in available_engines else "BLENDER_EEVEE"
scene.render.resolution_x = 1280
scene.render.resolution_y = 720
scene.render.resolution_percentage = 100
scene.render.filepath = os.path.join(out, "render.png")
scene.view_settings.view_transform = "Standard"
scene.view_settings.look = "None"
scene.view_settings.exposure = 0.0
scene.view_settings.gamma = 1.0

world = bpy.data.worlds.new("Background")
world.use_nodes = True
world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.5, 0.5, 0.5, 1.0)
scene.world = world

cam_data = bpy.data.cameras.new("GroundTruthCamera")
cam_data.lens = 920.0 * 36.0 / 1280
cam_data.sensor_width = 36.0
cam_data.sensor_fit = "HORIZONTAL"
cam_data.clip_start = 0.05
cam_data.clip_end = 20.0
cam = bpy.data.objects.new("Camera", cam_data)
scene.collection.objects.link(cam)
scene.camera = cam

mesh = bpy.data.meshes.new("TargetBoardMesh")
mesh.from_pydata([(-0.17, 0.118, 0), (0.17, 0.118, 0), (0.17, -0.118, 0), (-0.17, -0.118, 0)], [], [(0, 1, 2, 3)])
mesh.update()

uv = mesh.uv_layers.new(name="UVMap")
for loop_idx, uv_pt in zip(mesh.polygons[0].loop_indices, [(0, 1), (1, 1), (1, 0), (0, 0)]):
    uv.data[loop_idx].uv = uv_pt

board_obj = bpy.data.objects.new("TargetBoard", mesh)
scene.collection.objects.link(board_obj)
board_obj.location = (-0.06, -0.0, -0.82)
board_obj.rotation_euler = (0.0, 0.31, 0.0)

mat = bpy.data.materials.new("TargetMaterial")
mat.use_nodes = True
nodes = mat.node_tree.nodes
nodes.clear()
img_node = nodes.new("ShaderNodeTexImage")
img_node.image = bpy.data.images.load(r"/home/mfk01/Projects/econ-systems/demos/a2-01/experiment/16_synthetic_aprilgrid_calib_eval/shared/aprilgrid_texture.png", check_existing=True)
img_node.interpolation = "Linear"
emit_node = nodes.new("ShaderNodeEmission")
emit_node.inputs["Strength"].default_value = 1.0
out_node = nodes.new("ShaderNodeOutputMaterial")
mat.node_tree.links.new(img_node.outputs["Color"], emit_node.inputs["Color"])
mat.node_tree.links.new(emit_node.outputs["Emission"], out_node.inputs["Surface"])
board_obj.data.materials.append(mat)

bpy.ops.wm.save_as_mainfile(filepath=os.path.join(out, "scene.blend"))
bpy.ops.render.render(write_still=True)
