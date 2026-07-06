import bpy

# clean scene (drop default cube/camera/light)
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o, do_unlink=True)

# import the exported avatar (new stack: 52/52 ARKit morphs, photo texture, tongueOut)
bpy.ops.import_scene.gltf(filepath="/home/darpa/Desktop/newARC/out/head_arkit_v2.glb")

# The glTF importer assigns HASHED blend + "show backface" to the material, which
# renders the skin SEE-THROUGH (you then see the internal teeth/tongue/eyeballs and
# it looks like a hole in the back of the head). The GLB is opaque (alphaMode=OPAQUE);
# force the Blender material opaque so the head renders solid. This is a Blender-viewer
# fix only -- a compliant glTF viewer (three.js) already renders it opaque.
for _m in bpy.data.materials:
    for _attr, _val in (("blend_method", 'OPAQUE'),
                        ("surface_render_method", 'DITHERED'),
                        ("show_transparent_back", False),
                        ("use_backface_culling", True),
                        ("use_transparent_shadow", False)):
        try:
            setattr(_m, _attr, _val)
        except (AttributeError, TypeError):
            pass  # property renamed/removed across Blender versions

# select + frame the head, and show the baked texture (Material Preview)
mesh = next((o for o in bpy.data.objects if o.type == 'MESH'), None)
if mesh:
    bpy.ops.object.select_all(action='DESELECT')
    mesh.select_set(True)
    bpy.context.view_layer.objects.active = mesh

for win in bpy.context.window_manager.windows:
    for area in win.screen.areas:
        if area.type == 'VIEW_3D':
            for space in area.spaces:
                if space.type == 'VIEW_3D':
                    space.shading.type = 'MATERIAL'

print("[open_avatar] imported head_arkit_v2.glb (52 ARKit morph targets, incl. tongueOut)")
