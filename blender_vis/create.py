import sys
import bpy
import trimesh
import argparse
import numpy as np 
from pathlib import Path 
import os 

from mathutils import Vector
import matplotlib.pyplot as plt
from matplotlib import cm


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--track_len', type=int, default=10)

    argv = sys.argv
    argv = argv[argv.index("--") + 1:]

    args = parser.parse_args(argv)

    track_len = int(args.track_len)

    basename = args.input.split('.')[0]
    data = np.load(args.input)
    # xyz = data[:,:,[2,0,1]]
    # msk = data[..., 0] == 0
    xyz = data[...,:3]
    msk = (xyz[0,...,2] != 0)
    mean = xyz[0][msk].mean(axis=0)

    xyz -= mean.reshape(1,1,3)
    xyz[...,-2] *= -1
    xyz[...,-1] *= -1
    
    rgb = data[:,:,[3,4,5]]

    pcd = trimesh.PointCloud(vertices=xyz[0], colors=rgb[0])
    pcd.export(f'{basename}.ply')

    ply_path = Path(f'{basename}.ply').absolute()

    # delete all objects
    # bpy.ops.object.select_all(action='SELECT')

    bpy.data.objects['Cube'].select_set(True)
    bpy.ops.object.delete()


    bpy.data.objects['Camera'].location = (mean[0],mean[1],mean[2])
    bpy.data.objects['Camera'].rotation_euler = (0,0,0)

    # add a plane
    bpy.ops.mesh.primitive_plane_add(size=2, enter_editmode=False, align='WORLD', location=(0,0,xyz[...,2].min()), scale=(10, 10, 10))

    bpy.ops.import_mesh.ply(filepath=str(ply_path),use_verts=True)
    

    obj = bpy.data.objects[basename]
    bpy.context.view_layer.objects.active = obj

    geo_node_modifier = obj.modifiers.new(name='GeometryNodes', type='NODES')
    bpy.ops.node.new_geometry_node_group_assign()



    node_tree = geo_node_modifier.node_group
    group_input = node_tree.nodes['Group Input']
    group_output = node_tree.nodes['Group Output']

    mesh2points = node_tree.nodes.new(type='GeometryNodeMeshToPoints')
    mesh2points.inputs['Radius'].default_value=0.01
    node_tree.links.new(group_input.outputs[0], mesh2points.inputs[0])

    setMaterial = node_tree.nodes.new(type='GeometryNodeSetMaterial')
    node_tree.links.new(mesh2points.outputs[0], setMaterial.inputs['Geometry'])
    node_tree.links.new(setMaterial.outputs['Geometry'], group_output.inputs[0])


    point_material = bpy.data.materials.new(name="PointMaterial")
    point_material.use_nodes = True
    # obj.data.materials[0] = point_material
    if obj.data.materials:
        obj.data.materials[0] = point_material
    else:
        obj.data.materials.append(point_material)

    setMaterial.inputs['Material'].default_value = point_material

    MatNodeTree = point_material.node_tree

    MatNodeTree.nodes.clear()

    attribute_node = MatNodeTree.nodes.new(type='ShaderNodeAttribute')
    attribute_node.attribute_name = 'Col'
    attribute_node.attribute_type = 'GEOMETRY'

    shader_node = MatNodeTree.nodes.new(type='ShaderNodeBsdfPrincipled')
    MatNodeTree.links.new(attribute_node.outputs['Color'], shader_node.inputs['Base Color'])

    shader_node.inputs['Metallic'].default_value = 0.77

    shader_output = MatNodeTree.nodes.new(type='ShaderNodeOutputMaterial')
    MatNodeTree.links.new(shader_node.outputs['BSDF'], shader_output.inputs['Surface'])

    # set cycles as render engine
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.device = 'GPU'

    # set the number of samples

    # set start and end frame
    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = xyz.shape[0]


    cmap = cm.get_cmap("gist_rainbow")
    tcolors = np.zeros((data.shape[0],data.shape[1],4))

    T, N = xyz.shape[:2]
    for t in range(T):	
        y_min = np.min(data[t,:,1])
        y_max = np.max(data[t,:,1])
        norm = plt.Normalize(y_min,y_max)
        for n in range(N):
            color = cmap(norm(data[t,n,1]))
            tcolors[t,n] = color

    tmap = tcolors
    tmap[...,-1]*=0.3

    def create_trajectory(i, length=5):
        name = 'Curve.{:04d}'.format(i)
        curve_data = bpy.data.curves.new(name, type='CURVE')
        curve_data.dimensions = '3D'
        curve_data.fill_mode = 'FULL'
        curve_data.bevel_depth = 0.003
        curve_data.bevel_resolution = 3
    #    
        curve_object = bpy.data.objects.new(name, curve_data)
        bpy.context.collection.objects.link(curve_object)
    #    
        polyline = curve_data.splines.new('POLY')
        polyline.points.add(length-1)
        mat = create_emission_material('Mat.{:04d}'.format(i),tmap[0,i],1)

        # Add the points to the spline
        
        for _ in range(length):
            x, y, z = xyz[_,i]
            polyline.points[_].co = (0, 0, 0, 1)
        
        curve_object.data.materials.append(mat)
    #        
        return curve_object

    def create_emission_material(name, color, strength):
        material = bpy.data.materials.new(name)
        material.use_nodes = True
        nodes = material.node_tree.nodes
        nodes.clear()

        emission = nodes.new(type='ShaderNodeEmission')
        emission.inputs['Color'].default_value = color
        
        emission.inputs['Strength'].default_value = strength

        material_output = nodes.new(type='ShaderNodeOutputMaterial')
        material_output.location = 400, 0

        links = material.node_tree.links
        link = links.new(emission.outputs['Emission'], material_output.inputs['Surface'])

        return material    

    for t in range(T):
        for i in range(N):
            x,y,z = xyz[t,i]
            obj.data.vertices[i].co = Vector((x,y,z))
            obj.data.vertices[i].keyframe_insert(data_path='co',frame=t)
            
    curve_list = [ create_trajectory(i, track_len) for i in range(N)]
    #for i in range(N):
    #    create_trajectory(i)
    for t in range(T):
        for i, curve in enumerate(curve_list):
            for j in range(track_len):
                u = max(t-j,0)
                x,y,z = xyz[u,i,:3]
                if z == 0:
                    continue
                if j > 0:
                    if abs(curve.data.splines[0].points[j-1].co[2]-z)>0.5:
                        for k in range(j):
                            curve.data.splines[0].points[k].co = (x,y,z,1)
                            curve.data.splines[0].points[k].keyframe_insert(data_path='co',frame=t)
                curve.data.splines[0].points[j].co = (x,y,z,1)
                curve.data.splines[0].points[j].keyframe_insert(data_path='co',frame=t)        
            # filter out the points that are not visible in the first frame
    
    # set the render engine to cycles
    # create the blender file
    if os.path.exists(f'{basename}.blend'):
        os.remove(f'{basename}.blend')

    bpy.ops.wm.save_as_mainfile(filepath=f'{basename}.blend')
    # then quit
    # bpy.ops.wm.quit_blender()
