import pymesh
import trimesh
import numpy as np
from itertools import product
import open3d as o3d


def load_pc(path):
    # with stdout_redirected():
    #     res = o3d.io.read_point_cloud(path)
    res = pymesh.load_mesh(path)
    return res


def dump_pc(path, pc):
    # with stdout_redirected():
    #     o3d.io.write_point_cloud(path, pc)
    pymesh.save_mesh(path, pc)


def get_pc(pc):
    return pc.vertices


def pymesh_to_trimesh(mesh_pymesh):
    return trimesh.Trimesh(vertices=mesh_pymesh.vertices, faces=mesh_pymesh.faces)


def get_bbox(mesh):
    mesh_tri = pymesh_to_trimesh(mesh)
    obb = mesh_tri.bounding_box_oriented
    extents_raw = np.array(obb.primitive.extents)
    extents = np.array(obb.primitive.extents) / 2
    transform = np.array(obb.primitive.transform)
    template = [-1, 1]
    res = []
    for mask in product(template, template, template):
        relative = mask * extents
        relative_homo = np.array([*(relative.tolist()), 1])
        world_homo = np.dot(transform, relative_homo)
        res.append(world_homo[:-1])
    return np.row_stack(res), extents_raw, transform


def get_brect(plane_mesh):
    pc = plane_mesh.vertices
    p_mean = np.mean(pc, axis=0)
    cov_mat = np.cov(pc, rowvar=False, bias=True)
    eign, eigv = np.linalg.eigh(cov_mat)
    u, v, n = eigv[:, 2], eigv[:, 1], eigv[:, 0]
    # print(eign)
    # print(u, v)
    # print(np.linalg.norm(u), np.linalg.norm(v), np.dot(u, v))
    rotation = np.array((n, v, u)).T
    pc_projected = np.dot(rotation, pc.T).T

    pc_projected_min = np.min(pc_projected, axis=0)
    pc_projected_max = np.max(pc_projected, axis=0)

    pc_projected_rect_list = [
        pc_projected_min,
        np.array((pc_projected_min[0], pc_projected_min[1], pc_projected_max[2])),
        np.array((pc_projected_min[0], pc_projected_max[1], pc_projected_min[2])),
        np.array((pc_projected_min[0], pc_projected_max[1], pc_projected_max[2])),
        np.array((pc_projected_max[0], pc_projected_min[1], pc_projected_min[2])),
        np.array((pc_projected_max[0], pc_projected_min[1], pc_projected_max[2])),
        np.array((pc_projected_max[0], pc_projected_max[1], pc_projected_min[2])),
        pc_projected_max
    ]
    pc_projected_rect = np.stack(pc_projected_rect_list)
    bbox = np.dot(rotation.T, pc_projected_rect.T).T

    extent = pc_projected_max - pc_projected_min
    transform = np.zeros((4, 4))
    transform[:3, :3] = rotation.T
    centroid = np.dot(rotation.T, (pc_projected_max + pc_projected_min) / 2)
    transform[:3, 3] = centroid
    transform[3, 3] = 1

    return bbox, transform, extent


def get_bbox_extent(bbox):
    # hard-coded
    pairs = [(0, 1), (0, 2), (0, 4)]
    res = []
    for a, b in pairs:
        res.append(np.linalg.norm(bbox[a, :] - bbox[b, :]))
    return np.array(res)


def get_bbox_volume(bbox):
    # hard-coded
    pairs = [(0, 1), (0, 2), (0, 4)]
    res = []
    for a, b in pairs:
        res.append(np.linalg.norm(bbox[a, :] - bbox[b, :]))
    return res[0] * res[1] * res[2]


def get_border_edge(mesh):
    vertices, edges = pymesh.mesh_to_graph(mesh)
    print(np.all(vertices == mesh.vertices))


def draw_boxes3d(gt_boxes3d, color=(1, 1, 1), line_width=1, draw_text=True, text_scale=(0.01, 0.01, 0.01),
                 color_list=None):
    from mayavi import mlab
    num = len(gt_boxes3d)
    for n in range(num):
        b = gt_boxes3d[n]
        if color_list is not None:
            color = color_list[n]
        if draw_text: mlab.text3d(b[7, 0], b[7, 1], b[7, 2], '%d' % n,
                                  scale=text_scale, color=color)

        # hard coded
        # pairs = [(0, 1), (1, 2), (1, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
        pairs = [(0, 1), (0, 2), (3, 1), (3, 2), (4, 5), (4, 6), (7, 5), (7, 6), (0, 4), (1, 5), (2, 6), (3, 7)]
        for i, j in pairs:
            mlab.plot3d([b[i, 0], b[j, 0]], [b[i, 1], b[j, 1]], [b[i, 2], b[j, 2]],
                        color=color, tube_radius=None, line_width=line_width)
