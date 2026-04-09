"""STLメッシュの寸法を確認する簡易スクリプト"""
import struct
import os

def read_stl_bounds(filepath):
    """STLファイル(バイナリ)のバウンディングボックスを計算する"""
    with open(filepath, 'rb') as f:
        # ヘッダー(80バイト)とトライアングル数(4バイト)
        header = f.read(80)
        num_triangles = struct.unpack('<I', f.read(4))[0]

        min_xyz = [float('inf')] * 3
        max_xyz = [float('-inf')] * 3

        for _ in range(num_triangles):
            # 法線(12バイト) + 頂点3つ(各12バイト) + attr(2バイト) = 50バイト
            normal = struct.unpack('<3f', f.read(12))
            for v in range(3):
                vertex = struct.unpack('<3f', f.read(12))
                for i in range(3):
                    min_xyz[i] = min(min_xyz[i], vertex[i])
                    max_xyz[i] = max(max_xyz[i], vertex[i])
            attr = f.read(2)

    return min_xyz, max_xyz, num_triangles

mesh_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "urdf", "double pendulum", "double", "meshes"
)

for name in ["base.stl", "link1.stl", "link2.stl"]:
    path = os.path.join(mesh_dir, name)
    min_v, max_v, n_tri = read_stl_bounds(path)
    size = [max_v[i] - min_v[i] for i in range(3)]
    print(f"\n=== {name} ({n_tri} triangles) ===")
    print(f"  Min : ({min_v[0]:.3f}, {min_v[1]:.3f}, {min_v[2]:.3f}) [mm]")
    print(f"  Max : ({max_v[0]:.3f}, {max_v[1]:.3f}, {max_v[2]:.3f}) [mm]")
    print(f"  Size: ({size[0]:.3f}, {size[1]:.3f}, {size[2]:.3f}) [mm]")
    print(f"  Center: ({(min_v[0]+max_v[0])/2:.3f}, {(min_v[1]+max_v[1])/2:.3f}, {(min_v[2]+max_v[2])/2:.3f}) [mm]")
