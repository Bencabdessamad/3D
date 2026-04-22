"""
Visualize an exported .obj file with Open3D.
Auteur : BENCHERAIK Abdessamad

Usage:
    python visualize_obj.py outputs/a_red_wooden_chair.obj
    python visualize_obj.py outputs/a_red_wooden_chair.obj --screenshot
"""

import argparse
import sys
from pathlib import Path

try:
    import open3d as o3d
except ImportError:
    sys.exit("open3d is not installed. Run: pip install open3d")


def load_mesh(path: Path) -> o3d.geometry.TriangleMesh:
    mesh = o3d.io.read_triangle_mesh(str(path))
    if not mesh.has_triangles():
        sys.exit(f"No geometry found in {path}")
    mesh.compute_vertex_normals()
    return mesh


def show(mesh: o3d.geometry.TriangleMesh, screenshot: Path | None = None):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Shap-E 3D Viewer", width=800, height=600)
    vis.add_geometry(mesh)
    opt = vis.get_render_option()
    opt.background_color = [0.1, 0.1, 0.1]
    opt.light_on = True

    if screenshot:
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(str(screenshot))
        print(f"Screenshot saved → {screenshot}")

    vis.run()
    vis.destroy_window()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Open3D viewer for .obj files")
    p.add_argument("obj_file", help="Path to .obj file")
    p.add_argument("--screenshot", metavar="FILE", help="Save a screenshot instead of opening the window")
    return p.parse_args()


def main():
    args = parse_args()
    obj_path = Path(args.obj_file)
    if not obj_path.exists():
        sys.exit(f"File not found: {obj_path}")

    mesh = load_mesh(obj_path)
    screenshot = Path(args.screenshot) if args.screenshot else None
    show(mesh, screenshot)


if __name__ == "__main__":
    main()
