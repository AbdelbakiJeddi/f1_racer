"""Procedural Arena Generation CLI Tool.

Generates procedural racing tracks and corridors for Leatherback robot path tracking.
Acts as a simple CLI interface calling the core `ProceduralTrack` generator.
"""

import argparse
import math
import os
import sys
import numpy as np
from PIL import Image

# 1. Parse arguments BEFORE launching SimulationApp
parser = argparse.ArgumentParser(
    description="Procedural track generator CLI tool for Isaac Sim."
)
parser.add_argument(
    "--track_type",
    type=str,
    default="circuit",
    choices=["circuit", "organic", "wavy_circle", "s_curve", "slalom", "random"],
    help="Track style: circuit, organic, wavy_circle, s_curve, slalom, random.",
)
parser.add_argument(
    "--num_corners",
    type=int,
    default=12,
    help="Number of control points / corners for circuit mode.",
)
parser.add_argument(
    "--radius",
    type=float,
    default=12.0,
    help="Base scale / radius of the track in meters.",
)
parser.add_argument(
    "--waves",
    type=int,
    default=4,
    help="Number of wave oscillations (used in wavy_circle mode).",
)
parser.add_argument(
    "--amplitude",
    type=float,
    default=2.2,
    help="Amplitude of curves / variations in meters.",
)
parser.add_argument(
    "--width",
    type=float,
    default=1.2,
    help="Track width (distance between left & right lines) in meters.",
)
parser.add_argument(
    "--line_thickness",
    type=float,
    default=0.08,
    help="Thickness/width of each line stripe in meters.",
)
parser.add_argument(
    "--length",
    type=float,
    default=40.0,
    help="Path length for open tracks (s_curve, slalom, random) in meters.",
)
parser.add_argument(
    "--num_points",
    type=int,
    default=500,
    help="Number of points along the path.",
)
parser.add_argument(
    "--seed",
    type=int,
    default=42,
    help="Random seed for procedural variation.",
)
parser.add_argument(
    "--headless",
    action="store_true",
    default=False,
    help="Run headless without opening GUI window.",
)
parser.add_argument(
    "--arena_name",
    type=str,
    default=None,
    help="Folder and file name for the arena inside scenes directory.",
)
parser.add_argument(
    "--scenes_dir",
    type=str,
    default="isaacsim/scenes",
    help="Directory where arena folders are created.",
)
parser.add_argument(
    "--output_usd",
    type=str,
    default=None,
    help="Explicit USD path (overrides --scenes_dir / --arena_name).",
)
args = parser.parse_args()

# Determine arena directory and output paths
if args.output_usd:
    out_usd_path = os.path.abspath(args.output_usd)
    arena_dir = os.path.dirname(out_usd_path)
    base_name = os.path.splitext(os.path.basename(out_usd_path))[0]
else:
    base_name = (
        args.arena_name
        if args.arena_name
        else f"arena_{args.track_type}_s{args.seed}"
    )
    arena_dir = os.path.abspath(os.path.join(args.scenes_dir, base_name))
    out_usd_path = os.path.join(arena_dir, f"{base_name}.usd")

os.makedirs(arena_dir, exist_ok=True)

# 2. Start SimulationApp FIRST (Required for Omniverse Kit)
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": args.headless})

from pxr import Gf, UsdGeom
from isaacsim.core.utils.stage import get_current_stage, add_reference_to_stage
from isaacsim.sensors.camera import Camera

# Add src to path and import core library
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from leatherback_racer.arena import ProceduralTrack, TrackConfig

print("\n" + "=" * 70)
print(" PROCEDURAL ARENA GENERATOR (CLI)")
print("=" * 70)
print(f"  Track Style:    {args.track_type}")
print(f"  Base Radius:    {args.radius} m")
if args.track_type == "circuit":
    print(f"  Corners/Spline: {args.num_corners}")
print(f"  Track Width:    {args.width} m")
print(f"  Line Thickness: {args.line_thickness} m")
print(f"  Seed:           {args.seed}")
print("=" * 70 + "\n")

# 3. Generate Procedural Track via library
track_cfg = TrackConfig(
    track_type=args.track_type,
    num_corners=args.num_corners,
    radius=args.radius,
    waves=args.waves,
    amplitude=args.amplitude,
    track_width=args.width,
    line_thickness=args.line_thickness,
    length=args.length,
    num_samples=args.num_points,
    ground_size=max(80.0, (args.radius + args.amplitude) * 2.5),
    seed=args.seed,
)
track = ProceduralTrack(track_cfg)

# 4. Initialize USD Stage
stage = get_current_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
world_prim = stage.DefinePrim("/World", "Xform")
stage.SetDefaultPrim(world_prim)

# 5. Build Track Entities (Lighting, Ground, Ribbon Meshes, Top-Down Camera)
track.build_lighting(stage)
track.build_ground_plane(stage)
track.build_usd_meshes(stage)
top_cam_path, top_cam_height = track.build_top_down_camera(stage)

# 6. Spawn Leatherback Robot at track start pose
robot_usd_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../assets/leatherback.usd"))
if not os.path.exists(robot_usd_path):
    print(f"[ERROR] Could not find leatherback USD at: {robot_usd_path}")
    simulation_app.close()
    sys.exit(1)

robot_prim_path = "/World/Leatherback"
add_reference_to_stage(usd_path=robot_usd_path, prim_path=robot_prim_path)

start_pos, yaw0_rad = track.get_start_pose()
yaw0_deg = math.degrees(yaw0_rad)

robot_prim = stage.GetPrimAtPath(robot_prim_path)
trans_attr = robot_prim.GetAttribute("xformOp:translate")
if trans_attr.IsValid():
    trans_attr.Set(Gf.Vec3f(float(start_pos[0]), float(start_pos[1]), float(start_pos[2])))
else:
    UsdGeom.Xformable(robot_prim).AddTranslateOp().Set(
        Gf.Vec3f(float(start_pos[0]), float(start_pos[1]), float(start_pos[2]))
    )

quat_yaw = Gf.Rotation(Gf.Vec3d(0, 0, 1), yaw0_deg).GetQuat()
orient_attr = robot_prim.GetAttribute("xformOp:orient")
if orient_attr.IsValid():
    orient_attr.Set(quat_yaw)
else:
    UsdGeom.Xformable(robot_prim).AddOrientOp().Set(quat_yaw)

print(f"[INFO] Robot spawned at x={start_pos[0]:.2f}m, y={start_pos[1]:.2f}m, yaw={yaw0_deg:.1f} deg.")

# 7. Configure Cameras
chassis_path = f"{robot_prim_path}/Rigid_Bodies/Chassis"

# A. Centered Onboard Tracking Camera (mounted on hood centerline, pitched 25 deg down)
front_cam_path = f"{chassis_path}/FrontTrackCamera"
front_cam_prim = stage.DefinePrim(front_cam_path, "Camera")
front_cam_geom = UsdGeom.Camera(front_cam_prim)
front_cam_geom.CreateFocalLengthAttr(14.0)
front_cam_geom.CreateHorizontalApertureAttr(20.955)
front_cam_geom.CreateVerticalApertureAttr(15.716)
front_cam_geom.CreateClippingRangeAttr(Gf.Vec2f(0.05, 1000.0))

front_xform = UsdGeom.Xformable(front_cam_prim)
front_xform.AddTranslateOp().Set(Gf.Vec3f(16.0, 0.0, 19.5))
front_xform.AddOrientOp().Set(Gf.Quatf(0.548483, 0.443015, -0.548483, -0.443015))

front_cam = Camera(prim_path=front_cam_path, frequency=30, resolution=(640, 480))
front_cam.initialize()

# B. Third-Person Chase Camera
chase_cam_path = f"{chassis_path}/Camera_Chase"
chase_cam = Camera(prim_path=chase_cam_path, frequency=30, resolution=(640, 480))
chase_cam.initialize()

# C. Top-Down Camera
top_cam = Camera(prim_path=top_cam_path, frequency=30, resolution=(1024, 1024))
top_cam.initialize()

# 8. Save Stage and NPZ Metadata
stage.GetRootLayer().Export(out_usd_path)
print(f"[SUCCESS] Saved procedural arena stage to: {out_usd_path}")

npz_path = os.path.join(arena_dir, f"{base_name}_track_data.npz")
np.savez_compressed(
    npz_path,
    centerline=track.centerline,
    tangents=track.tangents,
    normals=track.normals,
    left_line=track.left_line,
    right_line=track.right_line,
    track_width=track.cfg.track_width,
    is_closed=track.is_closed,
)
print(f"[SUCCESS] Saved mathematical track metadata to: {npz_path}")

# 9. Capture Camera Views with Replicator
import omni.replicator.core as rep


def capture_camera_with_replicator(cam_prim_path, out_file, label, resolution=(640, 480)):
    print(f"[INFO] Capturing {label} via Replicator on {cam_prim_path}...")
    rp = rep.create.render_product(cam_prim_path, resolution)
    annot = rep.AnnotatorRegistry.get_annotator("rgb")
    annot.attach([rp])

    rep.orchestrator.step()
    for _ in range(15):
        simulation_app.update()

    data = annot.get_data()
    if data is not None and data.size > 0:
        rgb = data[:, :, :3]
        img = Image.fromarray(rgb)
        img.save(out_file)
        print(f"[SUCCESS] {label} saved to: {out_file}")
    else:
        print(f"[WARNING] Could not retrieve frame data for {label}.")
    rp.destroy()


top_down_png = os.path.join(arena_dir, f"{base_name}_top_down.png")
capture_camera_with_replicator(top_cam_path, top_down_png, "Top-Down Path Overview", resolution=(1024, 1024))
capture_camera_with_replicator(front_cam_path, os.path.join(arena_dir, "wavy_onboard_view.png"), "Onboard Camera View")
capture_camera_with_replicator(chase_cam_path, os.path.join(arena_dir, "wavy_chase_view.png"), "Chase Overview Camera View")

# 10. Interactive GUI loop if non-headless
if not args.headless:
    print("\n[INFO] Interactive GUI view active! Rotate/zoom in the viewport. Close window to exit.")
    while simulation_app.is_running():
        simulation_app.update()

simulation_app.close()
print("[INFO] Generation complete.")
