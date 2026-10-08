import argparse
import math
import os
import sys
import numpy as np
from PIL import Image

# 1. Parse arguments BEFORE launching SimulationApp
parser = argparse.ArgumentParser(description="Generate a closed wavy circle arena with left & right boundary lines.")
parser.add_argument("--track_type", type=str, default="circuit", choices=["circuit", "organic", "wavy_circle"],
                    help="Track style: 'circuit' (natural racetrack with straights and varying corners), 'organic' (continuous asymmetric flowing curves), or 'wavy_circle' (harmonic wave).")
parser.add_argument("--num_corners", type=int, default=12, help="Number of control points / corners for circuit mode.")
parser.add_argument("--radius", type=float, default=12.0, help="Base scale / radius of the track in meters.")
parser.add_argument("--waves", type=int, default=4, help="Number of wave oscillations (used in wavy_circle mode).")
parser.add_argument("--amplitude", type=float, default=2.2, help="Amplitude of curves / variations in meters.")
parser.add_argument("--width", type=float, default=1.2, help="Track width (distance between left & right lines) in meters.")
parser.add_argument("--line_thickness", type=float, default=0.08, help="Thickness/width of each line stripe in meters.")
parser.add_argument("--num_points", type=int, default=500, help="Number of points along the closed loop.")
parser.add_argument("--seed", type=int, default=42, help="Random seed for procedural variation.")
parser.add_argument("--headless", action="store_true", default=False, help="Run headless to save USD and capture images without opening GUI window.")
parser.add_argument("--arena_name", type=str, default=None, help="Folder and file name for the arena inside scenes directory (defaults to arena_<track_type>_s<seed>).")
parser.add_argument("--scenes_dir", type=str, default="isaacsim/scenes", help="Directory where arena folders are created.")
parser.add_argument("--output_usd", type=str, default=None, help="Explicit USD path (overrides --scenes_dir / --arena_name).")
args = parser.parse_args()

# Determine arena directory and output paths
if args.output_usd:
    out_usd_path = os.path.abspath(args.output_usd)
    arena_dir = os.path.dirname(out_usd_path)
    base_name = os.path.splitext(os.path.basename(out_usd_path))[0]
else:
    base_name = args.arena_name if args.arena_name else f"arena_{args.track_type}_s{args.seed}"
    arena_dir = os.path.abspath(os.path.join(args.scenes_dir, base_name))
    out_usd_path = os.path.join(arena_dir, f"{base_name}.usd")

os.makedirs(arena_dir, exist_ok=True)

# 2. Start SimulationApp FIRST (Required for Omniverse Kit)
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": args.headless})

import omni.usd
import omni.timeline
from pxr import Usd, UsdGeom, UsdShade, UsdLux, Gf, Sdf
from isaacsim.core.utils.stage import get_current_stage, add_reference_to_stage
from isaacsim.sensors.camera import Camera

print("\n" + "=" * 70)
print(" PROCEDURAL ARENA: MOTORSPORT CIRCUIT & PATH TRACKING ARENA")
print("=" * 70)
print(f"  Track Style:    {args.track_type}")
print(f"  Base Radius:    {args.radius} m")
if args.track_type == "circuit":
    print(f"  Corners/Spline: {args.num_corners}")
elif args.track_type == "organic":
    print(f"  Multi-Harmonics: 1..8 (Asymmetric Octaves)")
else:
    print(f"  Wave Count:     {args.waves}")
    print(f"  Wave Amplitude: {args.amplitude} m")
print(f"  Track Width:    {args.width} m (Left to Right line distance)")
print(f"  Line Thickness: {args.line_thickness} m")
print(f"  Seed:           {args.seed}")
print("=" * 70 + "\n")

# 3. Procedural Track Geometry Generation
rng = np.random.default_rng(args.seed)

if args.track_type == "circuit":
    # Natural Motorsport Circuit: Non-uniform spline control points with realistic straights and corners
    import scipy.interpolate as interp
    num_ctrl = max(8, args.num_corners)
    base_angles = np.linspace(0, 2 * math.pi, num_ctrl, endpoint=False)
    # Jitter angles non-uniformly so some sectors are long straights and others form tight corner complexes
    angle_jitter = rng.uniform(-0.35, 0.35, size=num_ctrl) * (2 * math.pi / num_ctrl)
    angles = np.sort(base_angles + angle_jitter)

    # Vary radii between wide sweeping corners and technical turns
    radii = rng.uniform(args.radius * 0.7, args.radius * 1.35, size=num_ctrl)

    # Organic aspect ratio stretching and principal axis rotation
    aspect = float(rng.uniform(1.15, 1.45))
    rot = float(rng.uniform(0.0, math.pi))
    c, s = math.cos(rot), math.sin(rot)

    x_raw = radii * np.cos(angles)
    y_raw = radii * np.sin(angles)
    x_rot = (c * x_raw + s * y_raw) * aspect
    y_rot = -s * x_raw + c * y_raw
    x_ctrl = c * x_rot - s * y_rot
    y_ctrl = s * x_rot + c * y_rot

    pts = np.vstack([x_ctrl, y_ctrl])
    pts_closed = np.hstack([pts, pts[:, :1]])
    tck, _ = interp.splprep(pts_closed, s=0, per=True, k=3)
    u_fine = np.linspace(0, 1, args.num_points, endpoint=False)
    x, y = interp.splev(u_fine, tck)
    centerline = np.stack([x, y], axis=1)

elif args.track_type == "organic":
    # Multi-Octave Fourier Synthesis: Summing independent non-harmonic octaves k=1..8 to break all symmetry
    theta = np.linspace(0.0, 2.0 * math.pi, args.num_points, endpoint=False)
    r = np.full_like(theta, args.radius)
    for k in range(1, 9):
        # 1/f falloff guarantees smooth, physical Ackermann-drivable curvature
        weight = args.amplitude * (1.0 / (k ** 0.85))
        a = rng.uniform(0.3 * weight, 1.2 * weight)
        phi = rng.uniform(0.0, 2.0 * math.pi)
        r += a * np.sin(k * theta + phi)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    centerline = np.stack([x, y], axis=1)

else:  # "wavy_circle"
    # Legacy harmonic wave
    theta = np.linspace(0.0, 2.0 * math.pi, args.num_points, endpoint=False)
    n1 = args.waves
    a1 = args.amplitude
    phi1 = float(rng.uniform(0.0, 2.0 * math.pi))
    n2 = args.waves * 2
    a2 = args.amplitude * 0.35
    phi2 = float(rng.uniform(0.0, 2.0 * math.pi))
    r = args.radius + a1 * np.sin(n1 * theta + phi1) + a2 * np.sin(n2 * theta + phi2)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    centerline = np.stack([x, y], axis=1)

# Analytical or finite-difference derivatives for seamless periodic tangents
dx = np.roll(x, -1) - np.roll(x, 1)
dy = np.roll(y, -1) - np.roll(y, 1)
tangent_lengths = np.sqrt(dx**2 + dy**2)
tangent_lengths[tangent_lengths < 1e-6] = 1e-6

tx = dx / tangent_lengths
ty = dy / tangent_lengths
tangents = np.stack([tx, ty], axis=1)

# Unit normal vectors pointing to the left: (-ty, tx)
nx = -ty
ny = tx
normals = np.stack([nx, ny], axis=1)

# Left and Right Boundary Lines (Copies offset by half track width)
half_w = args.width / 2.0
left_line = centerline + half_w * normals
right_line = centerline - half_w * normals

print(f"[INFO] Generated closed loop with {len(centerline)} sample points.")
print(f"       Total Track Perimeter: ~{np.sum(tangent_lengths / 2.0):.2f} meters.")

# 4. Initialize USD Stage
stage = get_current_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)

world_prim = stage.DefinePrim("/World", "Xform")
stage.SetDefaultPrim(world_prim)

# 5. Lighting (Calibrated for high contrast & realistic RTX exposure)
dome_light = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
dome_light.CreateIntensityAttr(250.0)
dome_light.CreateColorAttr(Gf.Vec3f(0.8, 0.85, 0.95))  # Soft sky ambient

sun_light = UsdLux.DistantLight.Define(stage, "/World/SunLight")
sun_light.CreateIntensityAttr(800.0)
sun_light.CreateColorAttr(Gf.Vec3f(1.0, 0.98, 0.92))   # Warm directional sun
sun_light.AddRotateXYZOp().Set(Gf.Vec3f(-45.0, 30.0, 0.0))

# 6. Materials: Dark Grey Ground + Racing Yellow Lines
def create_pbr_material(mat_path, color_rgb, emissive_rgb=(0, 0, 0), roughness=0.5):
    mat = UsdShade.Material.Define(stage, mat_path)
    shader = UsdShade.Shader.Define(stage, f"{mat_path}/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color_rgb))
    shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*emissive_rgb))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return mat

# Dark Grey Ground (Asphalt / Tarmac)
ground_mat = create_pbr_material("/World/Looks/GroundMaterial", [0.12, 0.12, 0.12], roughness=0.85)

# High-Visibility Racing Yellow Boundary Lines
line_mat = create_pbr_material("/World/Looks/LineMaterial", [1.0, 0.82, 0.0], emissive_rgb=(0.0, 0.0, 0.0), roughness=0.3)

# 7. Ground Plane
ground_size = max(50.0, (args.radius + args.amplitude) * 2.5)
ground_geom = UsdGeom.Plane.Define(stage, "/World/Ground")
ground_geom.CreateAxisAttr("Z")
ground_geom.CreateWidthAttr(ground_size)
ground_geom.CreateLengthAttr(ground_size)
UsdShade.MaterialBindingAPI.Apply(ground_geom.GetPrim()).Bind(ground_mat)

# 8. Create 3D Ribbon Meshes for Left and Right Lines
def build_closed_ribbon_mesh(prim_path, points_2d, norm_2d, thickness, material):
    mesh = UsdGeom.Mesh.Define(stage, prim_path)
    half_th = thickness / 2.0
    z_h = 0.003  # Slightly elevated above ground to prevent z-fighting

    v_left = points_2d + half_th * norm_2d
    v_right = points_2d - half_th * norm_2d

    n = len(points_2d)
    pts_usd = []
    for i in range(n):
        pts_usd.append(Gf.Vec3f(float(v_left[i, 0]), float(v_left[i, 1]), z_h))
        pts_usd.append(Gf.Vec3f(float(v_right[i, 0]), float(v_right[i, 1]), z_h))

    face_counts = [4] * n
    face_indices = []
    for i in range(n):
        i_next = (i + 1) % n
        idx0 = 2 * i
        idx1 = 2 * i + 1
        idx2 = 2 * i_next + 1
        idx3 = 2 * i_next
        face_indices.extend([idx0, idx1, idx2, idx3])

    mesh.CreatePointsAttr(pts_usd)
    mesh.CreateFaceVertexCountsAttr(face_counts)
    mesh.CreateFaceVertexIndicesAttr(face_indices)
    mesh.CreateNormalsAttr([Gf.Vec3f(0.0, 0.0, 1.0)] * len(pts_usd))
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    mesh.CreateDoubleSidedAttr(True)
    mesh.CreateDisplayColorAttr([Gf.Vec3f(1.0, 0.82, 0.0)])  # High-visibility racing yellow
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    return mesh

print("[INFO] Building 3D Left and Right boundary line meshes (Racing Yellow)...")
build_closed_ribbon_mesh("/World/Track/LeftBoundaryLine", left_line, normals, args.line_thickness, line_mat)
build_closed_ribbon_mesh("/World/Track/RightBoundaryLine", right_line, normals, args.line_thickness, line_mat)

# 9. Spawn Leatherback Robot between the two lines at theta=0
robot_usd_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../assets/leatherback.usd"))
if not os.path.exists(robot_usd_path):
    print(f"[ERROR] Could not find leatherback USD at: {robot_usd_path}")
    simulation_app.close()
    sys.exit(1)

robot_prim_path = "/World/Leatherback"
add_reference_to_stage(usd_path=robot_usd_path, prim_path=robot_prim_path)

# Spawn at centerline point 0, aligned along tangent 0
x0, y0 = centerline[0]
yaw0_rad = math.atan2(tangents[0, 1], tangents[0, 0])
yaw0_deg = math.degrees(yaw0_rad)

robot_prim = stage.GetPrimAtPath(robot_prim_path)
trans_attr = robot_prim.GetAttribute("xformOp:translate")
if trans_attr.IsValid():
    trans_attr.Set(Gf.Vec3f(float(x0), float(y0), 0.03))
else:
    UsdGeom.Xformable(robot_prim).AddTranslateOp().Set(Gf.Vec3f(float(x0), float(y0), 0.03))

# Rotate robot to face along the track direction (using existing xformOp:orient with GfQuatd)
quat_yaw = Gf.Rotation(Gf.Vec3d(0, 0, 1), yaw0_deg).GetQuat()
orient_attr = robot_prim.GetAttribute("xformOp:orient")
if orient_attr.IsValid():
    orient_attr.Set(quat_yaw)
else:
    UsdGeom.Xformable(robot_prim).AddOrientOp().Set(quat_yaw)

print(f"[INFO] Robot spawned at x={x0:.2f}m, y={y0:.2f}m, yaw={yaw0_deg:.1f} deg.")

# 10. Camera Sensors: Centered FrontTrackCamera for CNN & Chase Camera for Overview
chassis_path = f"{robot_prim_path}/Rigid_Bodies/Chassis"

# A. Centered Onboard Tracking Camera (mounted on front hood centerline, pitched down 25 deg)
# Note: Chassis link uses centimeter local units (16.0 cm forward, 0.0 cm center, 19.5 cm height)
front_cam_path = f"{chassis_path}/FrontTrackCamera"
front_cam_prim = stage.DefinePrim(front_cam_path, "Camera")
front_cam_geom = UsdGeom.Camera(front_cam_prim)
front_cam_geom.CreateFocalLengthAttr(14.0)
front_cam_geom.CreateHorizontalApertureAttr(20.955)
front_cam_geom.CreateVerticalApertureAttr(15.716)
front_cam_geom.CreateClippingRangeAttr(Gf.Vec2f(0.05, 1000.0))

# Pitch 25 degrees downward: Quatf(0.548483, 0.443015, -0.548483, -0.443015)
front_xform = UsdGeom.Xformable(front_cam_prim)
front_xform.AddTranslateOp().Set(Gf.Vec3f(16.0, 0.0, 19.5))
front_xform.AddOrientOp().Set(Gf.Quatf(0.548483, 0.443015, -0.548483, -0.443015))

print(f"[INFO] Initializing Centered FrontTrackCamera: {front_cam_path}")
front_cam = Camera(prim_path=front_cam_path, frequency=30, resolution=(640, 480))
front_cam.initialize()

# B. Third-Person Chase Camera
chase_cam_path = f"{chassis_path}/Camera_Chase"
print(f"[INFO] Initializing Chase Camera: {chase_cam_path}")
chase_cam = Camera(prim_path=chase_cam_path, frequency=30, resolution=(640, 480))
chase_cam.initialize()

# C. Top-Down Camera framing the entire arena from above
all_pts = np.vstack([left_line, right_line])
x_min, x_max = float(np.min(all_pts[:, 0])), float(np.max(all_pts[:, 0]))
y_min, y_max = float(np.min(all_pts[:, 1])), float(np.max(all_pts[:, 1]))
cx = (x_min + x_max) / 2.0
cy = (y_min + y_max) / 2.0
span_x = x_max - x_min
span_y = y_max - y_min
max_span = max(span_x, span_y) * 1.25  # 25% margin to fully frame the track

focal_length = 24.0
aperture = 20.955
top_cam_height = max((max_span / 2.0) * (2.0 * focal_length / aperture), 12.0)

top_cam_path = "/World/TopDownCamera"
top_cam_prim = stage.DefinePrim(top_cam_path, "Camera")
top_cam_geom = UsdGeom.Camera(top_cam_prim)
top_cam_geom.CreateFocalLengthAttr(focal_length)
top_cam_geom.CreateHorizontalApertureAttr(aperture)
top_cam_geom.CreateVerticalApertureAttr(aperture)
top_cam_geom.CreateClippingRangeAttr(Gf.Vec2f(0.1, 10000.0))

top_cam_xform = UsdGeom.Xformable(top_cam_prim)
top_cam_xform.AddTranslateOp().Set(Gf.Vec3f(float(cx), float(cy), float(top_cam_height)))
top_cam_xform.AddOrientOp().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))

print(f"[INFO] Initializing Top-Down Camera at ({cx:.1f}, {cy:.1f}, {top_cam_height:.1f}m)...")
top_cam = Camera(prim_path=top_cam_path, frequency=30, resolution=(1024, 1024))
top_cam.initialize()

# 11. Save Stage to USD File (including robot, track, lighting, and cameras)
stage.GetRootLayer().Export(out_usd_path)
print(f"[SUCCESS] Saved procedural arena stage to: {out_usd_path}")

# Also save math metadata (centerline, left, right, normals) for fast RL tensor lookups
npz_path = os.path.join(arena_dir, f"{base_name}_track_data.npz")
np.savez_compressed(
    npz_path,
    centerline=centerline,
    tangents=tangents,
    normals=normals,
    left_line=left_line,
    right_line=right_line,
    track_width=args.width,
)
print(f"[SUCCESS] Saved mathematical track metadata to: {npz_path}")

import omni.replicator.core as rep

# 12. Capture Camera Images Using Omniverse Replicator (Exact RTX Viewport Match)
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

# Capture Top-Down overview image showing the whole path from above
top_down_png = os.path.join(arena_dir, f"{base_name}_top_down.png")
capture_camera_with_replicator(top_cam_path, top_down_png, "Top-Down Path Overview", resolution=(1024, 1024))

# Also capture onboard and third-person chase views
capture_camera_with_replicator(front_cam_path, os.path.join(arena_dir, "wavy_onboard_view.png"), "Onboard Camera View (CNN Input)")
capture_camera_with_replicator(chase_cam_path, os.path.join(arena_dir, "wavy_chase_view.png"), "Chase Overview Camera View")

# 13. Interactive GUI loop if non-headless
if not args.headless:
    print("\n[INFO] Interactive GUI view active! Rotate/zoom in the viewport. Close window to exit.")
    while simulation_app.is_running():
        simulation_app.update()

simulation_app.close()
print("[INFO] Generation complete.")
