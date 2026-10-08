import argparse
import os
import sys

# 1. Parse arguments BEFORE starting SimulationApp
parser = argparse.ArgumentParser(description="Load Leatherback USD and visualize camera output.")
parser.add_argument("--headless", action="store_true", default=False, help="Run headless (save image to file without GUI).")
parser.add_argument("--steps", type=int, default=50, help="Number of warmup simulation steps to render before capturing.")
parser.add_argument("--output", type=str, default="outputs/camera_output.png", help="Path to save captured camera image.")
args = parser.parse_args()

# 2. Start SimulationApp FIRST (Required by Omniverse Carbonite runtime)
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": args.headless})

import numpy as np
from PIL import Image
import omni.usd
from pxr import Usd, UsdGeom, UsdPhysics, UsdLux, Gf, Sdf
from isaacsim.core.utils.stage import get_current_stage, add_reference_to_stage
import isaacsim.core.utils.prims as prim_utils
from isaacsim.sensors.camera import Camera

print("[INFO] Creating environment with ground and two path tracking lines...")

# 3. Create or access current stage
stage = get_current_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)

# Create /World root prim
world_prim = stage.DefinePrim("/World", "Xform")
stage.SetDefaultPrim(world_prim)

# 4. Add Lighting
dome_light = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
dome_light.CreateIntensityAttr(2500.0)

distant_light = UsdLux.DistantLight.Define(stage, "/World/SunLight")
distant_light.CreateIntensityAttr(2000.0)
distant_light.AddRotateXYZOp().Set(Gf.Vec3f(-45.0, 30.0, 0.0))

from pxr import UsdShade

# 5. Helper to create UsdPreviewSurface material
def create_pbr_material(stage, path, color_rgb, emissive_rgb=(0, 0, 0), roughness=0.6):
    mat = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color_rgb))
    shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*emissive_rgb))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return mat

ground_mat = create_pbr_material(stage, "/World/Looks/GroundMaterial", [0.08, 0.08, 0.08], roughness=0.8)
line_mat = create_pbr_material(stage, "/World/Looks/LineMaterial", [1.0, 1.0, 1.0], emissive_rgb=[0.8, 0.8, 0.8], roughness=0.3)

# Add Ground Plane (Dark Tarmac)
ground_path = "/World/Ground"
ground_geom = UsdGeom.Plane.Define(stage, ground_path)
ground_geom.CreateAxisAttr("Z")
ground_geom.CreateWidthAttr(50.0)
ground_geom.CreateLengthAttr(50.0)
UsdShade.MaterialBindingAPI.Apply(ground_geom.GetPrim()).Bind(ground_mat)

# 6. Add Two Boundary Lines Drawn On Ground (Corridor Width = 1.1 m, Track Length = 30 m)
left_line_geom = UsdGeom.Cube.Define(stage, "/World/LeftBoundaryLine")
left_line_geom.CreateSizeAttr(1.0)
left_line_geom.AddTranslateOp().Set(Gf.Vec3f(10.0, 0.55, 0.005))
left_line_geom.AddScaleOp().Set(Gf.Vec3f(30.0, 0.08, 0.005))
UsdShade.MaterialBindingAPI.Apply(left_line_geom.GetPrim()).Bind(line_mat)

right_line_geom = UsdGeom.Cube.Define(stage, "/World/RightBoundaryLine")
right_line_geom.CreateSizeAttr(1.0)
right_line_geom.AddTranslateOp().Set(Gf.Vec3f(10.0, -0.55, 0.005))
right_line_geom.AddScaleOp().Set(Gf.Vec3f(30.0, 0.08, 0.005))
UsdShade.MaterialBindingAPI.Apply(right_line_geom.GetPrim()).Bind(line_mat)

# 7. Add Leatherback Robot Reference
robot_usd_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../assets/leatherback.usd"))
if not os.path.exists(robot_usd_path):
    print(f"[ERROR] Could not find leatherback USD at: {robot_usd_path}")
    simulation_app.close()
    sys.exit(1)

print(f"[INFO] Referencing Leatherback USD from: {robot_usd_path}")
robot_prim_path = "/World/Leatherback"
add_reference_to_stage(usd_path=robot_usd_path, prim_path=robot_prim_path)

# Ensure robot is placed at origin on ground between the two lines
trans_attr = stage.GetPrimAtPath(robot_prim_path).GetAttribute("xformOp:translate")
if trans_attr.IsValid():
    trans_attr.Set(Gf.Vec3f(0.0, 0.0, 0.03))
else:
    robot_xform = UsdGeom.Xformable(stage.GetPrimAtPath(robot_prim_path))
    robot_xform.AddTranslateOp().Set(Gf.Vec3f(0.0, 0.0, 0.03))

# 8. Initialize Cameras: Onboard Camera_Left and Third-Person Camera_Chase
onboard_cam_path = f"{robot_prim_path}/Rigid_Bodies/Chassis/Camera_Left"
chase_cam_path = f"{robot_prim_path}/Rigid_Bodies/Chassis/Camera_Chase"

print(f"[INFO] Initializing Onboard Camera: {onboard_cam_path}")
onboard_camera = Camera(
    prim_path=onboard_cam_path,
    frequency=30,
    resolution=(640, 480),
)
onboard_camera.initialize()

print(f"[INFO] Initializing Chase Camera: {chase_cam_path}")
chase_camera = Camera(
    prim_path=chase_cam_path,
    frequency=30,
    resolution=(640, 480),
)
chase_camera.initialize()

import omni.timeline

timeline = omni.timeline.get_timeline_interface()
timeline.play()

# 9. Step simulation to warm up RTX rendering
print("[INFO] Warming up RTX renderer and capturing frames...")
for step_idx in range(args.steps):
    simulation_app.update()
    if step_idx % 10 == 0:
        print(f"  Step {step_idx}/{args.steps}...")

for _ in range(10):
    simulation_app.update()

def save_cam_image(camera_sensor, file_path, name):
    rgba = camera_sensor.get_rgba()
    if rgba is None or rgba.size == 0:
        rgba = camera_sensor.get_rgb()
    if rgba is not None and rgba.size > 0:
        rgb = rgba[:, :, :3]
        if rgb.dtype != np.uint8:
            rgb = (rgb * 255).astype(np.uint8) if rgb.max() <= 1.0 else rgb.astype(np.uint8)
        img = Image.fromarray(rgb)
        img.save(file_path)
        print(f"[SUCCESS] {name} saved to: {file_path} (Size: {img.size})")
    else:
        print(f"[WARNING] {name} returned empty buffer.")

save_cam_image(onboard_camera, os.path.abspath(args.output), "Onboard Camera (Camera_Left)")
chase_output = os.path.abspath(os.path.join(os.path.dirname(args.output), "chase_view.png"))
save_cam_image(chase_camera, chase_output, "Chase View Camera")

# 11. Interactive GUI Loop if running non-headless
if not args.headless:
    print("\n[INFO] Running interactive viewport. Close window or press Ctrl+C in terminal to exit.")
    while simulation_app.is_running():
        simulation_app.update()

simulation_app.close()
print("[INFO] Done.")
