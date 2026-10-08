import dataclasses
import math
from typing import List, Tuple, Optional
import numpy as np
from pxr import Usd, UsdGeom, UsdShade, UsdLux, Gf, Sdf


@dataclasses.dataclass
class TrackConfig:
    track_type: str = "wavy_circle"  # "wavy_circle", "s_curve", "circuit", "slalom", "random", "organic"
    num_corners: Optional[int] = None # Control points / corners count for circuit mode
    radius: float = 10.0             # Mean radius for wavy circle / circuit (m)
    waves: int = 4                   # Number of harmonic waves for wavy circle
    amplitude: float = 1.8           # Amplitude of waves (m)
    track_width: float = 1.2         # Distance between left and right boundary lines (m)
    line_thickness: float = 0.08     # Width of each boundary line stripe (m)
    length: float = 40.0             # Path length for open tracks (m)
    min_radius: float = 1.8          # Minimum turn radius (m)
    num_samples: int = 500           # Centerline discretization count
    ground_size: float = 80.0        # Ground plane size (m)
    line_color: Tuple[float, float, float] = (1.0, 0.82, 0.0)    # Racing yellow
    line_emissive: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    ground_color: Tuple[float, float, float] = (0.12, 0.12, 0.12) # Dark grey asphalt
    ground_roughness: float = 0.85
    sun_azimuth_deg: float = 40.0
    sun_elevation_deg: float = 45.0
    dome_intensity: float = 250.0
    sun_intensity: float = 800.0
    seed: Optional[int] = 42


class ProceduralTrack:
    """Generates procedural 2D paths and corresponding 3D USD mesh ribbons for path tracking."""

    def __init__(self, cfg: TrackConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)

        self.centerline: np.ndarray = np.empty((0, 2), dtype=np.float32)
        self.tangents: np.ndarray = np.empty((0, 2), dtype=np.float32)
        self.normals: np.ndarray = np.empty((0, 2), dtype=np.float32)
        self.left_line: np.ndarray = np.empty((0, 2), dtype=np.float32)
        self.right_line: np.ndarray = np.empty((0, 2), dtype=np.float32)
        self.is_closed: bool = False

        self._generate_path()

    def _generate_path(self):
        """Dispatches to the selected path generation strategy."""
        if self.cfg.track_type == "circuit":
            self._generate_circuit()
        elif self.cfg.track_type == "organic":
            self._generate_organic()
        elif self.cfg.track_type == "wavy_circle":
            self._generate_wavy_circle()
        elif self.cfg.track_type == "slalom":
            self._generate_slalom()
        elif self.cfg.track_type == "random":
            self._generate_random_spline()
        else:  # default s_curve
            self._generate_s_curves()

        self._compute_geometry()

    def _generate_wavy_circle(self):
        """Generates a smooth, closed wavy circle with harmonic oscillations."""
        self.is_closed = True
        theta = np.linspace(0.0, 2.0 * math.pi, self.cfg.num_samples, endpoint=False)

        n1 = self.cfg.waves
        a1 = self.cfg.amplitude
        phi1 = float(self.rng.uniform(0.0, 2.0 * math.pi))

        n2 = self.cfg.waves * 2
        a2 = self.cfg.amplitude * 0.3
        phi2 = float(self.rng.uniform(0.0, 2.0 * math.pi))

        r = self.cfg.radius + a1 * np.sin(n1 * theta + phi1) + a2 * np.sin(n2 * theta + phi2)
        x = r * np.cos(theta)
        y = r * np.sin(theta)

        self.centerline = np.stack([x, y], axis=1)

    def _generate_s_curves(self):
        """Generates a straight start followed by alternating smooth S-curves."""
        self.is_closed = False
        num_pts = self.cfg.num_samples
        s = np.linspace(0, self.cfg.length, num_pts)
        
        freq1 = self.rng.uniform(0.08, 0.14)
        amp1 = self.rng.uniform(1.2, 3.2)
        freq2 = self.rng.uniform(0.18, 0.28)
        amp2 = self.rng.uniform(0.4, 1.2)
        
        ramp = np.clip((s - 3.0) / 4.0, 0.0, 1.0)
        y = ramp * (amp1 * np.sin(freq1 * s) + amp2 * np.sin(freq2 * s))
        x = s
        
        self.centerline = np.stack([x, y], axis=1)

    def _generate_slalom(self):
        """Generates a slalom sequence with varying frequency and gate offsets."""
        self.is_closed = False
        num_pts = self.cfg.num_samples
        s = np.linspace(0, self.cfg.length, num_pts)
        
        num_gates = self.rng.integers(4, 8)
        omega = (2.0 * math.pi * num_gates) / self.cfg.length
        amp = self.rng.uniform(1.0, 2.5)
        
        ramp = np.clip((s - 3.0) / 3.0, 0.0, 1.0)
        y = ramp * amp * np.sin(omega * s)
        x = s
        
        self.centerline = np.stack([x, y], axis=1)

    def _generate_circuit(self):
        """Generates an authentic asymmetric motorsport racetrack with variable straights, hairpins, and chicanes."""
        self.is_closed = True
        if self.cfg.num_corners is not None:
            num_ctrl = max(6, self.cfg.num_corners)
        else:
            num_ctrl = self.rng.integers(10, 16)
        base_angles = np.linspace(0, 2 * math.pi, num_ctrl, endpoint=False)
        angle_jitter = self.rng.uniform(-0.35, 0.35, size=num_ctrl) * (2 * math.pi / num_ctrl)
        angles = np.sort(base_angles + angle_jitter)

        base_r = self.cfg.radius
        radii = self.rng.uniform(base_r * 0.7, base_r * 1.35, size=num_ctrl)

        aspect = float(self.rng.uniform(1.15, 1.45))
        rot = float(self.rng.uniform(0.0, math.pi))
        c, s = math.cos(rot), math.sin(rot)

        x_raw = radii * np.cos(angles)
        y_raw = radii * np.sin(angles)
        x_rot = (c * x_raw + s * y_raw) * aspect
        y_rot = -s * x_raw + c * y_raw
        x_ctrl = c * x_rot - s * y_rot
        y_ctrl = s * x_rot + c * y_rot

        from scipy.interpolate import splprep, splev
        ctrl_pts = [np.append(x_ctrl, x_ctrl[0]), np.append(y_ctrl, y_ctrl[0])]
        tck, _ = splprep(ctrl_pts, s=0, per=True, k=3)
        u_fine = np.linspace(0, 1, self.cfg.num_samples, endpoint=False)
        x_fine, y_fine = splev(u_fine, tck)

        self.centerline = np.stack([x_fine, y_fine], axis=1)

    def _generate_organic(self):
        """Generates an organic flowing loop using multi-octave non-repeating Fourier harmonics."""
        self.is_closed = True
        theta = np.linspace(0.0, 2.0 * math.pi, self.cfg.num_samples, endpoint=False)
        r = np.full_like(theta, self.cfg.radius)
        for k in range(1, 9):
            weight = self.cfg.amplitude * (1.0 / (k ** 0.85))
            a = self.rng.uniform(0.3 * weight, 1.2 * weight)
            phi = self.rng.uniform(0.0, 2.0 * math.pi)
            r += a * np.sin(k * theta + phi)

        x = r * np.cos(theta)
        y = r * np.sin(theta)
        self.centerline = np.stack([x, y], axis=1)

    def _generate_random_spline(self):
        """Generates an organic random spline with randomized waypoint offsets."""
        self.is_closed = False
        num_wp = self.rng.integers(7, 12)
        s_wp = np.linspace(0, self.cfg.length, num_wp)
        
        y_wp = np.zeros(num_wp)
        for i in range(2, num_wp):
            max_dev = min(4.0, 0.4 * s_wp[i])
            y_wp[i] = y_wp[i - 1] + self.rng.uniform(-1.8, 1.8)
            y_wp[i] = np.clip(y_wp[i], -max_dev, max_dev)
            
        from scipy.interpolate import CubicSpline
        cs = CubicSpline(s_wp, y_wp)
        s_fine = np.linspace(0, self.cfg.length, self.cfg.num_samples)
        y_fine = cs(s_fine)
        
        self.centerline = np.stack([s_fine, y_fine], axis=1)

    def _compute_geometry(self):
        """Calculates tangent vectors, unit normals, and left/right boundary lines."""
        if self.is_closed:
            # Finite differences with periodic circular wrap
            diff = np.roll(self.centerline, -1, axis=0) - np.roll(self.centerline, 1, axis=0)
        else:
            diff = np.diff(self.centerline, axis=0, append=[self.centerline[-1]])

        lengths = np.linalg.norm(diff, axis=1, keepdims=True)
        lengths[lengths < 1e-6] = 1e-6
        self.tangents = diff / lengths
        
        # Unit normal pointing left: (-ty, tx)
        self.normals = np.stack([-self.tangents[:, 1], self.tangents[:, 0]], axis=1)
        
        half_w = self.cfg.track_width / 2.0
        self.left_line = self.centerline + half_w * self.normals
        self.right_line = self.centerline - half_w * self.normals

    def get_start_pose(self) -> Tuple[np.ndarray, float]:
        """Returns the recommended spawn position (x, y, z) and yaw (rad) for the robot."""
        x0, y0 = self.centerline[0]
        tx, ty = self.tangents[0]
        yaw0 = math.atan2(ty, tx)
        return np.array([x0, y0, 0.03], dtype=np.float32), yaw0

    def build_usd_meshes(self, stage: Usd.Stage, root_path: str = "/World/Track"):
        """Creates 3D ribbon meshes for the left and right boundary lines in the USD stage."""
        track_prim = stage.DefinePrim(root_path, "Xform")
        
        line_mat_path = f"{root_path}/LineMaterial"
        line_mat = self._create_pbr_material(
            stage, line_mat_path,
            color=self.cfg.line_color,
            emissive=self.cfg.line_emissive,
            roughness=0.3
        )
        
        self._create_ribbon_mesh(
            stage, f"{root_path}/LeftBoundaryLine",
            line_center=self.left_line,
            normals=self.normals,
            material=line_mat
        )
        
        self._create_ribbon_mesh(
            stage, f"{root_path}/RightBoundaryLine",
            line_center=self.right_line,
            normals=self.normals,
            material=line_mat,
        )

    def get_track_bounds(self) -> Tuple[float, float, float, float]:
        """Returns (x_min, x_max, y_min, y_max) covering both boundary lines."""
        all_pts = np.vstack([self.left_line, self.right_line])
        return (
            float(np.min(all_pts[:, 0])),
            float(np.max(all_pts[:, 0])),
            float(np.min(all_pts[:, 1])),
            float(np.max(all_pts[:, 1])),
        )

    def build_top_down_camera(
        self,
        stage: Usd.Stage,
        camera_path: str = "/World/TopDownCamera",
        margin_factor: float = 1.25,
        focal_length: float = 24.0,
        aperture: float = 20.955,
    ) -> Tuple[str, float]:
        """Creates a top-down camera positioned above the track center framing the whole path."""
        x_min, x_max, y_min, y_max = self.get_track_bounds()
        cx = (x_min + x_max) / 2.0
        cy = (y_min + y_max) / 2.0
        span = max(x_max - x_min, y_max - y_min) * margin_factor
        height = max((span / 2.0) * (2.0 * focal_length / aperture), 10.0)

        cam_prim = stage.DefinePrim(camera_path, "Camera")
        cam_geom = UsdGeom.Camera(cam_prim)
        cam_geom.CreateFocalLengthAttr(focal_length)
        cam_geom.CreateHorizontalApertureAttr(aperture)
        cam_geom.CreateVerticalApertureAttr(aperture)
        cam_geom.CreateClippingRangeAttr(Gf.Vec2f(0.1, 10000.0))

        xform = UsdGeom.Xformable(cam_prim)
        xform.AddTranslateOp().Set(Gf.Vec3f(cx, cy, height))
        xform.AddOrientOp().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        return camera_path, height

    def _create_ribbon_mesh(self, stage: Usd.Stage, mesh_path: str, line_center: np.ndarray, normals: np.ndarray, material: UsdShade.Material):
        """Constructs a UsdGeom.Mesh ribbon along the given path points."""
        mesh = UsdGeom.Mesh.Define(stage, mesh_path)
        half_thick = self.cfg.line_thickness / 2.0
        z_height = 0.003  # Slightly elevated above ground to prevent z-fighting
        
        n_pts = len(line_center)
        v_left = line_center + half_thick * normals
        v_right = line_center - half_thick * normals
        
        points = []
        for i in range(n_pts):
            points.append(Gf.Vec3f(float(v_left[i, 0]), float(v_left[i, 1]), z_height))
            points.append(Gf.Vec3f(float(v_right[i, 0]), float(v_right[i, 1]), z_height))
            
        num_segments = n_pts if self.is_closed else (n_pts - 1)
        face_vertex_counts = [4] * num_segments
        face_vertex_indices = []
        
        for i in range(num_segments):
            idx0 = 2 * i
            idx1 = 2 * i + 1
            next_i = (i + 1) % n_pts if self.is_closed else (i + 1)
            idx2 = 2 * next_i + 1
            idx3 = 2 * next_i
            face_vertex_indices.extend([idx0, idx1, idx2, idx3])
            
        mesh.CreatePointsAttr(points)
        mesh.CreateFaceVertexCountsAttr(face_vertex_counts)
        mesh.CreateFaceVertexIndicesAttr(face_vertex_indices)
        mesh.CreateNormalsAttr([Gf.Vec3f(0.0, 0.0, 1.0)] * len(points))
        mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
        mesh.CreateDoubleSidedAttr(True)
        mesh.CreateDisplayColorAttr([Gf.Vec3f(*self.cfg.line_color)])
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)

    def build_ground_plane(self, stage: Usd.Stage, ground_path: str = "/World/Ground"):
        """Constructs the dark asphalt ground plane with randomized PBR properties."""
        mat_path = "/World/Looks/GroundMaterial"
        ground_mat = self._create_pbr_material(
            stage, mat_path,
            color=self.cfg.ground_color,
            emissive=(0.0, 0.0, 0.0),
            roughness=self.cfg.ground_roughness
        )
        
        ground_geom = UsdGeom.Plane.Define(stage, ground_path)
        ground_geom.CreateAxisAttr("Z")
        ground_geom.CreateWidthAttr(self.cfg.ground_size)
        ground_geom.CreateLengthAttr(self.cfg.ground_size)
        ground_geom.CreateDisplayColorAttr([Gf.Vec3f(*self.cfg.ground_color)])
        UsdShade.MaterialBindingAPI.Apply(ground_geom.GetPrim()).Bind(ground_mat)

    def build_lighting(self, stage: Usd.Stage):
        """Constructs balanced lighting (DomeLight + Distant SunLight)."""
        dome_light = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
        dome_light.CreateIntensityAttr(self.cfg.dome_intensity)
        dome_light.CreateColorAttr(Gf.Vec3f(0.8, 0.85, 0.95))
        
        sun_light = UsdLux.DistantLight.Define(stage, "/World/SunLight")
        sun_light.CreateIntensityAttr(self.cfg.sun_intensity)
        sun_light.CreateColorAttr(Gf.Vec3f(1.0, 0.98, 0.92))
        sun_light.AddRotateXYZOp().Set(Gf.Vec3f(
            -self.cfg.sun_elevation_deg,
            self.cfg.sun_azimuth_deg,
            0.0
        ))

    def _create_pbr_material(self, stage: Usd.Stage, path: str, color, emissive, roughness):
        mat = UsdShade.Material.Define(stage, path)
        shader = UsdShade.Shader.Define(stage, f"{path}/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*emissive))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
        mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        return mat
