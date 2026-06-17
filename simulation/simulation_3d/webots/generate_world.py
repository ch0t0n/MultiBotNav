"""Generate a Webots agricultural world from wheeled_configs.json."""

from __future__ import annotations

import argparse
import json
import os
import sys

WEBOTS_ROOT = os.path.dirname(os.path.abspath(__file__))
SIM_ROOT = os.path.dirname(WEBOTS_ROOT)
if SIM_ROOT not in sys.path:
    sys.path.insert(0, SIM_ROOT)

from core.env_loader import load_env_from_json  # noqa: E402
from core.geometry import world_to_scene  # noqa: E402
from core.paths import default_wheeled_json  # noqa: E402

ROBOT_COLORS = [
    "0.15 0.45 0.85",
    "0.85 0.35 0.15",
    "0.20 0.70 0.30",
    "0.75 0.20 0.55",
    "0.55 0.55 0.15",
]
OBSTACLE_STYLES = [
    ("0.55 0.35 0.15", 12.0),
    ("0.72 0.58 0.22", 8.0),
    ("0.18 0.42 0.12", 14.0),
]



def _viewpoint_lines(max_dim: float) -> list[str]:
    """
    Near top-down camera centered on the field.

    The previous orientation pointed at the sky (blue screen only). Users can
    still rotate/pan/zoom freely in the Webots 3D view after load.
    """
    height = max_dim * 1.3
    return [
        "Viewpoint {",
        f"  position 0.0 {height:.1f} 0.0",
        "  orientation 1 0 0 -1.5707963267948966",
        "  fieldOfView 1.05",
        "}",
    ]


def _pbr(color: str, transparency: float | None = None, roughness: float = 0.8) -> str:
    """PBRAppearance is valid directly on Shape in Webots R2023b."""
    lines = [
        "PBRAppearance {",
        f"  baseColor {color}",
        f"  roughness {roughness}",
        "  metalness 0",
    ]
    if transparency is not None:
        lines.append(f"  transparency {transparency}")
    lines.append("}")
    return "\n      ".join(lines)


def _field_ground(w: float, h: float) -> str:
    """Single horizontal ground slab — no overlapping crop-row geometry."""
    pbr = _pbr("0.32 0.48 0.18", roughness=1.0)
    thickness = 0.2
    return f"""DEF FIELD_GROUND Solid {{
  translation 0.000 {-thickness / 2:.3f} 0.000
  children [
    Shape {{
      geometry Box {{
        size {w:.3f} {thickness:.3f} {h:.3f}
      }}
      appearance {pbr}
    }}
  ]
  physics NULL
  name "FIELD_GROUND"
}}"""


def _polygon_obstacle(
    name: str,
    poly: list,
    world_w: float,
    world_h: float,
    height: float,
    color: str,
    def_name: str | None = None,
    base_y: float = 0.05,
) -> str:
    """Extruded polygon footprint matching the JSON obstacle definition."""
    n = len(poly)
    if n < 3:
        return ""

    bottom = []
    top = []
    for vx, vy in poly:
        px, _, pz = world_to_scene(float(vx), float(vy), world_w, world_h)
        bottom.append((px, pz))
        top.append((px, pz))

    points: list[str] = []
    for px, pz in bottom:
        points.append(f"{px:.4f} {base_y:.4f} {pz:.4f}")
    for px, pz in top:
        points.append(f"{px:.4f} {base_y + height:.4f} {pz:.4f}")

    top_face = " ".join(str(i) for i in range(n)) + " -1"
    bottom_face = " ".join(str(2 * n - 1 - i) for i in range(n)) + " -1"
    side_faces = []
    for i in range(n):
        j = (i + 1) % n
        side_faces.append(f"{i} {j} {j + n} {i + n} -1")

    coord_index = ",\n            ".join([top_face, bottom_face, *side_faces])
    points_str = ",\n            ".join(points)
    prefix = f"DEF {def_name} " if def_name else ""
    pbr = _pbr(color)

    return f"""{prefix}Solid {{
  children [
    Shape {{
      geometry IndexedFaceSet {{
        coord Coordinate {{
          point [
            {points_str}
          ]
        }}
        coordIndex [
          {coord_index}
        ]
      }}
      appearance {pbr}
    }}
  ]
  physics NULL
  name "{name}"
}}"""


def _solid_box(name: str, px: float, py: float, pz: float,
               sx: float, sy: float, sz: float, color: str,
               def_name: str | None = None, static: bool = True):
    prefix = f"DEF {def_name} " if def_name else ""
    pbr = _pbr(color)
    physics = "  physics NULL\n" if static else ""
    return f"""{prefix}Solid {{
  translation {px:.3f} {py:.3f} {pz:.3f}
  children [
    Shape {{
      geometry Box {{ size {sx:.3f} {sy:.3f} {sz:.3f} }}
      appearance {pbr}
    }}
  ]
{physics}  name "{name}"
}}"""


def _robot_solid(idx: int, px: float, pz: float, theta: float,
                 length: float, width: float, color: str):
    body_pbr = _pbr(color, roughness=0.6)
    cab_pbr = _pbr("0.12 0.12 0.12")
    return f"""DEF ROBOT_{idx} Solid {{
  translation {px:.3f} 2.250 {pz:.3f}
  rotation 0 1 0 {-theta:.6f}
  children [
    Shape {{
      geometry Box {{ size {length:.3f} 4.500 {width:.3f} }}
      appearance {body_pbr}
    }}
    Transform {{
      translation {length * 0.22:.3f} 2.800 0.000
      children [
        Shape {{
          geometry Box {{ size {length * 0.35:.3f} 4.000 {width * 0.75:.3f} }}
          appearance {cab_pbr}
        }}
      ]
    }}
  ]
  physics NULL
  name "ROBOT_{idx}"
}}"""


def _goal_solid(idx: int, px: float, pz: float, radius: float):
    # Opaque geometry — transparency causes depth-sort flicker when orbiting camera.
    zone_pbr = _pbr("0.15 0.75 0.25", roughness=0.5)
    flag_pbr = _pbr("0.90 0.85 0.20", roughness=0.4)
    return f"""DEF GOAL_{idx} Solid {{
  translation {px:.3f} 0.150 {pz:.3f}
  children [
    Shape {{
      geometry Cylinder {{ radius {radius:.3f} height 0.250 }}
      appearance {zone_pbr}
    }}
    Transform {{
      translation {radius * 0.5:.3f} 6.000 0.000
      scale 1 1 1
      children [
        Shape {{
          geometry Box {{ size 1.200 10.000 1.200 }}
          appearance {flag_pbr}
        }}
      ]
    }}
  ]
  physics NULL
  name "GOAL_{idx}"
}}"""


def generate_world(env_key: str, json_path: str | None = None) -> str:
    json_path = json_path or default_wheeled_json()
    params = load_env_from_json(json_path, key=env_key)
    w = params["SCREEN_WIDTH"]
    h = params["SCREEN_HEIGHT"]
    max_dim = max(w, h)
    rlen = params["ROBOT_LENGTH"]
    rwid = params["ROBOT_WIDTH"]

    lines = [
        "#VRML_SIM R2023b utf8",
        "# MultiBotNav agricultural field — generated from wheeled_configs.json",
        f"# env_key: {env_key}",
        "",
        "WorldInfo {",
        '  title "MultiBotNav Agricultural Field"',
        "  basicTimeStep 50",
        '  coordinateSystem "ENU"',
        "}",
        *_viewpoint_lines(max_dim),
        "Background {",
        "  skyColor [0.45 0.65 0.95]",
        "}",
        "DirectionalLight {",
        "  direction 0.35 -1.0 0.25",
        "  intensity 1.8",
        "  castShadows FALSE",
        "}",
        "",
        _field_ground(w, h),
    ]

    fence_h = 6.0
    fence_base = 0.05
    for name, sx, sy, sz, px, py, pz in [
        ("FENCE_N", w, fence_h, 2, 0, fence_base + fence_h / 2, h / 2),
        ("FENCE_S", w, fence_h, 2, 0, fence_base + fence_h / 2, -h / 2),
        ("FENCE_E", 2, fence_h, h, w / 2, fence_base + fence_h / 2, 0),
        ("FENCE_W", 2, fence_h, h, -w / 2, fence_base + fence_h / 2, 0),
    ]:
        lines.append(_solid_box(name, px, py, pz, sx, sy, sz, "0.55 0.38 0.20"))

    for idx, poly in enumerate(params["OBSTACLES"]):
        color, height = OBSTACLE_STYLES[idx % len(OBSTACLE_STYLES)]
        mesh = _polygon_obstacle(
            f"OBSTACLE_{idx}",
            poly,
            w,
            h,
            height,
            color,
            def_name=f"OBSTACLE_{idx}",
        )
        if mesh:
            lines.append(mesh)

    for idx, (gx, gy) in enumerate(params["GOAL_POSITIONS"]):
        px, _, pz = world_to_scene(gx, gy, w, h)
        radius = max(3.0, params["GOAL_SIZE"])
        lines.append(_goal_solid(idx, px, pz, radius))

    init_configs = params["ROBOT_INIT_CONFIGS"]
    for i, (x, y, theta) in enumerate(init_configs):
        px, _, pz = world_to_scene(x, y, w, h)
        color = ROBOT_COLORS[i % len(ROBOT_COLORS)]
        lines.append(_robot_solid(i, px, pz, theta, rlen, rwid, color))

    wind_pbr = _pbr("0.7 0.8 0.95")
    sup_pbr = _pbr("0.9 0.2 0.2")
    lines += [
        "",
        "DEF WIND_ARROW Solid {",
        f"  translation {w / 2 - 30:.1f} 8.0 {-h / 2 + 20:.1f}",
        "  rotation 0 1 0 0",
        "  children [",
        "    Shape {",
        "      geometry Box { size 20 0.5 2 }",
        f"      appearance {wind_pbr}",
        "    }",
        "  ]",
        "  physics NULL",
        '  name "WIND_ARROW"',
        "}",
        "",
        "Robot {",
        "  translation 0 0.5 0",
        '  controller "wheeled_nav"',
        "  supervisor TRUE",
        "  synchronization TRUE",
        "  children [",
        "    Shape {",
        "      geometry Sphere { radius 0.3 }",
        f"      appearance {sup_pbr}",
        "    }",
        "  ]",
        '  name "MULTIBOT_SUPERVISOR"',
        "}",
        "",
    ]
    return "\n".join(lines)


def write_sim_config(
    env_key: str,
    config_path: str,
    num_robots: int | None = None,
    weights_path: str | None = None,
    random_policy: bool = False,
    max_steps: int = 1000,
    json_path: str | None = None,
):
    payload = {
        "env_key": env_key,
        "num_robots": num_robots,
        "weights_path": weights_path,
        "random_policy": random_policy,
        "max_steps": max_steps,
        "json_path": json_path or default_wheeled_json(),
    }
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def generate_world_file(
    env_key: str,
    json_path: str | None = None,
    num_robots: int | None = None,
    weights_path: str | None = None,
    random_policy: bool = False,
    max_steps: int = 1000,
) -> str:
    content = generate_world(env_key, json_path)
    out_dir = os.path.join(WEBOTS_ROOT, "worlds")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"ag_field_{env_key}.wbt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(content)

    config_path = os.path.join(
        WEBOTS_ROOT, "controllers", "wheeled_nav", "sim_config.json"
    )
    write_sim_config(
        env_key=env_key,
        config_path=config_path,
        num_robots=num_robots,
        weights_path=weights_path,
        random_policy=random_policy,
        max_steps=max_steps,
        json_path=json_path,
    )
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Generate Webots agricultural world.")
    parser.add_argument("--env-key", default="env1")
    parser.add_argument("--json", default=None)
    parser.add_argument("--num-robots", type=int, default=None)
    parser.add_argument("--weights", default=None)
    parser.add_argument("--random-policy", action="store_true")
    parser.add_argument("--max-steps", type=int, default=1000)
    args = parser.parse_args()

    out_path = generate_world_file(
        env_key=args.env_key,
        json_path=args.json,
        num_robots=args.num_robots,
        weights_path=args.weights,
        random_policy=args.random_policy,
        max_steps=args.max_steps,
    )
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
