from .geometry import world_to_scene
from .meshing import extrude_polygon_mesh, merge_meshes
from .paths import default_weights_path, default_wheeled_json, ensure_repo_on_path, resolve_path
from .policy import infer_num_robots, prepare_env, resolve_num_robots
from .scene_config import load_scene_config

ensure_repo_on_path()
from src.env import MultiWheeled  # noqa: E402
from src.utils import load_wheeled_env  # noqa: E402

__all__ = [
    "MultiWheeled",
    "load_wheeled_env",
    "world_to_scene",
    "extrude_polygon_mesh",
    "merge_meshes",
    "load_scene_config",
    "default_wheeled_json",
    "default_weights_path",
    "resolve_path",
    "prepare_env",
    "infer_num_robots",
    "resolve_num_robots",
]
