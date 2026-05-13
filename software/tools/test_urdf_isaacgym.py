#!/usr/bin/env python3
"""
test_urdf_isaacgym.py — Quick sanity check that the generated URDF loads in IsaacGym.

Spawns a single Asimov v1 robot in an empty scene, applies zero torques, and
lets it fall under gravity for a few seconds. If it loads without errors and
the viewer shows a humanoid, the URDF is structurally valid.

Usage:
  python software/tools/test_urdf_isaacgym.py
"""

import os
import sys

try:
    from isaacgym import gymapi, gymutil
except ImportError:
    sys.exit("IsaacGym not available. Install isaacgymenvs or point PYTHONPATH to IsaacGym.")

import numpy as np


def main():
    # Parse args
    args = gymutil.parse_arguments(
        description="Test Asimov v1 URDF in IsaacGym",
        headless=False,
    )

    # Init gym
    gym = gymapi.acquire_gym()

    # Sim params
    sim_params = gymapi.SimParams()
    sim_params.dt = 1.0 / 60.0
    sim_params.substeps = 2
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)

    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 4
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.contact_offset = 0.01
    sim_params.physx.rest_offset = 0.0
    sim_params.physx.use_gpu = args.use_gpu

    sim = gym.create_sim(
        args.compute_device_id, args.graphics_device_id, gymapi.SIM_PHYSX, sim_params
    )
    if sim is None:
        print("Failed to create sim")
        return

    # Ground plane
    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0, 0, 1)
    gym.add_ground(sim, plane_params)

    # Load URDF
    asset_root = os.path.join(os.path.dirname(__file__), "..", "resources", "robots")
    asset_file = "asimov_v1/urdf/asimov_v1.urdf"

    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = False
    asset_options.default_dof_drive_mode = gymapi.DOF_MODE_EFFORT
    asset_options.collapse_fixed_joints = False
    asset_options.flip_visual_attachments = False
    asset_options.armature = 0.01  # small default armature

    print(f"Loading asset: {asset_root}/{asset_file}")
    robot_asset = gym.load_asset(sim, asset_root, asset_file, asset_options)
    if robot_asset is None:
        print("Failed to load asset")
        return

    # Print asset info
    num_dofs = gym.get_asset_dof_count(robot_asset)
    num_bodies = gym.get_asset_rigid_body_count(robot_asset)
    print(f"Asset loaded: {num_bodies} bodies, {num_dofs} DOFs")

    dof_names = [gym.get_asset_dof_name(robot_asset, i) for i in range(num_dofs)]
    print("DOF names:", dof_names)

    # Create env
    env_spacing = 2.0
    env_lower = gymapi.Vec3(-env_spacing, -env_spacing, 0.0)
    env_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)
    env = gym.create_env(sim, env_lower, env_upper, 1)

    # Spawn robot at standing height
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0.0, 0.0, 0.65)  # slightly above ground
    pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

    actor_handle = gym.create_actor(env, robot_asset, pose, "asimov", 0, 1)

    # Set initial DOF state to zero (all joints at zero angle)
    dof_states = np.zeros(num_dofs, dtype=gymapi.DofState.dtype)
    gym.set_actor_dof_states(env, actor_handle, dof_states, gymapi.STATE_ALL)

    # Viewer
    if not args.headless:
        cam_props = gymapi.CameraProperties()
        viewer = gym.create_viewer(sim, cam_props)
        if viewer is None:
            print("Failed to create viewer")
            return
        # Position camera
        cam_pos = gymapi.Vec3(2.5, 2.5, 1.5)
        cam_target = gymapi.Vec3(0, 0, 0.6)
        gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)
    else:
        viewer = None

    # Simulation loop
    print("Running simulation (robot will fall under gravity with zero torques)...")
    frame = 0
    max_frames = 600  # 10 seconds at 60 Hz

    while frame < max_frames:
        # Step physics
        gym.simulate(sim)
        gym.fetch_results(sim, True)

        # Step rendering
        if viewer:
            gym.step_graphics(sim)
            gym.draw_viewer(viewer, sim, True)
            gym.sync_frame_time(sim)

            # Check for exit
            if gym.query_viewer_has_closed(viewer):
                break

        frame += 1

    print(f"Simulation complete ({frame} frames).")

    if viewer:
        gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)


if __name__ == "__main__":
    main()
