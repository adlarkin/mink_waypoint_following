import argparse
import json
import mink
import mujoco
import mujoco.viewer
import numpy as np
import time

from pathlib import Path


# Model files (MJCF)
_HERE = Path(__file__).parent
_PANDA_XML = _HERE / "franka_emika_panda" / "scene.xml"
_WAM_XML = _HERE / "barrett" / "wam_7dof_wam_bhand.xml"
# Robot types to their model files
_ROBOTS = {
    'panda': _PANDA_XML,
    'wam': _WAM_XML,
}
_VALID_ROBOT_TYPES = list(_ROBOTS.keys())


# Function to add a coordinate frame to the scene.
#
# origin is a vector in R^3 that gives the (x,y,z) position of the frame origin.
# rot_mat is a 3x3 rotation matrix that specifies the frame orientation.
# Both origin and rot_mat are assumed to be defined in the world frame.
def add_frame(scene: mujoco.MjvScene, origin, rot_mat, axis_radius=0.0075, axis_length=.0525):
    # Axis colors (RGBA): X (red), Y (green), Z (blue)
    colors = {
        'x': [1.0, 0.0, 0.0, 1.0],
        'y': [0.0, 1.0, 0.0, 1.0],
        'z': [0.0, 0.0, 1.0, 1.0]
    }
    # Define axis directions
    axes = {
        'x': np.array([1.0, 0.0, 0.0]),
        'y': np.array([0.0, 1.0, 0.0]),
        'z': np.array([0.0, 0.0, 1.0])
    }

    for axis, color in colors.items():
        # Rotate axis direction by the orientation
        direction = rot_mat @ axes[axis]
        # Set the end point of the axis marker.
        end_point = origin + axis_length * direction

        assert scene.ngeom < scene.maxgeom
        scene.ngeom += 1
        mujoco.mjv_initGeom(
            scene.geoms[scene.ngeom - 1],
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=np.zeros(3),
            pos=np.zeros(3),
            mat=np.eye(3).flatten(),
            rgba=np.array(color),
        )
        mujoco.mjv_connector(
            scene.geoms[scene.ngeom - 1],
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            width=axis_radius,
            from_=origin,
            to=end_point,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("robot", type=str, help=f"The robot to use. Must be one of {_VALID_ROBOT_TYPES}")
    args = parser.parse_args()
    if args.robot not in _VALID_ROBOT_TYPES:
        raise ValueError(f"The given robot type {args.robot} is invalid, must be one of {_VALID_ROBOT_TYPES}")

    model = mujoco.MjModel.from_xml_path(_ROBOTS[args.robot].as_posix())
    data = mujoco.MjData(model)

    configuration = mink.Configuration(model)

    tasks = [
        end_effector_task := mink.FrameTask(
            frame_name="ee_site",
            frame_type="site",
            position_cost=1.0,
            orientation_cost=1.0,
            lm_damping=1.0,
        ),
    ]

    if args.robot == 'panda':
        hand_geoms = mink.get_subtree_geom_ids(model, model.body("hand").id)
        max_velocities = {
            'joint1': np.pi,
            'joint2': np.pi,
            'joint3': np.pi,
            'joint4': np.pi,
            'joint5': np.pi,
            'joint6': np.pi,
            'joint7': np.pi,
        }
    elif args.robot == 'wam':
        hand_geoms = mink.get_subtree_geom_ids(model, model.body("wam/bhand/bhand_palm_link").id)
        max_velocities = {
            'wam/base_yaw_joint'      : np.pi,
            'wam/shoulder_pitch_joint': np.pi,
            'wam/shoulder_yaw_joint'  : np.pi,
            'wam/elbow_pitch_joint'   : np.pi,
            'wam/wrist_pitch_joint'   : np.pi,
            'wam/wrist_yaw_joint'     : np.pi,
            'wam/palm_yaw_joint'      : np.pi,
        }
    # Enable collision avoidance between the following geoms:
    collision_pairs = [
        (hand_geoms, ["floor"]),
        # TODO: add collision avoidance between the hand and rest of the robot?
    ]
    limits = [
        mink.ConfigurationLimit(model=model),
        mink.CollisionAvoidanceLimit(model=model, geom_pairs=collision_pairs),
        mink.VelocityLimit(model, max_velocities),
    ]

    # Use the "home" keyframe as the initial joint configuration
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    mujoco.mj_kinematics(model, data)
    configuration.update(data.qpos)

    # Define a few waypoints to follow (simple rectangular motion)
    ee_pose_init = configuration.get_transform_frame_to_world(
        frame_name="ee_site",
        frame_type="site",
    )
    side_length = .185
    waypoint_1 = mink.lie.SE3.from_rotation_and_translation(
        ee_pose_init.rotation(),
        ee_pose_init.translation() + [-side_length, 0.5 * side_length, 0],
    )
    waypoint_2 = mink.lie.SE3.from_rotation_and_translation(
        waypoint_1.rotation(),
        waypoint_1.translation() + [side_length, 0.0, 0],
    )
    waypoint_3 = mink.lie.SE3.from_rotation_and_translation(
        waypoint_2.rotation(),
        waypoint_2.translation() + [0.0, -side_length, 0],
    )
    waypoint_4 = mink.lie.SE3.from_rotation_and_translation(
        waypoint_3.rotation(),
        waypoint_3.translation() + [-side_length, 0.0, 0],
    )
    path = [
        waypoint_1,
        waypoint_2,
        waypoint_3,
        waypoint_4,
        waypoint_1,
        waypoint_2,
        waypoint_3,
        waypoint_4,
    ]

    # List of joint configurations at each waypoint for the rectangular motion (computed by IK).
    # Must convert from numpy array to a list so that this data can be saved to a json file.
    joint_configs = [configuration.q.tolist()]

    with mujoco.viewer.launch_passive(model=model, data=data, show_left_ui=False, show_right_ui=False) as viewer:
        # Update the viewer's orientation to capture the arm movement.
        viewer.cam.lookat = [0, 0, 0.35]
        viewer.cam.distance = 2.5
        viewer.cam.azimuth = 145
        viewer.cam.elevation = -25

        # Visualize the waypoint frames (for debugging)
        for w in path:
            add_frame(viewer.user_scn, w.translation(), w.rotation().as_matrix())

        # Show the start configuration
        viewer.sync()
        time.sleep(1)

        curr_waypoint_idx = 0
        while viewer.is_running() and curr_waypoint_idx < len(path):
            # Run at 100hz.
            # NOTE: this is for visualization purposes.
            # If visualization is not important, the rate limit can be removed,
            # or the code can be modified to not use the mujoco.viewer at all.
            time_between_updates = 1 / 100

            # Move the robot to the next waypoint by solving IK
            end_effector_task.set_target(path[curr_waypoint_idx])
            pos_threshold = 1e-4
            ori_threshold = 1e-4
            pos_achieved = False
            ori_achieved = False
            while not pos_achieved or not ori_achieved:
                start_time = time.time()

                vel = mink.solve_ik(
                    configuration,
                    tasks,
                    model.opt.timestep,
                    solver="quadprog",
                    damping=1e-3,
                    limits=limits
                )
                configuration.integrate_inplace(vel, model.opt.timestep)
                err = end_effector_task.compute_error(configuration)
                pos_achieved = np.linalg.norm(err[:3]) <= pos_threshold
                ori_achieved = np.linalg.norm(err[3:]) <= ori_threshold

                # Since we only need the joint configuration, running kinematic updates is sufficient.
                data.qpos = configuration.q
                mujoco.mj_kinematics(model, data)
                viewer.sync()

                time_until_next_step = time_between_updates - (time.time() - start_time)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

            # Save the joint configuration once the robot has reached the waypoint
            joint_configs.append(configuration.q.tolist())
            curr_waypoint_idx += 1

        with open('waypoint_configurations.json', 'w') as json_file:
            json.dump(joint_configs, json_file)
