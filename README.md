Waypoint following with [mink](https://github.com/kevinzakka/mink).

## Setup

Install mink if needed (using a venv is encouraged):
```
pip install mink
```

## Usage

```shell
python follow_waypoints.py -h

# How to run the waypoint following script with a franka panda robot.
# Joint configurations for the waypoints will be saved to 'waypoint_configurations.json'.
python follow_waypoints.py panda
```

## Notes

A site named `ee_site` must be defined at the end-effector of the robot model so that mink can perform end-effector tracking.

The Franka Panda robot model was taken from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie).
The Barrett WAM model was taken from https://www.roboti.us/forum/index.php?resources/wam-and-barrett-hand.20/.
