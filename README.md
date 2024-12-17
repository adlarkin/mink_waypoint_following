Waypoint following on a Franka Panda robot with [mink](https://github.com/kevinzakka/mink).

## Usage

```
# Install mink if needed.
# Using a venv is encouraged.
pip install mink

# Run the waypoint following script.
# Joint configurations for the waypoints will be saved to 'waypoint_configurations.json'.
python follow_waypoints.py

# Use 'waypoint_configurations.json' as you wish!
```

## Notes

The Franka Panda robot model was taken from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie).
A site named `ee_site` was added to [panda.xml](./franka_emika_panda/panda.xml) which is used for tracking the end effector with mink.
