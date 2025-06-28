# Initial Positioning State Implementation

## Overview

A new state called `INITIAL_POSITIONING` has been added to the AMCL navigation system. This state ensures that the robot rotates to face the direction of the first target point in the planned path before starting navigation.

## How It Works

1. **State Flow**: `IDLE` → `PLANNING` → `INITIAL_POSITIONING` → `NAVIGATING`

2. **Initial Positioning Process**:
   - When a new goal is received, the system plans a path
   - Before starting navigation, it calculates the target orientation based on the first segment of the path
   - The robot rotates to face this target orientation
   - Once the robot is within the tolerance angle, it transitions to `NAVIGATING` state

3. **Target Orientation Calculation**:
   - Uses the first point in the planned path as the target
   - Calculates the angle from the robot's current position to this target
   - This ensures the robot faces the direction it needs to travel

## Parameters

The following parameters control the initial positioning behavior:

- `initial_positioning_angular_speed` (default: 0.3 rad/s): Speed at which the robot rotates during initial positioning
- `initial_positioning_tolerance` (default: 0.1 rad ≈ 5.7°): Angular tolerance for considering the positioning complete

## Example Scenario

1. Robot is at position (0, 0) facing 65° (northeast)
2. Goal is received at position (2, 2) 
3. Path is planned from (0, 0) to (2, 2)
4. Target orientation is calculated as 45° (northeast)
5. Robot rotates from 65° to 45° (20° rotation)
6. Once within tolerance, robot starts navigating along the path

## Benefits

- **Smoother Navigation**: Robot starts navigation already facing the correct direction
- **Better Path Following**: Reduces initial path deviation
- **Improved Efficiency**: Minimizes unnecessary movements at the start of navigation

## Testing

A test script `test_initial_positioning.py` is provided to verify the functionality:

```bash
# Run the test script
python3 test_initial_positioning.py
```

The test script sends different goals to test various rotation scenarios:
- 90° rotation (goal to the right)
- -90° rotation (goal to the left) 
- 180° rotation (goal behind)
- Minimal rotation (goal in front)

## Logging

The system provides detailed logging during initial positioning:

```
[INFO] [my_py_amcl]: Path planned. Starting initial positioning to target orientation: 45.00°
[INFO] [my_py_amcl]: Initial positioning - Current: 65.00°, Target: 45.00°, Diff: -20.00°
[INFO] [my_py_amcl]: Initial positioning complete. Current yaw: 44.95°, Target: 45.00°
```

## Integration

The initial positioning state is fully integrated with the existing state machine:
- Works with obstacle avoidance
- Handles path replanning after obstacle avoidance
- Resets properly when new goals or initial poses are received
- Compatible with all existing navigation features 