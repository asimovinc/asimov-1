# Simulation Model

MuJoCo model for Asimov v1. 25 actuated joints, 28 link meshes, friction-tuned foot contacts. Built for locomotion policy training and hardware-in-the-loop testing.

## Contents

```
sim-model/
├── xmls/asimov.xml     Full robot model
└── assets/meshes/      28 STL link meshes
```

## Usage

```bash
python3 -m mujoco.viewer --mjcf=sim-model/xmls/asimov.xml
```

Requires [MuJoCo](https://mujoco.org/) 3.x.

## Resources

- [Locomotion Control guide](https://manual.asimov.inc/v1/locomotion) — policy training, control modes, and hardware-in-the-loop setup
- [Asimov API](https://github.com/asimovinc/asimov-api) — deploy policies to the real robot
- [Discord](https://discord.gg/HzDfGN7kUw) — questions and discussion
