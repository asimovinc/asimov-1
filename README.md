# Asimov v1: Open-Source Humanoid Robot

[![License: CERN-OHL-S-2.0](https://img.shields.io/badge/Hardware-CERN--OHL--S--2.0-blue)](HARDWARE-LICENSE.txt)
[![License: Software](https://img.shields.io/badge/Software-See%20License-blue)](SOFTWARE-LICENSE.txt)

Asimov is an open-source humanoid robot that you can build, train and customize.

![Asimov v1](assets/asimov-v1.jpg)
<p align="center">
  <a href="https://asimov.inc">Website</a> ·
  <a href="https://manual.asimov.inc">Manual</a> ·
  <a href="https://asimov.inc/diy-kit">DIY Kit</a> ·
  <a href="https://static.asimov.inc/asimov/v1/asimov-v1-20260420.html">3D Model</a> ·
  <a href="https://discord.gg/HzDfGN7kUw">Discord</a> ·
  <a href="https://x.com/asimovinc">X</a> ·
  <a href="https://forum.menlo.ai">Forum</a>
</p>

Asimov v1 is a 1.2 m, 35 kg biped with 25 actuated degrees of freedom. This repository contains the mechanical CAD, electrical harness, simulation model, and onboard software to build, simulate, and customize Asimov v1.

---

## Specifications

| Spec | Value |
|---|---|
| Height | 1.2 m |
| Weight | 35 kg |
| Degrees of Freedom | 25 actuated + 2 passive |
| Legs | 6 DOF x 2 + toe x 2 |
| Arms | 5 DOF x 2 (shoulder pitch/roll/yaw, elbow, wrist yaw) |
| Torso | 1 DOF waist yaw |
| Head | Integrated sensor suite + onboard compute |
| Load | 35 kg squat to standing |
| Motor Bus | 5 CAN buses |
| Onboard Compute | Raspberry Pi (API) + Radxa (firmware) |
| Structural Materials | 7075 aluminium, MJF PA12 nylon |
| Simulation | MuJoCo |

---

## Getting Started

1. **Read the manual first.** [manual.asimov.inc](https://manual.asimov.inc) covers safety, prerequisites, assembly, and software setup. Start here.
2. **Get the hardware.** [Pre-order the DIY Kit](https://asimov.inc/diy-kit) or pull the [BOM](https://manual.asimov.inc/v1/bom) and source everything yourself.
3. **Try the simulation.** Built for locomotion policy training and hardware-in-the-loop testing. Clone the repo and load the MuJoCo model.

```bash
git clone https://github.com/asimovinc/asimov-v1
python3 -m mujoco.viewer --mjcf=sim-model/xmls/asimov.xml
```

### What Asimov v1 does

**In scope:**
- Data collection: camera, audio, IMU, motor joint states
- Basic walking via teleoperation
- Custom AI agents via the Cloud API
- Virtual Asimov digital twin via the Cloud API

**Not in scope:**
- Manipulation (no hands or grippers)
- Advanced locomotion (e.g. running, jumping)
- Onboard training (pre-trained policies only)

---

## Repository Contents

```
asimov-v1/
├── mechanical/      STEP files for all 7 subassemblies + fabrication-ready parts
├── electrical/      Wiring harness definition (WireViz), power and CAN routing
├── sim-model/       MuJoCo model, XML + 28 STL link meshes
└── asimov-api/      Onboard API service, LiveKit bridge and safety layer (submodule)
```

### Mechanical

CAD for all 7 subassemblies. STEP files, CNC aluminium parts, MJF nylon parts, and off-the-shelf hardware.

Full assembly: [`mechanical/ASV1/ASIMOV_V1.STEP`](mechanical/ASV1/ASIMOV_V1.STEP)

**[View the 3D model in your browser →](https://static.asimov.inc/asimov/v1/asimov-v1-20260420.html)**

### Electrical

Wiring harness defined in [`electrical/wiring.yaml`](electrical/wiring.yaml) using [WireViz](https://github.com/wireviz/WireViz). Power and CAN routing for every joint.

### Simulation

A [MuJoCo](https://mujoco.org/) model with 25 actuated joints and 28 link meshes. Built for locomotion policy training and hardware-in-the-loop testing.

```bash
python3 -m mujoco.viewer --mjcf=sim-model/xmls/asimov.xml
```

### API

[`asimov-api`](https://github.com/asimovinc/asimov-api) is the onboard service running on the Raspberry Pi. It bridges remote clients to motor firmware over UDP and enforces safety on every command.

Full protocol reference: **[Asimov API Manual →](https://manual.asimov.inc/v1/api)**

---

## Build Your Own

> [!TIP]
> **DIY Kit:** Everything you need to build Asimov v1, unassembled. $499 deposit to reserve. $15,000 target price. Ships summer 2026. [Pre-order →](https://asimov.inc/diy-kit)

> [!NOTE]
> **Self-source:** Pull the [BOM](https://manual.asimov.inc/v1/bom) and fabricate everything yourself. [Assembly Manual →](https://manual.asimov.inc)

### DIY Kit

| Category | Included | Not Included |
|---|---|---|
| Hardware | All BOM components (unassembled), power supply & cabling, spare parts | Tools, hands |
| Compute | RPi edge board, motion control board, network board, power distribution board | 4G/5G modules |
| Sensors | Monocular camera, IMUs, mic, speaker, motor joint states | Lidar, 360 cam |
| Safety | Wireless E-Stop, safety guidelines | Battery |
| Docs | Quick start guide, manual, DIY build videos | — |

**[Pre-order the Asimov v1 DIY Kit →](https://asimov.inc/diy-kit)**

### Self-source

Pull the [BOM](https://manual.asimov.inc/v1/bom), source the parts, fabricate what needs fabricating.

**[Assembly Manual →](https://manual.asimov.inc)**

---

## Roadmap

| Status | Item |
|---|---|
| ✅ | Mechanical CAD — 7 subassemblies |
| ✅ | MuJoCo simulation model |
| ✅ | Electrical wiring harness |
| 🔜 | Electrical schematics & PCB files |
| 🔜 | MCU board design |
| 🔜 | Asimov API |
| 🔜 | Locomotion policy |
| 🔜 | Mobile app |

---

## Work With Us

**Build Questions?**
Ask in the [forum](https://forum.menlo.ai) or open a [GitHub Issue](https://github.com/asimovinc/asimov-v1/issues) for bugs and contributions.

**Deploying Asimov?**
[Talk to us →](mailto:bd@menlo.ai)

**Supply chain partner?**
If you manufacture actuators, structural components, or electronics and want to be part of the Asimov supply chain, reach out.
[bd@menlo.ai](mailto:bd@menlo.ai)

---

## License

Hardware: [CERN-OHL-S-2.0](HARDWARE-LICENSE.txt)
