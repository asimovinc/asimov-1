# Asimov v1: Open-Source Humanoid Robot

[![License: CERN-OHL-S-2.0](https://img.shields.io/badge/Hardware-CERN--OHL--S--2.0-blue)](HARDWARE-LICENSE.txt)
[![License: Software](https://img.shields.io/badge/Software-GPL--2.0-blue)](SOFTWARE-LICENSE.txt)

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

Asimov v1 is a 1.2 m, 35 kg biped with 25 actuated degrees of freedom. This repository contains the mechanical CAD, electrical CAD, simulation model, and onboard software to build, simulate, and customize Asimov v1.

---

## Specifications

| Spec | Value |
|---|---|
| Height | 1.2 m |
| Weight | 35 kg |
| Degrees of Freedom | 25 actuated + 2 passive |
| Legs | 6 DOF x 2 + toe x 2 |
| Arms | 5 DOF x 2 (shoulder pitch/roll/yaw, elbow, wrist yaw) |
| Torso | 1 DOF waist yaw, 10 W 4 ohm speaker, 6 DOF IMU |
| Head | 2 DOF neck (neck yaw, neck pitch), Quad microphone array, 2MP monocular camera |
| CAN Bus | 5 @ 1Mbps + 1 @ 500kbps |
| Onboard Compute | Raspberry Pi 5 (media + network) + Radxa CM5 (motion control) |
| Structural Materials | 7075 aluminium, MJF PA12 nylon |

| Activity | Load |
|---|---|
| Squat | 5 kg |
| Bicep curl | 15 kg each arm |
| Lateral raise | 18 kg each arm |

---

## Build your own Asimov

> [!TIP]
> **Option 1: DIY Kit:** Everything you need to build Asimov v1, unassembled. $499 deposit to reserve. $15,000 target price. Ships summer 2026. [Pre-order →](https://asimov.inc/diy-kit)

> [!NOTE]
> **Option 2: Self-source:** Pull the [BOM](https://manual.asimov.inc/v1/bom) and fabricate everything yourself. [Assembly Manual →](https://manual.asimov.inc)

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

Start with the repo-local [fabrication manifest](mechanical/FABRICATION_MANIFEST.csv) for the CAD-derived part inventory, then cross-reference the [BOM](https://manual.asimov.inc/v1/bom) and assembly manual for procurement details, sourcing, and fabrication notes.

**[Assembly Manual →](https://manual.asimov.inc)**

---

## Fabrication manifest

The CAD-derived fabrication manifest can be checked locally:

```bash
python3 scripts/generate_fabrication_manifest.py --check
```

---

## Roadmap

| Status | Item |
|---|---|
| ✅ | Mechanical CAD — 7 subassemblies |
| ✅ | MuJoCo simulation model |
| ✅ | Electrical wiring harness |
| ✅ | Electrical schematics & PCB files |
| 🔜 | Asimov API |
| 🔜 | Locomotion policy |
| 🔜 | Mobile app |

---

## Work with us

- **Build questions?**: Ask in the [forum](https://forum.menlo.ai) or open a [GitHub Issue](https://github.com/asimovinc/asimov-v1/issues) for bugs and contributions.
- **Deploying Asimov?**: [Talk to us →](mailto:bd@menlo.ai)
- **Supply chain partner?**: If you manufacture actuators, structural components, or electronics and want to be part of the Asimov supply chain, reach out.
[bd@menlo.ai](mailto:bd@menlo.ai)
