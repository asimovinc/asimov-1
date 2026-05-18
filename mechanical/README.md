# Mechanical Design

CAD for all 7 Asimov v1 subassemblies. STEP files, CNC aluminium parts, MJF nylon parts, and off-the-shelf hardware.

## Contents

```
mechanical/ASV1/
├── ASIMOV_V1.STEP          Full robot assembly
├── 100/ - 700/             Subassemblies
│   ├── ASV1_NNN.STEP       Subassembly STEP
│   └── FABRICATION/
│       ├── ALU_7075/       CNC-machined aluminium parts
│       ├── MJF_PA12/       3D-printed nylon parts (HP MJF)
│       └── OFF_THE_SHELF/  Standard hardware
└── NamingConvention.png    Part naming reference
```

`FABRICATION_MANIFEST.csv` and `FABRICATION_MANIFEST.json` are generated from the STEP files under `mechanical/ASV1/*/FABRICATION`. They provide the repo-local fabrication inventory for self-source builds and CI checks.

**[View the full assembly in your browser →](https://static.asimov.inc/asimov/v1/asimov-v1-20260420.html)**

## Validation

Regenerate or verify the fabrication manifest from the repository root:

```bash
python3 scripts/generate_fabrication_manifest.py
python3 scripts/generate_fabrication_manifest.py --check
```

## Naming Convention

![Naming Convention](./naming_convention.png)
