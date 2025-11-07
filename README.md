# CAM + FTorch Deep Convection (YOG) Integration

This repository documents and provides sources for integrating a PyTorch-based
deep-convection scheme (YOG) into CESM/CAM using **FTorch**.

- **Component:** CAM (CESM3)
- **What’s replaced:** ZM/YOG convection tendencies via FTorch TorchScript model
- **Key idea:** Keep CAM physics + vertical remapping intact; swap the NN call with
  a TorchScript forward pass (FTorch), preserving CAM data flow.



## 🧩 Repository Structure

```text
FTorch_CAM_integration/
├── src/
│   └── cam/                        # Modified CAM physics source files
│       ├── Phys_control.F90
│       ├── physpkg.F90
│       ├── yog_intr.F90
│       ├── nn_interface_CAM.F90
│       ├── nn_convection_flux.F90
│       └── nn_cf_net.F90
│
├── libraries/
│   └── FTorch/
│       └── FTorch_cesm_interface.F90   # Wrapper for FTorch model calls
│
├── docs/
│   ├── build_instructions.md
│   └── troubleshooting.md
│
├── examples/
│   └── user_nl_cam
│
├── MODEL_CARD.md
└── README.md

```

---

## Prerequisites

- CESM 3.0 (tested with `cesm3_0_alpha07f`)
- FTorch (Fortran interface to PyTorch)

## Overview of Integration

- yog_intr.F90 — reads new namelist options (yog_nn_weights, yog_device, etc.) and initializes the neural-network convection module.
- nn_interface_CAM.F90 — uses FTorch_cesm_interface.F90 to load the TorchScript model (*.pt) and perform inference.
- nn_convection_flux.F90 — maps model outputs to CAM tendencies and fluxes.
- Phys_control.F90 / physpkg.F90 — enable toggling the ML convection scheme.
- nn_cf_net.F90 — legacy netCDF weight reader replaced by FTorch loader.
