# GymBoxManipulation

A PyBullet-based environment for block rearrangement tasks using a Franka Panda robot.  
Designed for reinforcement learning with stacking, grasping, and manipulation skills.

---

## 🚧 Features

- Modular robot control with high-level primitives
- Support for stacking, color-coded targets, and multiple dump areas
- Curriculum-friendly task configurations (YAML-based)
- Fast headless simulation for training

---

## 🧠 Environments

### `PhysicsBlockRearrangement-v0`

- Block picking and placing
- Supports stacking and dumping
- Goal configurations customizable via python files & config files

---

## 🛠️ Installation

Clone the repo and install in editable mode:

```bash
git clone https://github.com/NikoFeith/GymBoxManipulation.git
cd GymBoxManipulation
pip install -e .
