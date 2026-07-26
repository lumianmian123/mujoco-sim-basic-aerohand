# AEROHAND MuJoCo Assets

This folder is the expected location for the AEROHAND MuJoCo model assets.

The XML, STL, and OBJ model files are **not redistributed** by this tutorial package. Please download them from the official TetherIA simulation repository at the pinned commit used for this tutorial:

[https://github.com/TetherIA/aero-open-sim/tree/dfeff43507fe82f515fed81592070e39a9bb92f6](https://github.com/TetherIA/aero-open-sim/tree/dfeff43507fe82f515fed81592070e39a9bb92f6)

## Expected Files

Place the downloaded files directly in this `assets/` directory so the final layout includes:

- `assets/right_hand.xml`
- `assets/scene_right.xml`
- `assets/*.STL`
- `assets/*.obj`

The XML uses `meshdir="assets/"`, so the mesh files must stay in this same directory.

## License Note

According to TetherIA's license summary:

- Commercial integration of purchased Aero Hand units is allowed.
- Software such as firmware and SDK is Apache-2.0.
- Design files such as CAD, STEP, STL, drawings, BOM, docs, and model assets are CC BY-NC-SA 4.0.
- The design files are non-commercial only; derivatives must use the same license with attribution.
- Manufacturing, printing parts, or making commercial clones/spares/kits from the design files requires a commercial manufacturing license from TetherIA.

Check the upstream repository and its `LICENSE.md` for the authoritative license terms before redistribution or commercial use.
