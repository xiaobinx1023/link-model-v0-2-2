# Wireline Model Framework Structure and Improvement Plan

## 1) Scope
This repository currently contains two Python wireline modeling frameworks:

- `src_py/link_model`: bidirectional `Link` (master/slave) model with AFE + channel + CDR RX.
- `src_py_uni/uni_link_model`: unidirectional `UniDirLink` model with TX FFE + channel + RX + configurable aggressors.

Both frameworks support channel S-parameter based impulse generation, time-domain simulation, and eye/pulse diagnostics.

## 2) High-Level Architecture

### 2.1 Shared Building Blocks (`src_py/link_model`)
- `clock_gen.py`, `pi.py`, `clock_delay.py`: clock generation and phase interpolation.
- `data_gen.py`, `tx.py`, `driver.py`: data pattern generation and TX path.
- `channel.py`, `tools.py`: channel filtering and S-parameter utilities.
- `rx.py`: RX chain with CTLE/DFE/slicer/aperture/CDR.
- `ctle.py`, `dfe.py`, `aperture.py`, `slicer.py`: RX DSP/equalization blocks.
- `eye_monitor.py`: eye trace accumulation, statistics, mask drawing, matplotlib/plotly output.
- `controller.py`: per-side control object (used in bidirectional link).
- `io_termination.py`: IO/termination abstraction for bidirectional model.

### 2.2 Bidirectional Link (`src_py/link_model/link.py`)
- Two TX/RX sides (`master`, `slave`) with independent `Controller`.
- AFE on each side and channel transfer generation from `.sNp`.
- Crosstalk (NEXT/FEXT) from configurable aggressor ports.
  - !#TODO find the dominant aggressors(descending) by inpecting the channel .sNp file and termination in link.
  - Aggressor driver lanes now mirror victim-side termination assumptions and support per-port bit pattern plus TX/RX PI-code configuration to control aggressor-victim phase relationship.
- Diagnostics:
  - path impulse responses
    !#TODO: have a sanity check againts the channel .sNp mag response.
  - aggressor-to-victim pulse responses
  - eye metrics via each RX monitor

### 2.3 Unidirectional Link (`src_py_uni/uni_link_model/uni_link.py`)
- TX FFE (`tx_ffe.py`) with segment-based tap definition.
- RX termination controlled by `rx_term_code`.
- IO abstraction in `io_termination_uni.py`.
- Aggressor framework:
  - selectable ports
  - per-port aggressor lane driver model (`PI(tx/rx) -> DataGen -> Driver`) using the same IO/termination assumptions as victim
  - source mode: `manual` or `pattern`
  - per-port pattern/amplitude override
  - per-port aggressor `tx_pi_code` / `rx_pi_code` and phase-offset controls to set aggressor-victim phase relationship
- Diagnostics:
  - monitor-point pulse responses
  - specific aggressor port to victim port pulse responses
  - eye diagram with/without aggressors

## 3) Config Surfaces in Current Code

### 3.1 Bidirectional (`Link`)
- Constructor inputs:
  - `chan_file`, `chan_port_one_sel`, `chan_port_two_sel`, `channel_pairs`, `aggressor_ports`
- Per-side runtime controls via `Controller`:
  - TX pattern/codes/polarity/enables
  - PI phase code
  - RX slicer ref, CDR gain
  - RX CTLE/DFE/sensitivity/aperture
- IO/termination tuning via `link.io_termination` dataclass configs:
  - `DriverResistanceConfig`
  - `SideTerminationConfig` for `master` and `slave`

### 3.2 Unidirectional (`UniDirLink`)
- Constructor inputs:
  - `chan_file`, victim TX/RX ports, `tx_pattern`, `tx_ffe_taps`
  - PI codes, `rx_term_code`, `rx_clk_ofst`, `rx_slicer_ref`, `rx_pd_out_gain`
  - `channel_pairs`, `aggressor_ports`, `aggressor_enable`
- Runtime aggressor controls:
  - `set_aggressor_enable`, `set_aggressor_source_mode`
  - `set_aggressor_sources`, `broadcast_aggressor_pattern`, `set_aggressor_pattern`
- RX EQ controls through `link.rx`:
  - CTLE (`ctle_*`)
  - DFE (`dfe_*`)
  - slicer and aperture
  - eye monitor timing (`samples_per_ui`, `eye_trace_span_ui`, `sample_rate_hz`)
- IO/termination tuning through `link.io_term`:
  - `TxTerminationConfig`, `RxTerminationConfig`

## 4) `specs.yaml` Contract
`specs.yaml` is added at project root (`v0.2.2/specs.yaml`) to serve as a unified user-facing configuration interface.

Design goals:
- One file for all model knobs and plotting controls.
- Support both model variants:
  - `active_model: dual_dir`
  - `active_model: uni_dir`
- Keep naming close to code fields to minimize mapping ambiguity.
- Include defaults and comments for guided editing.

## 5) Recommended Next Improvements

1. Add a typed loader (`pydantic` or dataclass validators) for `specs.yaml`.
2. Create `from_spec()` builders:
   - `build_link_from_spec(spec)`
   - `build_uni_link_from_spec(spec)`
3. Standardize naming:
   - unify `rx_clk_ofset` (controller typo) vs `rx_clk_ofst`.
4. Move class constants (`SAMP_FREQ_HZ`, `CLK_FREQ_HZ`) into configurable runtime fields.
5. Add schema versioning/migration helpers for spec evolution.
6. Map test cases to spec IDs (align with `AGENTS.md` rule for test traceability).
7. Add artifact policy in runner:
   - save figures/tables with spec hash + timestamp for reproducibility.

## 6) Minimal Usage Pattern (target)
```python
import yaml
from pathlib import Path

spec = yaml.safe_load(Path("specs.yaml").read_text())

if spec["active_model"] == "uni_dir":
    # build UniDirLink from spec["uni_dir"] + spec["common"]
    pass
else:
    # build Link from spec["dual_dir"] + spec["common"]
    pass
```
