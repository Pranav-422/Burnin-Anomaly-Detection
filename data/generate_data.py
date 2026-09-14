"""
Physics-informed synthetic burn-in dataset generator -- multi-parameter.

Simulates THREE burn-in parameters per component across a 125C burn-in
cycle, measured at 0h / 24h / 96h / 168h:
  - leakage_ua     : standby leakage current (uA)
  - iddq_ua        : quiescent supply current, Iddq (uA)
  - prop_delay_ns  : propagation delay (ns)

Physical basis -- the Arrhenius equation:

    AF = exp[ (Ea / k_B) * (1/T_use - 1/T_test) ]

  where:
    AF      = acceleration factor (how much faster degradation
              proceeds at the elevated burn-in temperature vs. the
              component's normal operating temperature)
    Ea      = activation energy (eV) of the dominant failure/drift
              mechanism -- varies by parameter and by defect type
    k_B     = Boltzmann constant, 8.617e-5 eV/K
    T_test  = burn-in temperature (125C = 398.15K)
    T_use   = normal operating temperature (25C = 298.15K)

Each component gets ONE is_defective draw (not one per parameter) --
a genuinely defective part shows CORRELATED elevated drift across all
three parameters, using that parameter's own ea_defect. This mirrors a
real physical defect (e.g. a gate-oxide weakness) that manifests across
multiple electrical characteristics simultaneously, not three
independent unrelated anomalies.

For each parameter, the BASE_RATE growth-rate constant is derived
(not hand-tuned) so that, on average, a defective part's 168h value
lands at ~50% of that parameter's max_limit -- deliberately mirroring
the original single-parameter (leakage-only) calibration, so every
parameter's "latent defect" stays plausible under its own datasheet
spec, same as the original design intent.
"""

import numpy as np
import pandas as pd

RNG_SEED = 42
CHECKPOINTS = [0, 24, 96, 168]  # hours

K_B = 8.617e-5  # Boltzmann constant, eV/K
T_TEST_K = 125 + 273.15  # burn-in temperature
T_USE_K = 25 + 273.15    # normal operating temperature

# One entry per screened parameter. ea_normal/ea_defect are representative
# literature-range activation energies (eV); baseline_mean/max_limit are
# representative datasheet-order-of-magnitude values -- all placeholders
# per the task sheet, kept physically self-consistent via the calibration
# below rather than hand-tuned per parameter.
PARAMETERS = {
    "leakage_ua": {
        "baseline_mean": 10.0, "ea_normal": 0.35, "ea_defect": 0.70, "max_limit": 50.0,
    },
    "iddq_ua": {
        "baseline_mean": 25.0, "ea_normal": 0.30, "ea_defect": 0.65, "max_limit": 120.0,
    },
    "prop_delay_ns": {
        "baseline_mean": 4.5, "ea_normal": 0.25, "ea_defect": 0.55, "max_limit": 12.0,
    },
}

ALL_PARAMETERS = list(PARAMETERS.keys())


def acceleration_factor(ea_ev: float) -> float:
    """Arrhenius acceleration factor between burn-in and use temperature."""
    return np.exp((ea_ev / K_B) * (1 / T_USE_K - 1 / T_TEST_K))


def _calibrate_base_rate(ea_defect: float, baseline_mean: float, max_limit: float,
                          t_ref: float = 168.0, n_ref: float = 0.65) -> float:
    """
    Derives BASE_RATE so a defective part's 168h value lands at ~50% of
    this parameter's max_limit on average -- same target ratio the
    original single-parameter (leakage) model was calibrated to,
    applied generically to any parameter's own baseline/limit.
    """
    target_ratio = 0.5 * max_limit / baseline_mean
    target_growth = target_ratio - 1.0  # value = baseline * (1 + growth)
    af_defect = acceleration_factor(ea_defect)
    return target_growth / (af_defect * (t_ref ** n_ref))


def _drift_curve(i0, k, n, t):
    """Leakage-style growth curve; k is derived from the Arrhenius AF."""
    return i0 * (1 + k * (t ** n))


def generate_burnin_dataset(
    n_lots: int = 25,
    parts_per_lot: int = 40,
    defect_rate: float = 0.06,
    seed: int = RNG_SEED,
    parameters=None,
) -> pd.DataFrame:
    """
    Returns a LONG-format DataFrame:
        component_id, lot_id, hour, parameter_name, value, is_defective

    `parameters` -- list of parameter keys from PARAMETERS to generate.
    Defaults to ALL_PARAMETERS (all three). Pass e.g. ["leakage_ua"] for
    the legacy single-parameter behavior.
    """
    if parameters is None:
        parameters = ALL_PARAMETERS
    for p in parameters:
        if p not in PARAMETERS:
            raise ValueError(f"Unknown parameter: {p}. Known: {ALL_PARAMETERS}")

    rng = np.random.default_rng(seed)
    rows = []
    component_counter = 0

    # Pre-calibrate BASE_RATE per parameter (derived, not hand-tuned).
    base_rates = {
        p: _calibrate_base_rate(cfg["ea_defect"], cfg["baseline_mean"], cfg["max_limit"])
        for p, cfg in PARAMETERS.items()
    }

    for lot_idx in range(n_lots):
        lot_id = f"LOT_{lot_idx:03d}"

        # Per-lot baseline behavior (process variation), one per parameter.
        lot_i0 = {
            p: max(rng.normal(PARAMETERS[p]["baseline_mean"],
                               PARAMETERS[p]["baseline_mean"] * 0.12), 0.5)
            for p in parameters
        }
        # Per-lot curvature exponent, one per parameter (different physical
        # quantities drift with different characteristic curvature).
        lot_n = {p: rng.uniform(0.55, 0.75) for p in parameters}

        for _ in range(parts_per_lot):
            component_counter += 1
            component_id = f"C_{component_counter:05d}"
            # ONE defect draw per component -- correlated across parameters.
            is_defective = rng.random() < defect_rate

            for p in parameters:
                cfg = PARAMETERS[p]
                i0 = max(rng.normal(lot_i0[p], lot_i0[p] * 0.06), 0.1)

                if is_defective:
                    ea = max(rng.normal(cfg["ea_defect"], cfg["ea_defect"] * 0.05), 0.15)
                else:
                    ea = max(rng.normal(cfg["ea_normal"], cfg["ea_normal"] * 0.11), 0.1)

                af = acceleration_factor(ea)
                k = base_rates[p] * af * rng.uniform(0.9, 1.1)

                for t in CHECKPOINTS:
                    noise = rng.normal(0, i0 * 0.02)
                    value = _drift_curve(i0, k, lot_n[p], t) + noise
                    rows.append({
                        "component_id": component_id,
                        "lot_id": lot_id,
                        "hour": t,
                        "parameter_name": p,
                        "value": round(max(value, 0.001), 4),
                        "is_defective": is_defective,
                    })

    return pd.DataFrame(rows)


def to_wide(df: pd.DataFrame, parameters=None) -> pd.DataFrame:
    """
    Pivots long-format data into one row per component.

    BACKWARD-COMPAT: if exactly one parameter is present, columns are
    unsuffixed (Value_0h, Value_24h, ...) to match the original
    single-parameter schema exactly. With multiple parameters, columns
    are suffixed per parameter (Value_0h_leakage_ua, Value_0h_iddq_ua, ...).
    """
    if parameters is None:
        parameters = sorted(df["parameter_name"].unique().tolist())

    meta = df[["component_id", "lot_id", "is_defective"]].drop_duplicates(subset="component_id")

    wide = meta.copy()
    single_param_mode = len(parameters) == 1

    for p in parameters:
        sub = df[df["parameter_name"] == p]
        piv = sub.pivot(index="component_id", columns="hour", values="value")
        for hour in piv.columns:
            col_name = f"Value_{hour}h" if single_param_mode else f"Value_{hour}h_{p}"
            wide = wide.merge(
                piv[[hour]].rename(columns={hour: col_name}),
                left_on="component_id", right_index=True, how="left",
            )

    wide = wide.sort_values("component_id").reset_index(drop=True)
    return wide


def value_col(hour: int, parameter: str = None) -> str:
    """Column-name helper shared with downstream modules. Pass
    parameter=None for the legacy single-parameter (unsuffixed) schema."""
    return f"Value_{hour}h" if parameter is None else f"Value_{hour}h_{parameter}"


def detect_parameters(wide_df: pd.DataFrame):
    """
    Inspects a wide-format dataframe's columns and returns the list of
    parameters present. Returns [None] for legacy unsuffixed
    single-parameter data (so value_col(hour, None) resolves correctly).
    """
    suffixed = sorted({
        c[len("Value_168h_"):] for c in wide_df.columns if c.startswith("Value_168h_")
    })
    if suffixed:
        return suffixed
    if "Value_168h" in wide_df.columns:
        return [None]
    raise ValueError("No Value_168h* column found in dataframe")


if __name__ == "__main__":
    long_df = generate_burnin_dataset()
    wide_df = to_wide(long_df)

    long_df.to_csv("outputs/burnin_data_long.csv", index=False)
    wide_df.to_csv("outputs/burnin_data_wide.csv", index=False)

    print(f"Generated {wide_df.shape[0]} components across "
          f"{wide_df['lot_id'].nunique()} lots, {len(ALL_PARAMETERS)} parameters "
          f"({', '.join(ALL_PARAMETERS)}).")
    print(f"Defective parts: {wide_df['is_defective'].sum()} "
          f"({wide_df['is_defective'].mean():.1%})")
    print()
    for p in ALL_PARAMETERS:
        col168 = value_col(168, p)
        limit = PARAMETERS[p]["max_limit"]
        n_over = (wide_df[wide_df.is_defective][col168] > limit).sum()
        n_def = wide_df.is_defective.sum()
        print(f"[{p}] mean(normal)={wide_df[~wide_df.is_defective][col168].mean():.2f}, "
              f"mean(defective)={wide_df[wide_df.is_defective][col168].mean():.2f}, "
              f"defects exceeding spec ({limit}): {n_over}/{n_def}")
    print()
    print(wide_df.head())
