import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config as C


def apply_intervention(facility_count, intervention):
    kind = intervention.get("kind", "do_nothing")
    if kind == "do_nothing":
        return facility_count, 0.0, 0.0
    if kind == "build_facilities":
        n = intervention.get("n_facilities", 1)
        return facility_count + n, n * C.COST_PER_NEW_PHC_FACILITY_USD, 0.0
    if kind == "deploy_staff":
        n = intervention.get("n_staff", 5)
        equiv = n / intervention.get("staff_per_facility_equiv", 5)
        return facility_count + equiv, 0.0, n * C.COST_PER_HEALTH_WORKER_YEAR_USD
    if kind == "supply_chain":
        boost = intervention.get("capacity_boost", 0.15)
        return (facility_count * (1 + boost),
                intervention.get("fixed_cost", 50000.0),
                intervention.get("annual_cost", 20000.0))
    raise ValueError(f"unknown intervention kind: {kind}")


def simulate_lga(lga, intervention, model_slope_per_ppf, model_rel_uncertainty,
                 horizons=None, n_runs=None, seed=0):
    horizons = sorted(horizons or C.HORIZONS_YEARS)
    n_runs = n_runs or C.MONTE_CARLO_RUNS
    rng = np.random.default_rng(seed)
    max_h = max(horizons)
    H = len(horizons)

    pop0 = float(lga["population"])
    fac0 = float(lga["facility_count"])
    fac1, capital, recurring = apply_intervention(fac0, intervention)

    deaths = np.zeros((n_runs, H))
    dalys = np.zeros((n_runs, H))
    value = np.zeros((n_runs, H))

    for i in range(n_runs):
        u5mr = rng.uniform(*C.U5MR_CI) / 1000.0
        vpf = rng.uniform(*C.VACCINE_PREVENTABLE_FRACTION_RANGE)
        elast = rng.uniform(*C.UTILISATION_TO_MORTALITY_ELASTICITY_RANGE)
        val_per_daly = rng.uniform(*C.VALUE_PER_DALY_USD_RANGE)
        slope = model_slope_per_ppf * (1 + rng.normal(0, model_rel_uncertainty))

        cum_deaths = 0.0
        cum_value = 0.0
        hi = 0
        for y in range(1, max_h + 1):
            pop_y = pop0 * (1 + C.ANNUAL_POP_GROWTH_RATE) ** y
            births_y = pop_y * C.CRUDE_BIRTH_RATE_PER_1000 / 1000.0
            ppf_base = pop_y / fac0 if fac0 > 0 else pop_y
            ppf_after = pop_y / fac1 if fac1 > 0 else pop_y
            delta = ppf_base - ppf_after
            util_gain = max(0.0, slope * delta) if delta > 0 else 0.0

            preventable = births_y * u5mr * vpf
            d_av = preventable * min(0.95, elast * util_gain)
            disc = 1 / (1 + C.DISCOUNT_RATE) ** y

            cum_deaths += d_av
            cum_value += d_av * C.DALYS_PER_U5_DEATH * val_per_daly * disc

            if y == horizons[hi]:
                deaths[i, hi] = cum_deaths
                dalys[i, hi] = cum_deaths * C.DALYS_PER_U5_DEATH
                value[i, hi] = cum_value
                hi += 1

    def pctl(a):
        return {"p10": float(np.percentile(a, 10)),
                "p50": float(np.percentile(a, 50)),
                "p90": float(np.percentile(a, 90))}

    results = {}
    for j, year in enumerate(horizons):
        total_cost = capital + recurring * year
        benefit = pctl(value[:, j])
        results[year] = {
            "deaths_averted": pctl(deaths[:, j]),
            "dalys_averted": pctl(dalys[:, j]),
            "benefit_usd": benefit,
            "intervention_cost_usd": total_cost,
            "cost_of_inaction_usd": benefit["p50"],
            "net_benefit_usd": benefit["p50"] - total_cost,
        }
    return results


def compare_scenarios(lga, scenarios, model_slope_per_ppf, model_rel_uncertainty):
    return {name: simulate_lga(lga, iv, model_slope_per_ppf, model_rel_uncertainty)
            for name, iv in scenarios.items()}


def explain_chain(lga, intervention, model_slope_per_ppf):
    pop = float(lga["population"])
    fac0 = float(lga["facility_count"])
    fac1, capital, recurring = apply_intervention(fac0, intervention)
    ppf_before = pop / fac0 if fac0 > 0 else pop
    ppf_after = pop / fac1 if fac1 > 0 else pop
    util_gain = max(0.0, model_slope_per_ppf * (ppf_before - ppf_after))
    births = pop * C.CRUDE_BIRTH_RATE_PER_1000 / 1000.0
    preventable = births * (C.BASELINE_U5MR_PER_1000 / 1000.0) * C.VACCINE_PREVENTABLE_FRACTION
    deaths_per_year = preventable * min(0.95, C.UTILISATION_TO_MORTALITY_ELASTICITY * util_gain)
    return {
        "facilities_before": fac0, "facilities_after": fac1,
        "ppf_before": ppf_before, "ppf_after": ppf_after,
        "util_gain": util_gain, "births": births,
        "preventable_per_year": preventable, "deaths_per_year": deaths_per_year,
    }


def default_scenarios():
    return {
        "do_nothing":       {"kind": "do_nothing"},
        "build_2_clinics":  {"kind": "build_facilities", "n_facilities": 2},
        "deploy_15_staff":  {"kind": "deploy_staff", "n_staff": 15},
        "supply_chain_fix": {"kind": "supply_chain", "capacity_boost": 0.15},
    }


if __name__ == "__main__":
    demo = {"population": 320000, "facility_count": 12, "pop_per_facility": 26666}
    res = compare_scenarios(demo, default_scenarios(),
                            model_slope_per_ppf=0.0028, model_rel_uncertainty=0.27)
    for name, hz in res.items():
        h5 = hz[5]
        print(f"{name:18s} 5yr CUMULATIVE: deaths averted "
              f"p50={h5['deaths_averted']['p50']:.0f} "
              f"(p10={h5['deaths_averted']['p10']:.0f}, p90={h5['deaths_averted']['p90']:.0f}) "
              f"| cost ${h5['intervention_cost_usd']:,.0f} "
              f"| inaction cost ${h5['cost_of_inaction_usd']:,.0f}")