"""Check that the literature covariate model reproduces its source moments.

Run from the repository root: python dev/check_covariate_moments.py
"""
import sys
sys.path.insert(0, ".")
import numpy as np
from src.synthetic_generator import (build_base_cohort, get_generation_rule_mask,
                                     generate_score2_five_groups_heter_patients, ifcc_to_ngsp)

df = build_base_cohort(200_000, 20, 2027)
tc = df.non_hdl_cholesterol + df.hdl_cholesterol
rows = [
    ("Age, mean (SD)", "SCORE2 T1: 57 (9)", f"{df.age.mean():.1f} ({df.age.std():.1f})"),
    ("Men, %", "SCORE2 T1: 44", f"{100*df.is_male.mean():.1f}"),
    ("Current smokers, %", "SCORE2 T1: 15", f"{100*df.is_smoker.mean():.1f}"),
    ("SBP, mean (SD)", "SCORE2 T1: 136 (19)", f"{df.systolic_blood_pressure.mean():.1f} ({df.systolic_blood_pressure.std():.1f})"),
    ("Total cholesterol, mean (SD)", "SCORE2 T1: 5.8 (1.1) [not calibrated]", f"{tc.mean():.2f} ({tc.std():.2f})"),
    ("HDL, mean (SD)", "UKB 30760: 1.45 (0.38)", f"{df.hdl_cholesterol.mean():.2f} ({df.hdl_cholesterol.std():.2f})"),
    ("LDL, mean (SD)", "UKB 30780: 3.56 (0.87)", f"{df.ldl_cholesterol.mean():.2f} ({df.ldl_cholesterol.std():.2f})"),
    ("TG, median", "UKB 30870: 1.48", f"{df.triglycerides.median():.2f}"),
    ("HbA1c %, D1 / median / D9", "UKB 30750: %.2f / %.2f / %.2f" % tuple(ifcc_to_ngsp([30.5, 35.2, 41.1])),
     "%.2f / %.2f / %.2f" % tuple(np.quantile(df.biomarker_hba1c_ngsp_blood, [.1, .5, .9]))),
    ("eGFR, median at 40-44 / 70-74", "Gutenberg: ~105 / ~83",
     f"{df.estimated_glomerular_filtration_rate[df.age.between(40,44)].median():.0f} / "
     f"{df.estimated_glomerular_filtration_rate[df.age.between(70,74)].median():.0f}"),
    ("LDL > non-HDL, %", "0 by construction", f"{100*(df.ldl_cholesterol > df.non_hdl_cholesterol).mean():.2f}"),
]
w = max(len(r[0]) for r in rows)
for r in rows:
    print(f"{r[0]:<{w}}  target {r[1]:<42}  simulated {r[2]}")
s2 = generate_score2_five_groups_heter_patients(n_patients=200_000, expected_n_pros=20, random_state=7)
print(f"SCORE2 eligible fraction: {get_generation_rule_mask(s2, 'score2_five_groups_heter_patients').mean():.3f}")
d2 = build_base_cohort(200_000, 20, 2027, hba1c_law="lognormal_diabetes_mixture")
print(f"D.2 HbA1c >= 6.5%: {100*(d2.biomarker_hba1c_ngsp_blood >= 6.5).mean():.1f}%  (diabetes prevalence target 7.3%)")
