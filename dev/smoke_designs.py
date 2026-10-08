import sys, time; sys.path.insert(0,'.')
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from src.synthetic_generator import generate_score2_five_groups_heter_patients, get_generation_rule_mask
from src.benchmark.estimators import *
from src.benchmark.designs import *
fe=["age","is_male","is_smoker","ldl_cholesterol","non_hdl_cholesterol","hdl_cholesterol","systolic_blood_pressure","biomarker_hba1c_ngsp_blood","estimated_glomerular_filtration_rate"]
df=generate_score2_five_groups_heter_patients(n_patients=2000,expected_n_pros=20,random_state=142)
dev=generate_score2_five_groups_heter_patients(n_patients=2000,expected_n_pros=20,random_state=999)
m=get_generation_rule_mask(df,'score2_five_groups_heter_patients')
co=Cohort.from_frame(df,fe); cdev=Cohort.from_frame(dev,fe)
folds=stratified_folds(co.physician,5,seed=1)
rows=[]
for est in build_estimators(include_genetic=False):
    for d in DESIGNS:
        if not supports(est,d) or equivalent_design(est,d)!=d: continue
        t=time.time(); r=run_design(est,co,d,folds=folds,eligible=m,development_cohort=cdev)
        rows.append(dict(method=est.name,design=d,mean_score=np.nanmean([e.score for e in r.estimates]),
             mean_pairs=np.mean([e.n_pairs for e in r.estimates]),n_phys=len(r.estimates),fit=r.fit_seconds,est=r.estimate_seconds,n_fits=r.n_fits))
out=pd.DataFrame(rows)
print(out.pivot(index='method',columns='design',values='mean_score').round(3).to_string())
print(out.pivot(index='method',columns='design',values='mean_pairs').round(0).to_string())
print(out.groupby('method')[['fit','est']].sum().round(1))
