import sys; sys.path.insert(0,'.')
import numpy as np, warnings; warnings.filterwarnings('ignore')
from src.synthetic_generator import generate_score2_five_groups_heter_patients, get_generation_rule_mask
from src.benchmark.estimators import *
from src.benchmark.designs import run_design
fe=["age","is_male","is_smoker","ldl_cholesterol","non_hdl_cholesterol","hdl_cholesterol","systolic_blood_pressure","biomarker_hba1c_ngsp_blood","estimated_glomerular_filtration_rate"]
df=generate_score2_five_groups_heter_patients(n_patients=2000,expected_n_pros=20,random_state=142)
m=get_generation_rule_mask(df,'score2_five_groups_heter_patients')
co=Cohort.from_frame(df,fe)
from src.analysis.analysis_pipeline import Analysis
gt={}
for p in np.unique(co.physician):
    i=np.where((co.physician==p)&m)[0]; y=co.y[i]; k=y.sum(); n=len(y); gt[p]=2*k*(n-k)/(n*(n-1))
for v in ['legacy','genmatch']:
    r=run_design(GeneticMahalanobisEstimator(v),co,'in_sample')
    sc={e.physician:e.score for e in r.estimates}
    print(v, 'mean delta vs manual = %.3f'%np.mean([sc[p]-gt[p] for p in sc]), ' fit %.0fs'%r.fit_seconds)
mh=MahalanobisEstimator(); mh.pairing='nn_replacement'
r=run_design(mh,co,'in_sample'); sc={e.physician:e.score for e in r.estimates}
print('plain mahalanobis NN-repl mean delta = %.3f'%np.mean([sc[p]-gt[p] for p in sc]))
g=GeneticMahalanobisEstimator('genmatch').fit(co,np.arange(co.n)); print('genmatch weights^2:',np.round(g.weights_**2,3))
