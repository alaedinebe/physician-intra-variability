import logging, sys, time, tempfile
import numpy as np, pandas as pd
sys.path.insert(0, '.')
from src.synthetic_generator import generate_score2_five_groups_heter_patients
from src.analysis.analysis_pipeline import Analysis
from src.benchmark.estimators import *
import main as M, yaml
logging.basicConfig(level=logging.WARNING)
log = logging.getLogger("t")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
df = generate_score2_five_groups_heter_patients(n_patients=N, expected_n_pros=20, random_state=142)
cfg = yaml.safe_load(open('configs/article_stats.yaml'))
acfg = M.build_analysis_config(cfg); acfg['generation_strategy'] = 'score2_five_groups_heter_patients'
A = Analysis(df=df, results_dir=tempfile.mkdtemp(), LOGGER=log, config=acfg)
t=time.time(); A.run_glmm_analysis()
for f in ['run_manual_matching_analysis','run_euclidean_matching_analysis','run_discordance_matching_analysis',
          'run_learning_matching_analysis','run_random_forest_matching_analysis','run_mutual_info_analysis',
          'run_lpa_guided_pairing_analysis','run_genetic_mahalanobis_matching_analysis']:
    getattr(A, f)()
print('old pipeline', round(time.time()-t,1),'s')
old = A.results['intra_physician_variability'].set_index('physician')
fe = acfg['fixed_effects']
co = Cohort.from_frame(df, fe)
allidx = np.arange(co.n)
cols = {'euclidean':'discordance_rate_euclidean','mahalanobis':'discordance_rate_mahalanobis',
        'learned_weights':'discordance_rate_learning','rf_proximity':'discordance_rate_rf_matching',
        'mutual_information':'discordance_rate_mutual_info','lpa_guided':'discordance_rate_lpa_guided_pairing',
        'genetic_mahalanobis_legacy':'discordance_rate_genetic_mahalanobis','glmm':'overdispersion_local'}
for est in build_estimators(genetic_variant='legacy'):
    t=time.time(); est.fit(co, allidx); est.prepare(co)
    res = est.estimate(co, allidx) if est.kind=='model' else estimate_matching(est, co, allidx)
    new = pd.Series({r.physician: r.score for r in res})
    o = old[cols[est.name]].reindex(new.index)
    tol = 1e-3 if est.name=='glmm' else 1e-12
    print(f"{est.name:28s} max|diff|={np.nanmax(np.abs(new.values-o.values)):.2e}  {'OK' if np.nanmax(np.abs(new.values-o.values))<=tol else 'DIFF'}  {time.time()-t:.1f}s")
