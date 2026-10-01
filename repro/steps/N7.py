"""N7: outcome-blind Validation scoring of the immutable N6 model."""
from google.colab import drive
drive.mount('/content/drive')
from pathlib import Path
from datetime import datetime, timezone
import hashlib,json,os,tempfile,joblib,numpy as np,pandas as pd

EXPECTED=14923
ROOT=Path('/content/drive/MyDrive/CKD_Project/outputs/section_5_modeling/step_5_9_longitudinal_primary_modeling')
N6_ROOT=ROOT/'05_12N6_full_training_pulscar_mean_xgb_final_refit_and_lock'
N6B_ROOT=ROOT/'05_12N6B_training_only_baseline_model_benchmark'
OUT_ROOT=ROOT/'05_12N7_outcome_blind_locked_pulscar_mean_xgb_validation_scoring'
VALIDATION_FILE=ROOT/('05_7B_exact_outcome_blind_validation_predictor_reconstruction/'
    '8d0da0600f47da3a619b7176/PRIVATE_INTERNAL_DO_NOT_SHARE/'
    'renal_eligible_validation_predictors_no_outcome.parquet')

def loadj(p):
    try:return json.loads(Path(p).read_text())
    except:return None
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for c in iter(lambda:f.read(1048576),b''):h.update(c)
    return h.hexdigest()
def safe(x):
    if isinstance(x,dict):return {str(k):safe(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [safe(v) for v in x]
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,Path):return str(x)
    return x
def atomic(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);fd,t=tempfile.mkstemp(dir=p.parent,suffix='.tmp')
    with os.fdopen(fd,'w') as f:json.dump(safe(x),f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
    os.replace(t,p)

# Require completion of the locked model and all prespecified Training analyses.
c6=[(p,loadj(p)) for p in N6_ROOT.rglob('completion.json') if loadj(p) and loadj(p).get('status')=='FULL_TRAINING_PULSCAR_MEAN_XGB_MODEL_LOCKED']
if not c6:raise FileNotFoundError('N6 lock not found.')
N6_FILE,n6=max(c6,key=lambda z:z[0].stat().st_mtime);N6=N6_FILE.parent
c6b=[(p,loadj(p)) for p in N6B_ROOT.rglob('completion.json') if loadj(p) and loadj(p).get('status')=='SUPPLEMENTARY_NESTED_TRAINING_ONLY_BASELINE_BENCHMARK_COMPLETE']
if not c6b:raise FileNotFoundError('Complete N6B before Validation scoring.')
N6B_FILE,n6b=max(c6b,key=lambda z:z[0].stat().st_mtime)
if n6b.get('validation_accessed_in_this_step') or n6b.get('test_accessed_in_this_step'):raise ValueError('N6B reports held-out access.')
if n6.get('uacr_used_anywhere_in_modeling') or n6.get('raw_feature_count')!=20 or n6.get('transformed_feature_count')!=21:raise ValueError('N6 feature lock changed.')
if n6.get('threshold_selected') or n6.get('priority_categories_selected') or n6.get('validation_accessed_in_this_step') or n6.get('test_accessed_in_this_step'):raise ValueError('N6 lock is incompatible.')

MODEL=N6/'locked_official_pulscar_mean_xgb_model.joblib'
MODEL_JSON=N6/'locked_official_pulscar_mean_xgb_model.json'
PREP=N6/'locked_mean_imputation_preprocessor.joblib'
MANIFEST=N6/'locked_transformed_feature_manifest.csv'
for p,k in [(MODEL,'model_joblib_sha256'),(MODEL_JSON,'model_json_sha256'),(PREP,'downstream_preprocessor_sha256'),(MANIFEST,'feature_manifest_sha256')]:
    if not p.is_file() or sha(p)!=n6['artifact_hashes'][k]:raise ValueError(f'Locked artifact changed: {p}')
if not VALIDATION_FILE.is_file():raise FileNotFoundError(VALIDATION_FILE)

model=joblib.load(MODEL);prep=joblib.load(PREP);manifest=pd.read_csv(MANIFEST);validation=pd.read_parquet(VALIDATION_FILE)
ID='patient_id'
if len(validation)!=EXPECTED or ID not in validation or validation[ID].isna().any() or not validation[ID].is_unique:raise ValueError('Unexpected Validation cohort.')
tokens=('ckd_class','documented_ckd','observed_label','true_label','outcome','persistent_evidence','final_label')
for col in validation.columns:
    if any(t in str(col).lower() for t in tokens):raise ValueError(f'Outcome-like Validation column present: {col}')
numeric=list(prep['numeric_features']);features=numeric+[prep['categorical_feature']]
if len(features)!=20 or any('uacr' in x.lower() for x in features):raise ValueError('uACR or wrong feature panel reached N7.')
missing=sorted(set(features+[ID])-set(validation.columns))
if missing:raise KeyError(f'Validation predictors missing: {missing}')

raw_numeric=validation[numeric].apply(pd.to_numeric,errors='coerce').replace([np.inf,-np.inf],np.nan)
availability=pd.DataFrame({ID:validation[ID].to_numpy()})
for f in features:availability[f]=validation[f].notna().to_numpy(bool)
num=prep['imputer'].transform(raw_numeric).astype('float32')
gender=prep['gender_encoder'].transform(validation[[prep['categorical_feature']]].astype('string')).astype('float32')
matrix=np.column_stack([num,gender]).astype('float32')
names=list(prep['transformed_features']);manifest_names=manifest.sort_values('transformed_position')['transformed_feature'].astype(str).tolist()
if matrix.shape!=(EXPECTED,21) or not np.isfinite(matrix).all() or names!=manifest_names:raise ValueError('Locked Validation transformation failed.')
score=np.asarray(model.predict_proba(matrix)[:,1],float)
if not np.isfinite(score).all() or np.any((score<0)|(score>1)):raise ValueError('Invalid Validation scores.')

context={'step':'5.12N7','status':'OUTCOME_BLIND_LOCKED_PULSCAR_MEAN_XGB_VALIDATION_SCORING',
 'n6_completion_sha256':sha(N6_FILE),'n6b_completion_sha256':sha(N6B_FILE),'model_sha256':sha(MODEL),
 'preprocessor_sha256':sha(PREP),'manifest_sha256':sha(MANIFEST),'validation_predictors':str(VALIDATION_FILE),
 'validation_predictors_sha256':sha(VALIDATION_FILE),'features':features,'numeric_missing_policy':'locked Training-mean imputation; no indicators; no NaN reaches XGBoost',
 'validation_outcomes_loaded':False,'validation_metrics_calculated':False,'threshold_selected':False,'priority_categories_selected':False,'test_accessed':False}
cid=hashlib.sha256(json.dumps(context,sort_keys=True).encode()).hexdigest()[:24];OUT=OUT_ROOT/cid;PRIV=OUT/'PRIVATE_INTERNAL_DO_NOT_SHARE';PRIV.mkdir(parents=True,exist_ok=True);atomic(OUT/'context.json',context)
SCORES=PRIV/'locked_pulscar_mean_xgb_validation_scores.parquet';AVAIL=PRIV/'validation_original_feature_availability.parquet'
tmp=SCORES.with_suffix('.tmp.parquet');pd.DataFrame({ID:validation[ID].to_numpy(),'locked_pulscar_mean_xgb_validation_score':score}).to_parquet(tmp,index=False);os.replace(tmp,SCORES)
tmp=AVAIL.with_suffix('.tmp.parquet');availability.to_parquet(tmp,index=False);os.replace(tmp,AVAIL)
summary=pd.DataFrame({'feature':features,'observed_patients':[int(availability[x].sum()) for x in features],'observed_percent':[100*float(availability[x].mean()) for x in features]});summary['missing_patients']=EXPECTED-summary['observed_patients'];summary['missing_percent']=100-summary['observed_percent'];SUMMARY=OUT/'validation_predictor_availability_summary.csv';summary.to_csv(SUMMARY,index=False)
distribution={'patients_scored':EXPECTED,'minimum':float(score.min()),'p01':float(np.quantile(score,.01)),'p05':float(np.quantile(score,.05)),'median':float(np.median(score)),'mean':float(score.mean()),'p95':float(np.quantile(score,.95)),'p99':float(np.quantile(score,.99)),'maximum':float(score.max())}
completion={**context,'status':'OUTCOME_BLIND_LOCKED_PULSCAR_MEAN_XGB_VALIDATION_SCORING_COMPLETE','completed_at_utc':datetime.now(timezone.utc).isoformat(),'output_dir':str(OUT),'score_file':str(SCORES),'score_file_sha256':sha(SCORES),'availability_file':str(AVAIL),'availability_file_sha256':sha(AVAIL),'availability_summary_file':str(SUMMARY),'availability_summary_sha256':sha(SUMMARY),'score_distribution':distribution,'model_changed':False,'next_step':'Open Validation outcomes once; characterize and lock rank-based workload before Test.'};atomic(OUT/'completion.json',completion)
print('='*112);print('STEP 5.12N7 — OUTCOME-BLIND LOCKED VALIDATION SCORING COMPLETE');print('\nSCORE DISTRIBUTION');print(json.dumps(distribution,indent=2));print('\nScore file:',SCORES);print('SHA256:',completion['score_file_sha256']);print('Availability:',AVAIL);print('Validation outcomes loaded: No');print('Validation metrics calculated: No');print('Threshold selected: No');print('Test accessed: No');print('Model changed: No')
