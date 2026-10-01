"""N6B: nested Training-only baseline benchmark; no Validation/Test access."""
from google.colab import drive
drive.mount('/content/drive')
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, os, tempfile, gc, time
import joblib, numpy as np, optuna, pandas as pd
from IPython.display import display
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss, log_loss

SEED=71000047; TRIALS=20; BOOTSTRAPS=500; WEIGHT_COL='TEMPORAL_BALANCING_WEIGHT'
N=69701; POS=5042; UNLAB=64659; FOLDS=5
NEW=['penalized_logistic_regression','random_forest','extra_trees','hist_gradient_boosting']
ALL=NEW+['naive_xgboost','pulscar_xgboost']
ROOT=Path('/content/drive/MyDrive/CKD_Project/outputs/section_5_modeling/step_5_9_longitudinal_primary_modeling')
N6A_ROOT=ROOT/'05_12N6A_training_only_imputation_sensitivity'
N6_ROOT=ROOT/'05_12N6_full_training_pulscar_mean_xgb_final_refit_and_lock'
OUT_ROOT=ROOT/'05_12N6B_training_only_baseline_model_benchmark'

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
def metrics(y,p):
    p=np.clip(np.asarray(p,float),1e-7,1-1e-7)
    return {'Average_precision':float(average_precision_score(y,p)),'ROC_AUC':float(roc_auc_score(y,p)),
            'Brier_score':float(brier_score_loss(y,p)),'Log_loss':float(log_loss(y,p,labels=[0,1]))}

# Lock to completed N6A and N6.
ca=[(p,loadj(p)) for p in N6A_ROOT.rglob('completion.json') if loadj(p) and loadj(p).get('status')=='SUPPLEMENTARY_TRAINING_ONLY_IMPUTATION_SENSITIVITY_COMPLETE']
if not ca:raise FileNotFoundError('Run N6A first.')
N6A_FILE,n6a=max(ca,key=lambda z:z[0].stat().st_mtime)
cn=[(p,loadj(p)) for p in N6_ROOT.rglob('completion.json') if loadj(p) and loadj(p).get('status')=='FULL_TRAINING_PULSCAR_MEAN_XGB_MODEL_LOCKED']
if not cn:raise FileNotFoundError('Run N6 first.')
N6_FILE,n6=max(cn,key=lambda z:z[0].stat().st_mtime)
if n6a.get('primary_locked_policy_remains')!='mean_imputation' or n6a.get('validation_accessed_in_this_step') or n6a.get('test_accessed_in_this_step'):raise ValueError('Invalid N6A.')
N5=loadj(Path(n6['n5_completion']));N4_FILE=Path(N5['n4_completion']);n4=loadj(N4_FILE);N4_RUN=N4_FILE.parent
N0=loadj(Path(n4['n0_policy_lock'])); SNAP=Path(n4['training_snapshot']); D0=loadj(Path(N0['source_protocol']))
ID=D0['cohort']['patient_id_column'];LABEL=D0['cohort']['observed_label_column'];FOLD=D0['cohort']['locked_fold_column']
FEATURES=list(N0['retained_features']); NUM=[x for x in FEATURES if x!='GENDER']
df=pd.read_parquet(SNAP,columns=[ID,LABEL,FOLD,WEIGHT_COL,*FEATURES])
y=pd.to_numeric(df[LABEL]).astype(int).to_numpy();w=pd.to_numeric(df[WEIGHT_COL]).to_numpy(float);w/=w.mean();folds=pd.to_numeric(df[FOLD]).astype(int).to_numpy()
if len(df)!=N or y.sum()!=POS or (y==0).sum()!=UNLAB or len(FEATURES)!=20:raise ValueError('Training lock changed.')
num=df[NUM].apply(pd.to_numeric,errors='coerce').replace([np.inf,-np.inf],np.nan); gender=df[['GENDER']].astype('string')

def prep(fit,ev,scale=False):
    imp=SimpleImputer(strategy='mean');a=imp.fit_transform(num.iloc[fit]);b=imp.transform(num.iloc[ev])
    if scale:
        s=StandardScaler();a=s.fit_transform(a);b=s.transform(b)
    enc=OneHotEncoder(handle_unknown='ignore',sparse_output=False);ga=enc.fit_transform(gender.iloc[fit]);gb=enc.transform(gender.iloc[ev])
    a=np.column_stack([a,ga]).astype('float32');b=np.column_stack([b,gb]).astype('float32')
    if not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('Nonfinite matrix.')
    return a,b
def suggest(t,m):
    if m=='penalized_logistic_regression':return {'C':t.suggest_float('C',1e-4,1e2,log=True)}
    if m in {'random_forest','extra_trees'}:return {'n_estimators':t.suggest_int('n_estimators',200,600,step=100),'max_depth':t.suggest_categorical('max_depth',[None,8,12,16,24]),'min_samples_leaf':t.suggest_int('min_samples_leaf',1,20),'max_features':t.suggest_categorical('max_features',['sqrt',0.5,1.0])}
    return {'max_iter':t.suggest_int('max_iter',100,400,step=50),'learning_rate':t.suggest_float('learning_rate',.02,.2,log=True),'max_leaf_nodes':t.suggest_int('max_leaf_nodes',15,63),'l2_regularization':t.suggest_float('l2_regularization',1e-5,10,log=True),'min_samples_leaf':t.suggest_int('min_samples_leaf',10,60)}
def model(m,p,seed):
    if m=='penalized_logistic_regression':return LogisticRegression(penalty='l2',solver='lbfgs',max_iter=2000,n_jobs=-1,random_state=seed,**p)
    if m=='random_forest':return RandomForestClassifier(n_jobs=-1,random_state=seed,**p)
    if m=='extra_trees':return ExtraTreesClassifier(n_jobs=-1,random_state=seed,**p)
    return HistGradientBoostingClassifier(random_state=seed,**p)

context={'step':'5.12N6B','status':'SUPPLEMENTARY_NESTED_TRAINING_ONLY_BASELINE_BENCHMARK','n6_sha256':sha(N6_FILE),'n6a_sha256':sha(N6A_FILE),'n4_sha256':sha(N4_FILE),'snapshot_sha256':sha(SNAP),'models':ALL,'new_model_trials_per_outer_fold':TRIALS,'features':FEATURES,'imputation':'fold-fitted mean','validation_accessed_in_this_step':False,'test_accessed_in_this_step':False,'locked_n6_model_changed':False}
cid=hashlib.sha256(json.dumps(context,sort_keys=True).encode()).hexdigest()[:24];OUT=OUT_ROOT/cid;PRIV=OUT/'PRIVATE_INTERNAL_DO_NOT_SHARE';ST=PRIV/'studies';PRED=PRIV/'predictions'
for d in [OUT,PRIV,ST,PRED]:d.mkdir(parents=True,exist_ok=True)
atomic(OUT/'context.json',context)
oof={m:np.full(N,np.nan) for m in ALL};rows=[];best=[];optuna.logging.set_verbosity(optuna.logging.WARNING)

# Reuse the exact nested tuned XGBoost comparators.
for m,source in [('naive_xgboost','naive_pn'),('pulscar_xgboost','official_pulscar')]:
    t=pd.read_parquet(N4_RUN/'PRIVATE_INTERNAL_DO_NOT_SHARE'/f'{source}__nested_tuned_complete_oof.parquet')
    if not np.array_equal(t[ID].to_numpy(),df[ID].to_numpy()):raise ValueError('N4 OOF order changed.')
    oof[m]=t['probability'].to_numpy(float)

for outer in range(1,FOLDS+1):
    fit=np.flatnonzero(folds!=outer);ev=np.flatnonzero(folds==outer)
    for m in NEW:
        cache=PRED/f'{m}_outer_{outer}.npz'
        study_name=f'{m}_outer_{outer}';db=ST/f'{study_name}.sqlite'
        study=optuna.create_study(study_name=study_name,direction='maximize',sampler=optuna.samplers.TPESampler(seed=SEED+outer+NEW.index(m)*100),storage=f'sqlite:///{db.as_posix()}',load_if_exists=True)
        bundles=[]
        for inner in [z for z in range(1,FOLDS+1) if z!=outer]:
            fi=np.flatnonzero((folds!=outer)&(folds!=inner));ei=np.flatnonzero(folds==inner);a,b=prep(fi,ei,m=='penalized_logistic_regression');bundles.append((a,b,y[fi],y[ei],w[fi]))
        def objective(trial):
            p=suggest(trial,m);scores=[]
            for j,(a,b,yt,ye,wt) in enumerate(bundles):
                clf=model(m,p,SEED+trial.number*10+j);clf.fit(a,yt,sample_weight=wt);scores.append(average_precision_score(ye,clf.predict_proba(b)[:,1]))
            return float(np.mean(scores))
        done=[t for t in study.trials if t.state==optuna.trial.TrialState.COMPLETE];remaining=TRIALS-len(done)
        if remaining>0:
            started=time.time();study.optimize(objective,n_trials=remaining,gc_after_trial=True);print(f'Completed {m}, outer {outer}, {remaining} trials, {(time.time()-started)/60:.2f} min.')
        if len([t for t in study.trials if t.state==optuna.trial.TrialState.COMPLETE])!=TRIALS:raise RuntimeError('Incomplete study; rerun.')
        p=dict(study.best_trial.params);best.append({'Method':m,'Outer_fold':outer,'Best_inner_AP':study.best_value,**p})
        if cache.is_file():pred=np.load(cache)['prediction']
        else:
            a,b=prep(fit,ev,m=='penalized_logistic_regression');clf=model(m,p,SEED+outer*1000+NEW.index(m));clf.fit(a,y[fit],sample_weight=w[fit]);pred=clf.predict_proba(b)[:,1];np.savez_compressed(cache,prediction=pred)
        oof[m][ev]=pred;rows.append({'Method':m,'Outer_fold':outer,**metrics(y[ev],pred)});del bundles;gc.collect()

summary=pd.DataFrame([{'Method':m,**metrics(y,oof[m])} for m in ALL]).sort_values('Average_precision',ascending=False).reset_index(drop=True)
pd.DataFrame({ID:df[ID].to_numpy(),**{f'{m}_oof_score':oof[m] for m in ALL}}).to_parquet(PRIV/'all_baseline_nested_oof_scores.parquet',index=False)
rng=np.random.default_rng(SEED+9000);positive=np.flatnonzero(y==1);unlab=np.flatnonzero(y==0);boot={m:[] for m in ALL if m!='pulscar_xgboost'}
for r in range(BOOTSTRAPS):
    ix=np.concatenate([rng.choice(positive,len(positive),replace=True),rng.choice(unlab,len(unlab),replace=True)]);ref=average_precision_score(y[ix],oof['pulscar_xgboost'][ix])
    for m in boot:boot[m].append(average_precision_score(y[ix],oof[m][ix])-ref)
bootstrap=pd.DataFrame([{'Method':m,'Reference':'pulscar_xgboost','Delta_AP':average_precision_score(y,oof[m])-average_precision_score(y,oof['pulscar_xgboost']),'CI_lower':np.quantile(v,.025),'CI_upper':np.quantile(v,.975),'Replicates':BOOTSTRAPS} for m,v in boot.items()])
summary.to_csv(OUT/'baseline_nested_oof_summary.csv',index=False);pd.DataFrame(rows).to_csv(OUT/'new_baseline_outer_metrics.csv',index=False);pd.DataFrame(best).to_csv(OUT/'new_baseline_best_parameters.csv',index=False);bootstrap.to_csv(OUT/'paired_bootstrap_vs_pulscar.csv',index=False)
decision={'step':'5.12N6B','status':'SUPPLEMENTARY_NESTED_TRAINING_ONLY_BASELINE_BENCHMARK_COMPLETE','locked_n6_model_changed':False,'winner_selection_permitted':False,'next_step':'Outcome-blind locked Validation scoring.','validation_accessed_in_this_step':False,'test_accessed_in_this_step':False}
atomic(OUT/'completion.json',{**context,**decision,'completed_at_utc':datetime.now(timezone.utc).isoformat(),'output_dir':str(OUT)})
print('='*112);print('STEP 5.12N6B — TRAINING-ONLY BASELINE BENCHMARK COMPLETE');print('\nOOF SUMMARY');display(summary);print('\nPAIRED BOOTSTRAP VS PULSCAR');display(bootstrap);print('\nBEST PARAMETERS');display(pd.DataFrame(best));print('\nDECISION');print(json.dumps(decision,indent=2));print('\nSaved run:',OUT);print('Validation accessed: No');print('Test accessed: No');print('Locked N6 model changed: No')
