from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd

def curve(y,s):
    y=np.asarray(y,np.uint8);s=np.asarray(s,np.float32)
    order=np.argsort(-s,kind='stable');ss=s[order];yy=y[order]
    ends=np.r_[np.flatnonzero(np.diff(ss)),len(ss)-1]
    tp=np.cumsum(yy,dtype=np.int64)[ends];fp=1+ends-tp
    return tp,fp,ss[ends]
def roc_auc_score(y,s):
    tp,fp,_=curve(y,s)
    return (np.trapezoid if hasattr(np, 'trapezoid') else np.trapz)(np.r_[0,tp/tp[-1]],np.r_[0,fp/fp[-1]])
def average_precision_score(y,s):
    tp,fp,_=curve(y,s);recall=tp/tp[-1];precision=tp/(tp+fp)
    return np.sum(np.diff(np.r_[0,recall])*precision)
def precision_recall_curve(y,s):
    tp,fp,thresholds=curve(y,s)
    return np.r_[(tp/(tp+fp))[::-1],1.],np.r_[(tp/tp[-1])[::-1],0.],thresholds[::-1]

REPO=Path(__file__).resolve().parents[1]
OUT=REPO/'verification'/'recomputed'
OUT.mkdir(parents=True,exist_ok=True)
records=[];ablation=[];oracle=[];evidence=[];sources=[]
def read(p):
    p=REPO/p
    if not p.exists() and Path(str(p)+'.gz').exists():p=Path(str(p)+'.gz')
    sources.append({'path':str(p.relative_to(REPO)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size})
    return pd.read_csv(p)
def met(y,s,t,comparison_dtype=np.float32):
    y=np.asarray(y,np.uint8);s=np.asarray(s,comparison_dtype);pred=s>=t
    tp=int(((y==1)&pred).sum());fp=int(((y==0)&pred).sum())
    tn=int(((y==0)&~pred).sum());fn=int(((y==1)&~pred).sum())
    return {'auc':float(roc_auc_score(y,s)),'ap':float(average_precision_score(y,s)),
            'precision':tp/max(tp+fp,1),'recall':tp/max(tp+fn,1),'f1':2*tp/max(2*tp+fp+fn,1),
            'fpr':fp/max(fp+tn,1),'threshold':float(t),'tp':tp,'fp':fp,'tn':tn,'fn':fn}
def quant(s):return float(np.quantile(np.asarray(s,np.float32),.75,method='linear'))
def oraclemet(y,s):
    s=np.asarray(s,np.float32);p,r,t=precision_recall_curve(y,s)
    f=2*p[:-1]*r[:-1]/np.maximum(p[:-1]+r[:-1],1e-12)
    return met(y,s,float(t[int(np.argmax(f))]))
for model in ['tcn_gan','tcn_wgan_gp']:
    for seed in range(5):
        run=f'{"pilot" if seed==0 else "formal"}_{model}_seed{seed}'
        c=read(f'journal_rebuild/runs/scores/{run}/scores_calibration.csv')
        t=read(f'journal_rebuild/runs/scores/{run}/scores_test.csv')
        cfg=json.loads((REPO/f'journal_rebuild/runs/checkpoints/{run}/resolved_config.yaml').read_text())
        saved=json.loads((REPO/f'journal_rebuild/runs/metrics/{run}/metrics.json').read_text())
        th=quant(c.fused_score);res=met(t.label,t.fused_score,th)
        errs={}
        for k,sk in [('auc','auc'),('ap','ap'),('precision','precision'),('recall','recall'),('f1','f1'),('fpr','test_benign_fpr'),('threshold','threshold')]:
            errs[k]=abs(res[k]-saved[sk])
        fusion_error=max(float(np.max(np.abs(.24*x.SD_normalized+.76*x.SF_normalized-x.fused_score))) for x in [c,t])
        p=cfg.get('research_config',cfg)
        rec={'run_id':run,'seed':seed,'model':model,'n_calibration':len(c),'n_test':len(t),
             'beta_config':p.get('calibration_protocol',{}).get('target_fpr',.25),
             'alpha_config':p.get('scoring_protocol',{}).get('alpha',p.get('model_config',{}).get('alpha')),
             'all_calibration_benign':bool((c.label==0).all()),
             'calibration_fpr_recomputed':float((np.asarray(c.fused_score,np.float32)>=th).mean()),
             'fusion_max_abs_error':fusion_error,'max_metric_abs_error':max(errs.values()),
             'test_threshold_ties':int((np.asarray(t.fused_score,np.float32)==th).sum()),
             'training_time_seconds':saved['training_time_seconds'],'inference_time_seconds':saved['inference_time_seconds'],
             **res}
        records.append(rec)
        if model=='tcn_wgan_gp':
            oracle.append({'run_id':run,'strategy':'calibration',**res})
            oracle.append({'run_id':run,'strategy':'oracle',**oraclemet(t.label,t.fused_score)})
            for alpha in [0.,.24,1.]:
                cs=alpha*c.SD_normalized+(1-alpha)*c.SF_normalized
                ts=alpha*t.SD_normalized+(1-alpha)*t.SF_normalized
                ablation.append({'run_id':run,'alpha':alpha,**met(t.label,ts,quant(cs),np.float64)})
        if run=='formal_tcn_wgan_gp_seed1':
            pred=np.asarray(t.fused_score,np.float32)>=th
            for label,name in [(0,'TN'),(1,'TP')]:
                row=t.loc[(t.label==label)&(pred==bool(label))].iloc[0].to_dict()
                row.update({'run_id':run,'selection_rule':'first row in saved file satisfying TN or TP','outcome':name,
                            'threshold':th,'score_recomputed':.24*row['SD_normalized']+.76*row['SF_normalized'],
                            'source_csv':f'journal_rebuild/runs/scores/{run}/scores_test.csv.gz'})
                evidence.append(row)
        print(run,'max error',rec['max_metric_abs_error'],flush=True)
pd.DataFrame(records).to_csv(OUT/'main_runs.csv',index=False)
metrics=['auc','ap','precision','recall','f1','fpr','threshold','training_time_seconds','inference_time_seconds']
pd.DataFrame(records).groupby('model')[metrics].agg(['mean','std']).to_csv(OUT/'main_summary.csv')
pd.DataFrame(oracle).groupby('strategy')[['threshold','precision','recall','f1','fpr']].mean().to_csv(OUT/'table6.csv')
pd.DataFrame(ablation).groupby('alpha')[['auc','ap','precision','recall','f1','fpr']].mean().to_csv(OUT/'table7.csv')
pd.DataFrame(evidence).to_csv(OUT/'table2_records.csv',index=False)
external=[]
for detector in ['ganomaly','iforest','mlp_ae','tcn_wgan_gp_case']:
    df=read(f'runs/external_protocol_check/TON_IoT/{detector}/window_scores.csv')
    c=df.loc[df.split=='independent_calibration_benign'];t=df.loc[df.split=='test']
    saved=json.loads((REPO/f'runs/external_protocol_check/TON_IoT/{detector}/metrics.json').read_text())
    th=quant(c.score_norm);res=met(t.y_true,t.score_norm,th);o=oraclemet(t.y_true,t.score_norm)
    errs=[abs(res['auc']-saved['auc']),abs(res['ap']-saved['ap']),abs(res['f1']-saved['f1_calibration']),abs(res['fpr']-saved['test_benign_fpr_calibration']),abs(o['f1']-saved['f1_oracle'])]
    external.append({'detector':detector,'n_calibration':len(c),'n_test':len(t),'all_calibration_benign':bool((c.y_true==0).all()),
                     'beta':.25,'oracle_f1':o['f1'],'max_metric_abs_error':max(errs),
                     'threshold_saved':saved['threshold_calibration'],'threshold_abs_error':abs(th-saved['threshold_calibration']),**res})
pd.DataFrame(external).to_csv(OUT/'ton_iot.csv',index=False)
baseline=read('results/current_paper_compact_baselines/cicids2017_compact_baselines.csv')
baseline=baseline.loc[baseline.method.isin(['IsolationForest','AE','GANomaly'])]
baseline.to_csv(OUT/'baseline_records.csv',index=False)
summary={'main_runs':len(records),'external_runs':len(external),'main_max_metric_error':max(x['max_metric_abs_error'] for x in records),
         'fusion_max_error':max(x['fusion_max_abs_error'] for x in records),
         'external_max_metric_error':max(x['max_metric_abs_error'] for x in external),
         'beta':.25,'alpha':.24,'main_seeds':[0,1,2,3,4],'baseline_seed':42,
         'baseline_epochs':{r.method:json.loads(r.baseline_hyperparameters).get('epochs') for _,r in baseline.iterrows()},
         'table2_original_not_traced':True,'historical_alpha_selection_not_proven':True}
(OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
(OUT/'input_fingerprints.json').write_text(json.dumps(sources,ensure_ascii=False,indent=2))
print(json.dumps(summary,ensure_ascii=False,indent=2))

assert summary["main_max_metric_error"] < 1e-6
assert summary["external_max_metric_error"] < 1e-6
assert summary["fusion_max_error"] < 1e-6
print("PASS: saved-score metric verification; this is not a full training reproduction.")
