"""
EvidenceGraphLP: Link Prediction Pipeline
==========================================
Matches the manuscript exactly:
- 9 features (6 degree-derived, 3 bipartite topology)
- No weighted_paths (identically zero on bipartite graphs)
- No preferential attachment (algebraically redundant with degree product)
- Temporal hold-out with bootstrap 95% CIs
- Negative sampling sensitivity analysis

Usage:  cd src && python link_prediction.py
Reads:  ../data/sleep_corpus.json
Writes: ../results/ and ../figures/
"""
import json,re,os,sys,warnings
import numpy as np; import pandas as pd; import networkx as nx; import spacy
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.metrics import roc_auc_score,average_precision_score,f1_score
from sklearn.model_selection import StratifiedKFold,cross_val_score
import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt;import seaborn as sns
warnings.filterwarnings('ignore');np.random.seed(42)
BASE=os.path.dirname(os.path.abspath(__file__))
DATA=os.path.join(BASE,'..','data','sleep_corpus.json')
RES=os.path.join(BASE,'..','results');FIG=os.path.join(BASE,'..','figures')
os.makedirs(RES,exist_ok=True);os.makedirs(FIG,exist_ok=True)
with open(DATA) as f: corpus=json.load(f)
print(f"Corpus: {len(corpus)} abstracts, years {min(a['year'] for a in corpus)}-{max(a['year'] for a in corpus)}")
nlp=spacy.load("en_core_web_sm")
INTV={r'CBT[\-\s]?I\b':'CBT-I',r'cognitive\s*behavio(?:u)?ral\s*therapy':'CBT-I',r'(?:digital|online|internet|app)[\-\s](?:delivered\s*)?(?:CBT|cognitive)':'digital_CBT-I',r'dCBT[\-\s]?I':'digital_CBT-I',r'stimulus\s*control':'stimulus_control',r'sleep\s*restriction':'sleep_restriction',r'sleep\s*hygiene':'sleep_hygiene',r'relaxation\s*(?:therapy|techniques?|training)':'relaxation',r'cognitive\s*(?:therapy|restructur)':'cognitive_therapy',r'psychoeducation':'psychoeducation',r'melatonin':'melatonin',r'ramelteon':'ramelteon',r'suvorexant':'suvorexant',r'zolpidem':'zolpidem',r'eszopiclone':'eszopiclone',r'benzodiazepine':'benzodiazepines',r'trazodone':'trazodone',r'exercise|physical\s*activity':'exercise',r'(?:bright\s*)?light\s*(?:therapy|exposure)':'light_therapy',r'mindfulness|meditation':'mindfulness',r'acupuncture':'acupuncture',r'yoga':'yoga',r'valerian':'valerian',r'tai\s*chi':'tai_chi',r'music\s*therapy':'music_therapy',r'massage':'massage_therapy',r'aromatherapy':'aromatherapy'}
OUTC={r'insomnia\s*severity|ISI\b':'insomnia_severity',r'sleep\s*quality|PSQI\b|Pittsburgh':'sleep_quality',r'sleep\s*onset\s*latency|SOL\b':'sleep_onset_latency',r'total\s*sleep\s*time|TST\b':'total_sleep_time',r'sleep\s*efficiency':'sleep_efficiency',r'wake\s*after\s*sleep\s*onset|WASO\b':'WASO',r'nocturnal\s*awakening':'nocturnal_awakenings',r'polysomnogra':'polysomnography',r'actigraph':'actigraphy',r'depression':'depression',r'anxiety':'anxiety',r'fatigue':'fatigue',r'quality\s*of\s*life':'quality_of_life',r'daytime\s*(?:function|sleepiness)|Epworth':'daytime_functioning',r'cognitive\s*(?:function|performance)':'cognitive_function',r'pain\b':'pain',r'adverse\s*(?:event|effect)':'adverse_events',r'safety':'safety',r'blood\s*pressure|hypertension':'blood_pressure',r'mood':'mood',r'stress':'stress',r'sleep\s*(?:architecture|stage|REM|NREM)':'sleep_architecture',r'(?:body\s*mass|BMI|weight)':'body_composition'}
def extract(text):
    iv,oc=set(),set()
    for p,c in INTV.items():
        if re.search(p,text,re.IGNORECASE):iv.add(c)
    for p,c in OUTC.items():
        if re.search(p,text,re.IGNORECASE):oc.add(c)
    doc=nlp(text[:2000])
    for tk in doc:
        if tk.lemma_ in('treat','administer','receive','prescribe','assign') and tk.dep_=='ROOT':
            for ch in tk.children:
                if ch.dep_ in('dobj','pobj','prep'):
                    sp=doc[ch.i:min(ch.i+3,len(doc))].text.lower()
                    for p,c in INTV.items():
                        if re.search(p,sp,re.IGNORECASE):iv.add(c)
        if tk.lemma_ in('measure','assess','evaluate','monitor') and tk.dep_=='ROOT':
            for ch in tk.children:
                if ch.dep_ in('dobj','pobj','nsubjpass'):
                    sp=doc[ch.i:min(ch.i+4,len(doc))].text.lower()
                    for p,c in OUTC.items():
                        if re.search(p,sp,re.IGNORECASE):oc.add(c)
    iv.discard('placebo');iv.discard('waitlist')
    return list(iv),list(oc)
print("Extracting...")
recs=[]
for a in corpus:
    iv,oc=extract(a['title']+" "+a['abstract'])
    recs.append({'pmid':a['pmid'],'year':a['year'],'interventions':iv,'outcomes':oc})
with open(os.path.join(RES,'extraction_results.json'),'w') as f:json.dump(recs,f,indent=2)
SPLIT=2014
def mkgraph(rs):
    G=nx.Graph()
    for r in rs:
        for i in r['interventions']:
            for o in r['outcomes']:
                u,v=f"I:{i}",f"O:{o}";G.add_node(u,node_type='intervention');G.add_node(v,node_type='outcome')
                if G.has_edge(u,v):G[u][v]['weight']+=1
                else:G.add_edge(u,v,weight=1)
    return G
Gtr=mkgraph([r for r in recs if r['year']<=SPLIT])
Gfu=mkgraph(recs)
print(f"Train graph: {Gtr.number_of_nodes()} nodes, {Gtr.number_of_edges()} edges")
print(f"Full graph: {Gfu.number_of_nodes()} nodes, {Gfu.number_of_edges()} edges")
new_e=set(Gfu.edges())-set(Gtr.edges())
itr=[n for n,d in Gtr.nodes(data=True) if d.get('node_type')=='intervention']
otr=[n for n,d in Gtr.nodes(data=True) if d.get('node_type')=='outcome']
pos_test=[(u,v) for u,v in new_e if u in Gtr and v in Gtr]
allp=set((u,v) for u in itr for v in otr)
fe=set(Gfu.edges())|{(v,u) for u,v in Gfu.edges()}
neg_pool=list(allp-fe)
print(f"Test positive: {len(pos_test)}, Negative pool: {len(neg_pool)}")
FEATS=['deg_u','deg_v','deg_product','deg_sum','deg_diff','deg_ratio','shared_intermediate','projected_deg_u','projected_deg_v']
def feats(G,u,v):
    f={};f['deg_u']=G.degree(u) if u in G else 0;f['deg_v']=G.degree(v) if v in G else 0
    f['deg_product']=f['deg_u']*f['deg_v'];f['deg_sum']=f['deg_u']+f['deg_v']
    f['deg_diff']=abs(f['deg_u']-f['deg_v']);f['deg_ratio']=f['deg_u']/(f['deg_v']+1)
    if u in G and v in G:
        un,vn=set(G.neighbors(u)),set(G.neighbors(v))
        sh=sum(1 for n in un for nn in G.neighbors(n) if nn!=u and nn in vn)
        f['shared_intermediate']=sh
        f['projected_deg_u']=len({nn for n in un for nn in G.neighbors(n) if nn!=u})
        f['projected_deg_v']=len({nn for n in vn for nn in G.neighbors(n) if nn!=v})
    else:
        f['shared_intermediate']=0;f['projected_deg_u']=0;f['projected_deg_v']=0
    return f
def mkdata(G,pos,neg):
    X,y=[],[]
    for u,v in pos:
        had=G.has_edge(u,v)
        if had:w=G[u][v].get('weight',1);G.remove_edge(u,v)
        X.append(feats(G,u,v));y.append(1)
        if had:G.add_edge(u,v,weight=w)
    for u,v in neg:X.append(feats(G,u,v));y.append(0)
    return pd.DataFrame(X)[FEATS],np.array(y)
ni=np.random.choice(len(neg_pool),size=min(len(pos_test)*3,len(neg_pool)),replace=False)
def_neg=[neg_pool[i] for i in ni]
tp=list(Gtr.edges());tnp=list(allp-set(Gtr.edges())-{(v,u) for u,v in Gtr.edges()})
tni=np.random.choice(len(tnp),size=min(len(tp)*2,len(tnp)),replace=False)
tn=[tnp[i] for i in tni]
Xtr,ytr=mkdata(Gtr.copy(),tp,tn);Xte,yte=mkdata(Gtr.copy(),pos_test,def_neg)
print(f"Train: {Xtr.shape}, Test: {Xte.shape}")
def bci(yt,ys,fn,nb=2000):
    sc=[fn(yt[i:=np.random.choice(len(yt),len(yt),replace=True)],ys[i]) for _ in range(nb) if len(set(yt[i:=np.random.choice(len(yt),len(yt),replace=True)]))>1]
    return np.percentile(sc,2.5),np.percentile(sc,97.5)
print(f"\n{'='*60}\nMODEL COMPARISON + Bootstrap 95% CIs\n{'='*60}")
mdls={'Logistic Regression':LogisticRegression(max_iter=1000,random_state=42,class_weight='balanced'),'Random Forest':RandomForestClassifier(n_estimators=200,random_state=42,n_jobs=-1,class_weight='balanced'),'Gradient Boosting':GradientBoostingClassifier(n_estimators=200,random_state=42,max_depth=4)}
res={}
for nm,md in mdls.items():
    md.fit(Xtr,ytr);yp=md.predict(Xte);ypr=md.predict_proba(Xte)[:,1]
    a=roc_auc_score(yte,ypr);ap=average_precision_score(yte,ypr);f1=f1_score(yte,yp,zero_division=0)
    al,ah=bci(yte,ypr,roc_auc_score);apl,aph=bci(yte,ypr,average_precision_score)
    cv=cross_val_score(md,Xtr,ytr,cv=StratifiedKFold(5,shuffle=True,random_state=42),scoring='roc_auc').mean()
    res[nm]={'AUC':a,'AUC_CI':f"{al:.3f}-{ah:.3f}",'AP':ap,'AP_CI':f"{apl:.3f}-{aph:.3f}",'F1':f1,'CV':cv}
    print(f"\n  {nm}: AUC={a:.3f} (95% CI: {al:.3f}-{ah:.3f}), AP={ap:.3f} (95% CI: {apl:.3f}-{aph:.3f}), F1={f1:.3f}, CV={cv:.3f}")
print(f"\n--- Baselines ---")
for bn,col in[('Degree Product','deg_product'),('Shared Intermediate','shared_intermediate')]:
    sc=Xte[col].values;a=roc_auc_score(yte,sc) if sc.max()>sc.min() else 0.5
    al,ah=bci(yte,sc,roc_auc_score) if sc.max()>sc.min() else (0.5,0.5)
    print(f"  {bn}: AUC={a:.3f} (95% CI: {al:.3f}-{ah:.3f})")
print(f"\n--- Ablation ---")
abl={'Degree only (6)':FEATS[:6],'Bipartite topology (3)':FEATS[6:],'All (9)':FEATS}
for gn,ft in abl.items():
    m=GradientBoostingClassifier(n_estimators=200,random_state=42,max_depth=4);m.fit(Xtr[ft],ytr)
    yp=m.predict_proba(Xte[ft])[:,1];a=roc_auc_score(yte,yp);al,ah=bci(yte,yp,roc_auc_score)
    print(f"  {gn:<30} AUC={a:.3f} (95% CI: {al:.3f}-{ah:.3f})")
print(f"\n--- Negative Sampling Sensitivity ---")
gb=mdls['Gradient Boosting']
for rl,r in[('1:1',1),('1:2.6',2.6),('1:5',5)]:
    aucs=[]
    for s in range(5):
        rng=np.random.RandomState(s);nn=min(int(len(pos_test)*r),len(neg_pool))
        ni2=rng.choice(len(neg_pool),size=nn,replace=False);ng=[neg_pool[i] for i in ni2]
        Xt2,yt2=mkdata(Gtr.copy(),pos_test,ng);yp2=gb.predict_proba(Xt2)[:,1]
        aucs.append(roc_auc_score(yt2,yp2))
    print(f"  Ratio {rl}: AUC mean={np.mean(aucs):.3f}, std={np.std(aucs):.3f}, range={min(aucs):.3f}-{max(aucs):.3f}")
print(f"\n--- Top 5 Gap Predictions ---")
gaps=[]
for u in itr:
    for v in otr:
        if not Gfu.has_edge(u,v):
            fv=pd.DataFrame([feats(Gtr,u,v)])[FEATS]
            gaps.append({'intervention':u[2:],'outcome':v[2:],'link_score':gb.predict_proba(fv)[0][1]})
gdf=pd.DataFrame(gaps).sort_values('link_score',ascending=False)
gdf.to_csv(os.path.join(RES,'gap_predictions.csv'),index=False)
for _,r in gdf.head(5).iterrows():print(f"  {r['intervention']:<25} -> {r['outcome']:<25} {r['link_score']:.4f}")
print("\nGenerating figures...")
fig,axes=plt.subplots(1,2,figsize=(14,5))
am={n:r['AUC'] for n,r in res.items()};am['Degree Product']=roc_auc_score(yte,Xte['deg_product'].values) if Xte['deg_product'].max()>Xte['deg_product'].min() else 0.5
axes[0].barh(list(am.keys()),list(am.values()),color=['#e74c3c' if m in res else '#3498db' for m in am])
axes[0].set_xlabel('AUC-ROC');axes[0].set_title('(a) Temporal Link Prediction',fontweight='bold');axes[0].axvline(0.5,color='gray',ls='--',alpha=0.5);axes[0].set_xlim(0.3,1)
an=list(abl.keys());aa=[]
for ft in abl.values():
    m=GradientBoostingClassifier(n_estimators=200,random_state=42,max_depth=4);m.fit(Xtr[ft],ytr);aa.append(roc_auc_score(yte,m.predict_proba(Xte[ft])[:,1]))
axes[1].barh(an,aa,color=sns.color_palette('viridis',len(an)));axes[1].set_xlabel('AUC-ROC');axes[1].set_title('(b) Ablation',fontweight='bold');axes[1].set_xlim(0.3,1)
plt.tight_layout();plt.savefig(os.path.join(FIG,'fig1_model_comparison.png'),dpi=150,bbox_inches='tight');plt.close()
gb2=mdls['Gradient Boosting']
if hasattr(gb2,'feature_importances_'):
    fi=pd.DataFrame({'feature':FEATS,'importance':gb2.feature_importances_}).sort_values('importance',ascending=False)
    fig,ax=plt.subplots(figsize=(10,6));ax.barh(fi['feature'],fi['importance'],color='steelblue');ax.set_xlabel('Importance');ax.set_title('Feature Importance (Gradient Boosting)',fontweight='bold');ax.invert_yaxis();plt.tight_layout();plt.savefig(os.path.join(FIG,'fig2_feature_importance.png'),dpi=150,bbox_inches='tight');plt.close()
gm=gdf.pivot_table(index='intervention',columns='outcome',values='link_score',aggfunc='first').fillna(0)
fig,ax=plt.subplots(figsize=(16,10));sns.heatmap(gm,cmap='RdYlGn',annot=False,ax=ax,cbar_kws={'label':'Link score'});ax.set_title('Evidence Gap Scores',fontweight='bold');plt.xticks(rotation=45,ha='right',fontsize=7);plt.yticks(fontsize=8);plt.tight_layout();plt.savefig(os.path.join(FIG,'fig3_gap_predictions.png'),dpi=150,bbox_inches='tight');plt.close()
fig,ax=plt.subplots(figsize=(14,10));cm={'intervention':'#e74c3c','outcome':'#2ecc71'}
nc=[cm.get(Gfu.nodes[n].get('node_type',''),'#999') for n in Gfu.nodes()];ns=[max(80,Gfu.degree(n)*35) for n in Gfu.nodes()]
pos2=nx.spring_layout(Gfu,k=1.5,iterations=50,seed=42);nx.draw_networkx_edges(Gfu,pos2,alpha=0.12,edge_color='gray',ax=ax);nx.draw_networkx_nodes(Gfu,pos2,node_color=nc,node_size=ns,alpha=0.8,ax=ax)
lb={n:n.split(':')[1][:18] for n in Gfu.nodes() if Gfu.degree(n)>=3};nx.draw_networkx_labels(Gfu,pos2,lb,font_size=6,ax=ax)
from matplotlib.lines import Line2D
ax.legend(handles=[Line2D([0],[0],marker='o',color='w',markerfacecolor=c,markersize=12,label=t.title()) for t,c in cm.items()],loc='upper left')
ax.set_title('Bipartite Evidence Knowledge Graph',fontweight='bold');ax.axis('off');plt.tight_layout();plt.savefig(os.path.join(FIG,'fig4_knowledge_graph.png'),dpi=150,bbox_inches='tight');plt.close()
nx.write_graphml(Gfu,os.path.join(RES,'knowledge_graph.graphml'))
with open(os.path.join(RES,'experiment_summary.json'),'w') as f:json.dump(res,f,indent=2,default=str)
print("Done. All outputs in ../results/ and ../figures/")
