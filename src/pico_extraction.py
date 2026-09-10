import json, re, spacy, networkx as nx, csv, os, sys
from collections import Counter, defaultdict
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Auto-detect corpus: prefer 300-abstract version if available
CORPUS_FILE = sys.argv[1] if len(sys.argv) > 1 else None
if not CORPUS_FILE:
    for f in ["sleep_corpus_300.json", "sleep_corpus.json"]:
        if os.path.exists(f):
            CORPUS_FILE = f; break
        fp = os.path.join(os.path.dirname(__file__) or ".", f)
        if os.path.exists(fp):
            CORPUS_FILE = fp; break
if not CORPUS_FILE:
    print("ERROR: No corpus file found. Run fetch_pubmed_sleep.py first."); sys.exit(1)
with open(CORPUS_FILE) as f:
    corpus = json.load(f)
print(f"=== SleepGraph Pipeline ===\nCorpus: {CORPUS_FILE}\nLoaded {len(corpus)} abstracts")
print(f"Study types: {Counter(a['study_type'] for a in corpus)}")

INTERVENTION_SYNONYMS = {
    r'CBT[\-\s]?I': 'CBT-I',
    r'cognitive\s*behavio(?:u)?ral\s*therapy\s*(?:for\s*insomnia)?': 'CBT-I',
    r'(?:digital|online|internet|app|web)[\-\s](?:delivered\s*)?(?:CBT[\-\s]?I|cognitive\s*behavio)': 'digital CBT-I',
    r'dCBT[\-\s]?I': 'digital CBT-I',
    r'eCBT[\-\s]?I': 'digital CBT-I',
    r'SHUTi': 'digital CBT-I',
    r'face[\-\s]to[\-\s]face\s*CBT': 'face-to-face CBT-I',
    r'stimulus\s*control': 'stimulus control',
    r'sleep\s*restriction': 'sleep restriction therapy',
    r'sleep\s*hygiene': 'sleep hygiene education',
    r'relaxation\s*(?:therapy|techniques?|training)': 'relaxation therapy',
    r'cognitive\s*(?:therapy|techniques?|restructur)': 'cognitive therapy',
    r'psychoeducation': 'psychoeducation',
    r'melatonin': 'melatonin',
    r'prolonged[\-\s]release\s*melatonin': 'prolonged-release melatonin',
    r'PedPRM': 'prolonged-release melatonin',
    r'immediate[\-\s]release\s*melatonin': 'immediate-release melatonin',
    r'exo(?:genous)?\s*melatonin': 'exogenous melatonin',
    r'ramelteon': 'ramelteon',
    r'tasimelteon': 'tasimelteon',
    r'suvorexant': 'suvorexant',
    r'benzodiazepine': 'benzodiazepines',
    r'pharmacol(?:ogical|therapy)': 'pharmacotherapy',
    r'exercise': 'exercise',
    r'light\s*(?:therapy|exposure)': 'light therapy',
    r'hypnotherapy': 'hypnotherapy',
    r'meditat(?:ive|ion)': 'meditation/mindfulness',
    r'patient\s*education': 'patient education',
    r'placebo': 'placebo',
}

OUTCOME_SYNONYMS = {
    r'insomnia\s*severity(?:\s*index)?': 'insomnia severity (ISI)',
    r'ISI': 'insomnia severity (ISI)',
    r'(?:Pittsburgh\s*)?sleep\s*quality(?:\s*index)?': 'sleep quality (PSQI)',
    r'PSQI': 'sleep quality (PSQI)',
    r'sleep\s*onset\s*latency': 'sleep onset latency',
    r'SOL': 'sleep onset latency',
    r'total\s*sleep\s*time': 'total sleep time',
    r'TST': 'total sleep time',
    r'sleep\s*efficiency': 'sleep efficiency',
    r'wake\s*after\s*sleep\s*onset': 'wake after sleep onset',
    r'WASO': 'wake after sleep onset',
    r'(?:number\s*of\s*)?nocturnal\s*awakening': 'nocturnal awakenings',
    r'sleep\s*onset\s*time': 'sleep onset time',
    r'dim\s*light\s*melatonin\s*onset': 'dim light melatonin onset',
    r'DLMO': 'dim light melatonin onset',
    r'polysomnograph': 'polysomnography',
    r'actigraph': 'actigraphy',
    r'depression': 'depression',
    r'anxiety': 'anxiety',
    r'fatigue': 'fatigue',
    r'quality\s*of\s*life': 'quality of life',
    r'QoL': 'quality of life',
    r'cognitive\s*performance': 'cognitive function',
    r'daytime\s*(?:function|sleepiness)': 'daytime functioning',
    r'sleep\s*hygiene': 'sleep hygiene behaviour',
    r'adverse\s*(?:event|effect)': 'adverse events',
    r'safety': 'safety',
    r'dropout': 'dropout rate',
    r'chronotype': 'chronotype',
    r'mental\s*health': 'mental health',
    r'pain': 'pain',
}

POPULATION_PATTERNS = [
    (r'adults?\s*(?:with|suffering)', 'adults with insomnia'),
    (r'older\s*adults?', 'older adults'),
    (r'elderly', 'older adults'),
    (r'(?:ages?\s*)?55[\-\s](?:to\s*)?95', 'older adults'),
    (r'adolescents?', 'adolescents'),
    (r'children\s*(?:and\s*adolescents?)?', 'children/adolescents'),
    (r'young\s*(?:people|individuals|adults)', 'young adults'),
    (r'college\s*students?', 'young adults'),
    (r'women|female', 'women'),
    (r'menopausal', 'menopausal women'),
    (r'breast\s*cancer', 'cancer survivors'),
    (r'multiple\s*sclerosis|MS\b', 'MS patients'),
    (r'autism|ASD', 'ASD population'),
    (r'ADHD', 'ADHD population'),
    (r'Parkinson', 'Parkinson patients'),
    (r'Alzheimer|dementia', 'dementia patients'),
    (r'epilepsy', 'epilepsy patients'),
    (r'chronic\s*insomnia', 'chronic insomnia'),
    (r'primary\s*(?:sleep\s*)?disorder', 'primary sleep disorder'),
    (r'delayed\s*sleep\s*phase', 'delayed sleep phase'),
    (r'comorbid\s*insomnia', 'comorbid insomnia'),
    (r'insomnia\s*(?:disorder|patients?|symptoms?)', 'insomnia patients'),
]

def extract_interventions(text):
    found = set()
    for p, c in INTERVENTION_SYNONYMS.items():
        if re.search(p, text, re.IGNORECASE): found.add(c)
    return list(found)

def extract_outcomes(text):
    found = set()
    for p, c in OUTCOME_SYNONYMS.items():
        if re.search(p, text, re.IGNORECASE): found.add(c)
    return list(found)

def extract_populations(text):
    pops = set()
    for p, label in POPULATION_PATTERNS:
        if re.search(p, text, re.IGNORECASE): pops.add(label)
    if not pops: pops.add('insomnia patients')
    return list(pops)

def extract_comparators(text):
    comps = set()
    for p, c in [
        (r'placebo', 'placebo'),
        (r'control\s*group', 'control group'),
        (r'waitlist|wait[\-\s]list', 'waitlist'),
        (r'patient\s*education', 'patient education'),
        (r'(?:inactive|non[\-\s]active)\s*(?:control|comparator)', 'inactive control'),
        (r'sleep\s*hygiene\s*(?:education|alone)', 'sleep hygiene only'),
    ]:
        if re.search(p, text, re.IGNORECASE): comps.add(c)
    return list(comps)

# Run extraction
all_triples = []
extraction_results = []
for a in corpus:
    text = a['title'] + " " + a['abstract']
    intv = extract_interventions(text)
    out = extract_outcomes(text)
    pop = extract_populations(text)
    comp = extract_comparators(text)
    extraction_results.append({'pmid':a['pmid'],'title':a['title'],'year':a['year'],
        'study_type':a['study_type'],'interventions':intv,'outcomes':out,'populations':pop,'comparators':comp})
    for i in intv:
        for o in out:
            for pp in pop:
                all_triples.append({'population':pp,'intervention':i,'outcome':o,
                    'comparator':comp[0] if comp else 'not specified','pmid':a['pmid'],'year':a['year'],'study_type':a['study_type']})

print(f"\nExtracted {len(all_triples)} PICO triples")
print(f"Unique interventions: {len(set(t['intervention'] for t in all_triples))}")
print(f"Unique outcomes: {len(set(t['outcome'] for t in all_triples))}")
print(f"Unique populations: {len(set(t['population'] for t in all_triples))}")

# Build knowledge graph
G = nx.DiGraph()
for t in all_triples:
    pn,inn,on,cn = f"POP:{t['population']}",f"INT:{t['intervention']}",f"OUT:{t['outcome']}",f"CMP:{t['comparator']}"
    G.add_node(pn,entity_type='Population',label=t['population'])
    G.add_node(inn,entity_type='Intervention',label=t['intervention'])
    G.add_node(on,entity_type='Outcome',label=t['outcome'])
    G.add_node(cn,entity_type='Comparator',label=t['comparator'])
    for u,v,rel in [(inn,pn,'TESTED_IN'),(inn,on,'MEASURED_BY')]:
        if G.has_edge(u,v): G[u][v]['weight']+=1
        else: G.add_edge(u,v,relation=rel,weight=1)
    if t['comparator']!='not specified':
        if G.has_edge(inn,cn): G[inn][cn]['weight']+=1
        else: G.add_edge(inn,cn,relation='COMPARED_WITH',weight=1)

node_types = Counter(d.get('entity_type','?') for _,d in G.nodes(data=True))
edge_types = Counter(d.get('relation','?') for _,_,d in G.edges(data=True))
degrees = [d for _,d in G.degree()]
print(f"\nKnowledge Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
print(f"Node types: {dict(node_types)}")
print(f"Edge types: {dict(edge_types)}")
print(f"Mean degree: {np.mean(degrees):.2f}, Max: {max(degrees)}")
top = sorted(G.degree(),key=lambda x:x[1],reverse=True)[:10]
print("Top nodes:")
for n,d in top: print(f"  {G.nodes[n].get('label',n)} ({G.nodes[n].get('entity_type','?')}): {d}")

# Evidence matrix
all_i = sorted(set(t['intervention'] for t in all_triples))
all_o = sorted(set(t['outcome'] for t in all_triples))
em = pd.DataFrame(0,index=all_i,columns=all_o)
for t in all_triples: em.loc[t['intervention'],t['outcome']]+=1
total_cells = em.shape[0]*em.shape[1]
zero = (em==0).sum().sum()
print(f"\nEvidence matrix: {em.shape[0]}x{em.shape[1]}, populated: {total_cells-zero} ({(total_cells-zero)/total_cells*100:.1f}%), gaps: {zero} ({zero/total_cells*100:.1f}%)")

# Comparator stats
no_comp = sum(1 for r in extraction_results if not r['comparators'])
print(f"Abstracts without explicit comparator: {no_comp}/{len(corpus)} ({no_comp/len(corpus)*100:.1f}%)")

# Figures
fig,ax=plt.subplots(figsize=(18,12))
short_i={i:i[:25] for i in all_i}; short_o={o:o[:18] for o in all_o}
pm=em.copy(); pm.index=[short_i[i] for i in pm.index]; pm.columns=[short_o[o] for o in pm.columns]
sns.heatmap(pm,annot=True,fmt='d',cmap='YlOrRd',linewidths=0.5,ax=ax,cbar_kws={'label':'Evidence triples'})
ax.set_title('Evidence Density: Intervention × Outcome Matrix (Sleep Disorders)',fontsize=14,fontweight='bold')
plt.xticks(rotation=45,ha='right',fontsize=8); plt.yticks(fontsize=9); plt.tight_layout()
plt.savefig('sleep_fig1_heatmap.png',dpi=150,bbox_inches='tight'); plt.close()

fig,ax=plt.subplots(figsize=(12,6))
ic=Counter(t['intervention'] for t in all_triples)
idf=pd.DataFrame(ic.most_common(),columns=['Intervention','Count'])
ax.barh(idf['Intervention'],idf['Count'],color=sns.color_palette('viridis',len(idf)))
ax.set_xlabel('Evidence triples'); ax.set_title('Distribution by Intervention',fontsize=14,fontweight='bold')
ax.invert_yaxis(); plt.tight_layout()
plt.savefig('sleep_fig2_interventions.png',dpi=150,bbox_inches='tight'); plt.close()

fig,ax=plt.subplots(figsize=(16,12))
cmap={'Intervention':'#e74c3c','Population':'#3498db','Outcome':'#2ecc71','Comparator':'#f39c12'}
nc=[cmap.get(G.nodes[n].get('entity_type','?'),'#95a5a6') for n in G.nodes()]
ns=[max(100,G.degree(n)*30) for n in G.nodes()]
pos=nx.spring_layout(G,k=2,iterations=50,seed=42)
nx.draw_networkx_edges(G,pos,alpha=0.15,edge_color='gray',ax=ax)
nx.draw_networkx_nodes(G,pos,node_color=nc,node_size=ns,alpha=0.8,ax=ax)
labels={n:G.nodes[n].get('label',n)[:22] for n in G.nodes() if G.degree(n)>=3}
nx.draw_networkx_labels(G,pos,labels,font_size=7,ax=ax)
from matplotlib.lines import Line2D
ax.legend(handles=[Line2D([0],[0],marker='o',color='w',markerfacecolor=c,markersize=12,label=t) for t,c in cmap.items()],loc='upper left')
ax.set_title('SleepGraph Knowledge Graph',fontsize=16,fontweight='bold'); ax.axis('off')
plt.tight_layout(); plt.savefig('sleep_fig3_graph.png',dpi=150,bbox_inches='tight'); plt.close()

# Save outputs
em.to_csv('sleep_evidence_matrix.csv')
with open('sleep_triples.csv','w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=['pmid','year','study_type','population','intervention','outcome','comparator'])
    w.writeheader(); w.writerows(all_triples)
with open('sleep_extraction.json','w') as f: json.dump(extraction_results,f,indent=2)
G2=G.copy()
for u,v,d in G2.edges(data=True):
    for k,val in list(d.items()):
        if isinstance(val,list): d[k]=';'.join(str(x) for x in val)
nx.write_graphml(G2,'sleep_graph.graphml')

# Summary
ui=set(); uo=set()
for r in extraction_results: ui.update(r['interventions']); uo.update(r['outcomes'])
print(f"\n{'='*60}\nSUMMARY FOR PAPER\n{'='*60}")
rct=sum(1 for a in corpus if a['study_type']=='RCT')
ma=sum(1 for a in corpus if a['study_type'] in ('meta-analysis','review'))
print(f"Corpus: {len(corpus)} abstracts, {rct} RCTs ({rct/len(corpus)*100:.1f}%), {ma} reviews/MAs ({ma/len(corpus)*100:.1f}%)")
print(f"Triples: {len(all_triples)}, Interventions: {len(ui)}, Outcomes: {len(uo)}")
print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
print(f"Matrix: {em.shape[0]}x{em.shape[1]}, {(total_cells-zero)/total_cells*100:.1f}% populated, {zero/total_cells*100:.1f}% gaps")
print(f"No comparator: {no_comp}/{len(corpus)} ({no_comp/len(corpus)*100:.1f}%)")
print(f"\nInterventions: {sorted(ui)}")
print(f"\nOutcomes: {sorted(uo)}")
print("\nDone.")
