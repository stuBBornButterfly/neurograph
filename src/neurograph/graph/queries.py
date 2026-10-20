# 10 hand-written read-only queries over the study graph; "expect" is what the earlier phases found
QUERIES = [
    {"title": "1. Participants by group and sex",
     "expect": "CTL 75 (40 F) / DEP 45 (33 F)",
     "cypher": """MATCH (p:Participant)-[:IN_DATASET]->(:Dataset {id: 'ds003478'})
RETURN p.group AS group, p.sex AS sex, count(*) AS n ORDER BY group, sex"""},
    {"title": "2. Questionnaire scores by group",
     "expect": "BDI: CTL < 7, DEP >= 13; STAI higher in DEP",
     "cypher": """MATCH (p:Participant)-[:HAS_ASSESSMENT]->(a:Assessment)
RETURN a.instrument AS instrument, p.group AS group, count(*) AS n,
       round(avg(a.score), 1) AS mean, min(a.score) AS min, max(a.score) AS max
ORDER BY instrument, group"""},
    {"title": "3. Provenance of one recording (sub-001)",
     "expect": "250 Hz, 62 channels, git hash of the Phase 2 run, ICs removed, % epochs rejected",
     "cypher": """MATCH (p:Participant {code: 'sub-001'})-[:HAS_SESSION]->(:Session)-[:HAS_RECORDING]->(r:Recording)
      -[:PROCESSED_BY]->(run:PipelineRun)
RETURN r.id AS recording, r.sfreq AS sfreq, r.n_channels AS channels, r.clean_duration_s AS clean_s,
       run.git_hash AS git_hash, run.ica_removed AS ics_removed, run.epochs_rejected_pct AS pct_rejected"""},
    {"title": "4. QC: the 5 recordings with the most rejected epochs",
     "expect": "the high-rejection flagged subjects (028, 029, 071, 085, 100), max 44.9%",
     "cypher": """MATCH (p:Participant)-[:HAS_SESSION]->()-[:HAS_RECORDING]->(r)-[:PROCESSED_BY]->(run:PipelineRun)
RETURN p.code AS participant, p.group AS group, run.epochs_rejected_pct AS pct_rejected,
       r.epochs_closed AS closed_kept, run.qc_flags AS flags
ORDER BY pct_rejected DESC LIMIT 5"""},
    {"title": "5. Biomarker catalogue by band",
     "expect": "15 per band (3 measures x 5 regions); alpha + 3 FAA + IAF = 19",
     "cypher": """MATCH (b:Biomarker)-[:IN_BAND]->(f:FrequencyBand)
RETURN f.name AS band, f.fmin AS fmin, f.fmax AS fmax, count(b) AS biomarkers ORDER BY fmin"""},
    {"title": "6. Eyes-closed relative alpha by region and group",
     "expect": "occipital highest (~0.5-0.6), frontal lowest",
     "cypher": """MATCH (p:Participant)-[:HAS_SESSION]->()-[:HAS_RECORDING]->()-[:HAS_FEATURE]->(v:FeatureValue {condition: 'closed'})
      -[:OF]->(b:Biomarker {measure: 'rel_power'})-[:IN_BAND]->(:FrequencyBand {name: 'alpha'}),
      (b)-[:AT_REGION]->(g:BrainRegion)
RETURN g.name AS region, p.group AS group, round(avg(v.value), 3) AS mean_rel_alpha, count(v) AS n
ORDER BY mean_rel_alpha DESC"""},
    {"title": "7. Frontal alpha asymmetry (F4-F3, eyes closed) by group",
     "expect": "both near 0, DEP slightly positive: classic FAA finding not replicated",
     "cypher": """MATCH (p:Participant)-[:HAS_SESSION]->()-[:HAS_RECORDING]->()-[:HAS_FEATURE]->(v:FeatureValue {condition: 'closed'})
      -[:OF]->(:Biomarker {name: 'faa_F4_F3'})
RETURN p.group AS group, count(v) AS n, round(avg(v.value), 3) AS mean_faa,
       round(percentileCont(v.value, 0.5), 3) AS median_faa ORDER BY group"""},
    {"title": "8. Models ranked by AUC: top 5 and the 2 baselines at the bottom",
     "expect": "eeg_all_cal ~0.73 on top; sex_only 0.60 and majority 0.50 last",
     "cypher": """MATCH (m:Model)-[:HAS_METRIC]->(x:Metric {name: 'auc'})
WITH m.name AS model, max(x.mean) AS auc ORDER BY auc DESC
WITH collect({model: model, auc: auc}) AS ranked
UNWIND ranked[0..5] + ranked[-2..] AS r
RETURN r.model AS model, round(r.auc, 3) AS auc"""},
    {"title": "9. Which biomarkers drive the model? (eeg_all_lr weights, + = more DEP)",
     "expect": "open temporal theta (-), closed frontal delta (+), open parietal theta (-)",
     "cypher": """MATCH (:Model {name: 'eeg_all_lr'})-[w:WEIGHTS]->(b:Biomarker)-[:AT_REGION]->(g:BrainRegion)
RETURN b.name AS biomarker, w.condition AS condition, g.name AS region,
       round(w.mean, 3) AS weight, w.same_sign AS sign_stable
ORDER BY abs(w.mean) DESC LIMIT 6"""},
    {"title": "10. Risk tiers (headline calibrated model) vs actual group",
     "expect": "low 46 (33 CTL / 13 DEP), moderate 72 (42 / 30), high 2 (0 / 2)",
     "cypher": """MATCH (p:Participant)-[:HAS_RISK]->(e:RiskEstimate {model: 'sex_eeg_closed_cal'})
RETURN e.tier AS tier, sum(CASE p.group WHEN 'CTL' THEN 1 ELSE 0 END) AS ctl,
       sum(CASE p.group WHEN 'DEP' THEN 1 ELSE 0 END) AS dep, round(avg(e.p_dep), 3) AS mean_p
ORDER BY mean_p"""},
]
