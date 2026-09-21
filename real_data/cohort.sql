-- Reconstruct the MIMIC-IV 3.1 cohort in temporary PostgreSQL tables.
\set ON_ERROR_STOP on
\pset pager off
\o /dev/null

DROP TABLE IF EXISTS pg_temp._bc_adult;

CREATE TEMP TABLE _bc_adult ON COMMIT PRESERVE ROWS AS
SELECT a.subject_id,
       a.hadm_id,
       a.admittime,
       a.dischtime,
       a.admission_type,
       a.admission_location,
       a.race,
       p.gender,
       p.anchor_year_group,
       p.anchor_age + EXTRACT(YEAR FROM a.admittime)::int - p.anchor_year
         AS age
FROM mimiciv_hosp.admissions a
JOIN mimiciv_hosp.patients p USING (subject_id)
WHERE a.dischtime > a.admittime
  AND p.anchor_age + EXTRACT(YEAR FROM a.admittime)::int - p.anchor_year >= 18;

CREATE INDEX ON _bc_adult (subject_id,hadm_id);

ANALYZE _bc_adult;

DROP TABLE IF EXISTS pg_temp._bc_admin;

CREATE TEMP TABLE _bc_admin ON COMMIT PRESERVE ROWS AS
SELECT a.*,
       e.emar_id,
       e.charttime AS admin_time,
       e.medication,
       r.route,
       CASE
         WHEN lower(e.medication) ~ 'ceftriaxone' THEN 'ceftriaxone'
         WHEN lower(e.medication) ~ 'cefepime' THEN 'cefepime'
         WHEN lower(e.medication) ~ '(piperacillin.*tazobactam|zosyn)'
           THEN 'piperacillin_tazobactam'
       END AS arm
FROM _bc_adult a
JOIN mimiciv_hosp.emar e
  ON e.subject_id = a.subject_id
 AND e.hadm_id = a.hadm_id
 AND e.charttime >= a.admittime
 AND e.charttime < LEAST(a.admittime + INTERVAL '72 hours', a.dischtime)
LEFT JOIN LATERAL (
  SELECT MIN(p.route) AS route
  FROM mimiciv_hosp.prescriptions p
  WHERE e.pharmacy_id IS NOT NULL
    AND p.pharmacy_id = e.pharmacy_id
) r ON TRUE
WHERE e.event_txt = 'Administered'
  AND lower(e.medication) ~
    '(ceftriaxone|cefepime|piperacillin.*tazobactam|zosyn)'
  AND lower(e.medication) !~
    '(desensitization|graded challenge|placebo|ophth|eye|otic|heparin lock)';

CREATE INDEX ON _bc_admin (hadm_id,admin_time);

ANALYZE _bc_admin;

DROP TABLE IF EXISTS pg_temp._bc_chosen;

CREATE TEMP TABLE _bc_chosen ON COMMIT PRESERVE ROWS AS
WITH first_time AS (
  SELECT hadm_id, MIN(admin_time) AS index_time
  FROM _bc_admin
  WHERE lower(COALESCE(route,'')) ~ '(^iv|intraven|central|peripheral)'
  GROUP BY hadm_id
), unique_arm AS (
  SELECT f.hadm_id,
         f.index_time,
         COUNT(DISTINCT a.arm)::int AS n_arms_at_index
  FROM first_time f
  JOIN _bc_admin a
    ON a.hadm_id = f.hadm_id
   AND a.admin_time = f.index_time
   AND lower(COALESCE(a.route,'')) ~ '(^iv|intraven|central|peripheral)'
  GROUP BY f.hadm_id,f.index_time
)
SELECT DISTINCT ON (a.hadm_id)
       a.*,
       u.index_time,
       u.n_arms_at_index
FROM _bc_admin a
JOIN unique_arm u
  ON u.hadm_id = a.hadm_id
 AND u.index_time = a.admin_time
WHERE lower(COALESCE(a.route,'')) ~ '(^iv|intraven|central|peripheral)'
ORDER BY a.hadm_id,a.arm,a.emar_id;

CREATE UNIQUE INDEX ON _bc_chosen (hadm_id);

CREATE INDEX ON _bc_chosen (subject_id,index_time);

ANALYZE _bc_chosen;

DROP TABLE IF EXISTS pg_temp._bc_culture;

CREATE TEMP TABLE _bc_culture ON COMMIT PRESERVE ROWS AS
WITH by_specimen AS (
  SELECT c.hadm_id,
         c.subject_id,
         c.index_time,
         m.micro_specimen_id,
         MAX(m.charttime) AS culture_time,
         MIN(m.storetime) AS minimum_recorded_storetime,
         BOOL_OR(
           m.org_name IS NOT NULL
           AND btrim(m.org_name) <> ''
           AND COALESCE(m.org_itemid,-1) <> 90856
         ) AS positive_culture
  FROM _bc_chosen c
  JOIN mimiciv_hosp.microbiologyevents m
    ON m.subject_id = c.subject_id
   AND (m.hadm_id = c.hadm_id OR m.hadm_id IS NULL)
   AND m.spec_type_desc = 'BLOOD CULTURE'
   AND m.charttime >= c.index_time - INTERVAL '24 hours'
   AND m.charttime < c.index_time
  GROUP BY c.hadm_id,c.subject_id,c.index_time,m.micro_specimen_id
)
SELECT DISTINCT ON (hadm_id) *
FROM by_specimen
ORDER BY hadm_id,culture_time DESC,micro_specimen_id;

CREATE UNIQUE INDEX ON _bc_culture (hadm_id);

ANALYZE _bc_culture;

DROP TABLE IF EXISTS pg_temp._bc_wbc;

CREATE TEMP TABLE _bc_wbc ON COMMIT PRESERVE ROWS AS
SELECT DISTINCT ON (c.hadm_id)
       c.hadm_id,
       l.valuenum AS blood_wbc,
       l.charttime AS wbc_time,
       l.storetime AS wbc_storetime,
       ABS(EXTRACT(EPOCH FROM (l.charttime-c.culture_time)))/3600.0
         AS wbc_culture_gap_h
FROM _bc_culture c
JOIN mimiciv_hosp.labevents l
  ON l.subject_id = c.subject_id
 AND (l.hadm_id = c.hadm_id OR l.hadm_id IS NULL)
 AND l.itemid = 51301
 AND l.charttime >= c.culture_time - INTERVAL '6 hours'
 AND l.charttime <= c.culture_time + INTERVAL '6 hours'
 AND l.charttime < c.index_time
 AND COALESCE(l.storetime,l.charttime) <= c.index_time
 AND l.valuenum IS NOT NULL
 AND l.valuenum >= 0
ORDER BY c.hadm_id,
         ABS(EXTRACT(EPOCH FROM (l.charttime-c.culture_time))),
         l.storetime DESC NULLS LAST,
         l.labevent_id DESC;

CREATE UNIQUE INDEX ON _bc_wbc (hadm_id);

ANALYZE _bc_wbc;

DROP TABLE IF EXISTS pg_temp._bc_prior_abx;

CREATE TEMP TABLE _bc_prior_abx ON COMMIT PRESERVE ROWS AS
SELECT c.hadm_id,
       BOOL_OR(
         e.charttime < c.culture_time
         AND lower(e.medication) ~
           '(cef|cillin|penem|floxacin|vancomycin|linezolid|daptomycin|aztreonam|gentamicin|tobramycin|amikacin|trimethoprim|sulfamethoxazole|metronidazole|clindamycin|doxycycline|azithromycin|erythromycin)'
         AND lower(e.medication) !~
           '(ophth|eye|otic|desensitization|graded challenge|placebo|heparin lock|vancomycin oral)'
       ) AS systemic_abx_pre_specimen_24h,
       BOOL_OR(
         e.charttime >= c.culture_time
         AND e.charttime < c.index_time
         AND lower(e.medication) ~
           '(cef|cillin|penem|floxacin|aztreonam|gentamicin|tobramycin|amikacin|trimethoprim|sulfamethoxazole)'
         AND lower(e.medication) !~
           '(ceftriaxone|cefepime|piperacillin.*tazobactam|zosyn|ophth|eye|otic|desensitization|graded challenge|placebo|heparin lock)'
       ) AS competing_backbone_specimen_to_a
FROM _bc_culture c
LEFT JOIN mimiciv_hosp.emar e
  ON e.subject_id = c.subject_id
 AND e.hadm_id = c.hadm_id
 AND e.charttime >= c.culture_time - INTERVAL '24 hours'
 AND e.charttime < c.index_time
 AND e.event_txt = 'Administered'
GROUP BY c.hadm_id;

CREATE UNIQUE INDEX ON _bc_prior_abx (hadm_id);

ANALYZE _bc_prior_abx;

DROP TABLE IF EXISTS pg_temp._bc_episode;

CREATE TEMP TABLE _bc_episode ON COMMIT PRESERVE ROWS AS
SELECT a.*,
       c.micro_specimen_id,
       c.culture_time,
       c.minimum_recorded_storetime,
       c.positive_culture,
       w.blood_wbc,
       w.wbc_time,
       w.wbc_storetime,
       w.wbc_culture_gap_h,
       COALESCE(p.systemic_abx_pre_specimen_24h,false)
         AS systemic_abx_pre_specimen_24h,
       COALESCE(p.competing_backbone_specimen_to_a,false)
         AS competing_backbone_specimen_to_a,
       EXTRACT(EPOCH FROM (a.index_time-a.admittime))/3600.0
         AS admission_to_a_h,
       EXTRACT(EPOCH FROM (a.index_time-c.culture_time))/3600.0
         AS specimen_to_a_h
FROM _bc_chosen a
LEFT JOIN _bc_culture c USING (hadm_id,subject_id,index_time)
LEFT JOIN _bc_wbc w USING (hadm_id)
LEFT JOIN _bc_prior_abx p USING (hadm_id);

CREATE UNIQUE INDEX ON _bc_episode (hadm_id);

ANALYZE _bc_episode;

DROP TABLE IF EXISTS pg_temp._bc_definitions;

CREATE TEMP TABLE _bc_definitions (
  definition text PRIMARY KEY,
  treatment_h int NOT NULL,
  specimen_h int NOT NULL
) ON COMMIT PRESERVE ROWS;

INSERT INTO _bc_definitions VALUES ('B_24h_treatment_24h_specimen',24,24);

DROP TABLE IF EXISTS pg_temp._bc_candidates;

CREATE TEMP TABLE _bc_candidates ON COMMIT PRESERVE ROWS AS
WITH eligible AS (
  SELECT d.definition,
         e.*,
         ROW_NUMBER() OVER (
           PARTITION BY d.definition,e.subject_id
           ORDER BY e.index_time,e.hadm_id
         ) AS subject_rn
  FROM _bc_definitions d
  JOIN _bc_episode e
    ON e.n_arms_at_index=1
   AND e.admission_to_a_h <= d.treatment_h
   AND e.specimen_to_a_h <= d.specimen_h
   AND e.culture_time IS NOT NULL
   AND e.minimum_recorded_storetime > e.index_time
   AND e.blood_wbc IS NOT NULL
   AND NOT e.systemic_abx_pre_specimen_24h
   AND NOT e.competing_backbone_specimen_to_a
)
SELECT * FROM eligible WHERE subject_rn=1;

CREATE UNIQUE INDEX ON _bc_candidates (definition,subject_id);

CREATE INDEX ON _bc_candidates (subject_id,hadm_id,index_time);

ANALYZE _bc_candidates;

DROP TABLE IF EXISTS pg_temp._bc_z_rich;

CREATE TEMP TABLE _bc_z_rich ON COMMIT PRESERVE ROWS AS
WITH lab_map(itemid,concept) AS (
  VALUES
    (51301,'wbc'),
    (52075,'anc'),
    (51133,'alc'),
    (52769,'alc'),
    (51245,'lymphocyte_percent'),
    (51256,'neutrophil_percent'),
    (51144,'bands'),
    (52135,'immature_granulocytes')
), mapped AS (
  SELECT c.definition,
         c.subject_id,
         m.concept,
         l.valuenum,
         ROW_NUMBER() OVER (
           PARTITION BY c.definition,c.subject_id,m.concept
           ORDER BY ABS(EXTRACT(EPOCH FROM (l.charttime-c.culture_time))),
                    l.storetime DESC NULLS LAST,l.labevent_id DESC
         ) AS rn
  FROM _bc_candidates c
  JOIN mimiciv_hosp.labevents l
    ON l.subject_id=c.subject_id
   AND (l.hadm_id=c.hadm_id OR l.hadm_id IS NULL)
   AND l.charttime >= c.culture_time-INTERVAL '6 hours'
   AND l.charttime <= c.culture_time+INTERVAL '6 hours'
   AND l.charttime < c.index_time
   AND COALESCE(l.storetime,l.charttime) <= c.index_time
   AND l.valuenum IS NOT NULL
  JOIN lab_map m ON m.itemid=l.itemid
)
SELECT definition,subject_id,
       MAX(valuenum) FILTER (WHERE concept='wbc' AND rn=1) AS z_wbc,
       MAX(valuenum) FILTER (WHERE concept='anc' AND rn=1) AS z_anc,
       MAX(valuenum) FILTER (WHERE concept='alc' AND rn=1) AS z_alc,
       MAX(valuenum) FILTER (
         WHERE concept='lymphocyte_percent' AND rn=1
       ) AS z_lymphocyte_percent,
       MAX(valuenum) FILTER (
         WHERE concept='neutrophil_percent' AND rn=1
       ) AS z_neutrophil_percent,
       MAX(valuenum) FILTER (WHERE concept='bands' AND rn=1) AS z_bands,
       MAX(valuenum) FILTER (
         WHERE concept='immature_granulocytes' AND rn=1
       ) AS z_immature_granulocytes
FROM mapped
GROUP BY definition,subject_id;

CREATE UNIQUE INDEX ON _bc_z_rich (definition,subject_id);

ANALYZE _bc_z_rich;

DROP TABLE IF EXISTS pg_temp._bc_w_rich;

CREATE TEMP TABLE _bc_w_rich ON COMMIT PRESERVE ROWS AS
WITH organism_classes AS (
  SELECT DISTINCT c.definition,c.subject_id,
         CASE
           WHEN upper(m.org_name) ~
             '(ESCHERICHIA|KLEBSIELLA|PSEUDOMONAS|PROTEUS|ENTEROBACTER|SERRATIA|CITROBACTER|MORGANELLA|PROVIDENCIA|ACINETOBACTER|GRAM NEGATIVE)'
             THEN 'gram_negative'
           WHEN upper(m.org_name) ~
             '(ENTEROCOCC|STAPH|STREP|CORYNEBACTER|BACILLUS|GRAM POSITIVE)'
             THEN 'gram_positive'
           WHEN upper(m.org_name) ~ '(YEAST|CANDIDA)' THEN 'yeast'
           WHEN upper(m.org_name) ~
             '(BACTEROIDES|CLOSTRIDIUM|FUSOBACTER|ANAEROB)'
             THEN 'anaerobe'
           ELSE 'other'
         END AS pathogen_class
  FROM _bc_candidates c
  JOIN mimiciv_hosp.microbiologyevents m
    ON m.subject_id=c.subject_id
   AND m.micro_specimen_id=c.micro_specimen_id
  WHERE m.org_name IS NOT NULL
    AND btrim(m.org_name)<>''
    AND COALESCE(m.org_itemid,-1)<>90856
), aggregated AS (
  SELECT definition,subject_id,
         COUNT(DISTINCT pathogen_class)::int AS class_n,
         MIN(pathogen_class) AS only_class
  FROM organism_classes
  GROUP BY definition,subject_id
)
SELECT c.definition,c.subject_id,
       CASE
         WHEN NOT c.positive_culture THEN 'negative'
         WHEN a.class_n>1 THEN 'mixed'
         WHEN a.only_class IS NOT NULL THEN a.only_class
         ELSE 'other'
       END AS w_pathogen_class
FROM _bc_candidates c
LEFT JOIN aggregated a USING (definition,subject_id);

CREATE UNIQUE INDEX ON _bc_w_rich (definition,subject_id);

ANALYZE _bc_w_rich;

DROP TABLE IF EXISTS pg_temp._bc_baseline;

CREATE TEMP TABLE _bc_baseline ON COMMIT PRESERVE ROWS AS
WITH location AS (
  SELECT c.definition,c.subject_id,
         COALESCE(t.careunit,'NO_RECORDED_CAREUNIT') AS careunit_at_a
  FROM _bc_candidates c
  LEFT JOIN LATERAL (
    SELECT tr.careunit
    FROM mimiciv_hosp.transfers tr
    WHERE tr.subject_id=c.subject_id
      AND tr.hadm_id=c.hadm_id
      AND tr.intime <= c.index_time
      AND (tr.outtime IS NULL OR tr.outtime > c.index_time)
    ORDER BY tr.intime DESC,tr.transfer_id DESC
    LIMIT 1
  ) t ON TRUE
), lab_map(itemid,concept) AS (
  VALUES
    (50813,'lactate'),(50862,'albumin'),(50882,'bicarbonate'),
    (50885,'bilirubin'),(50912,'creatinine'),(50931,'glucose'),
    (50971,'potassium'),(50983,'sodium'),(51006,'bun'),
    (51222,'hemoglobin'),(51237,'inr'),(51265,'platelet')
), labs AS (
  SELECT c.definition,c.subject_id,m.concept,l.valuenum,
         ROW_NUMBER() OVER (
           PARTITION BY c.definition,c.subject_id,m.concept
           ORDER BY l.charttime DESC,l.storetime DESC NULLS LAST,
                    l.labevent_id DESC
         ) AS rn
  FROM _bc_candidates c
  JOIN mimiciv_hosp.labevents l
    ON l.subject_id=c.subject_id
   AND (l.hadm_id=c.hadm_id OR l.hadm_id IS NULL)
   AND l.charttime >= c.index_time-INTERVAL '24 hours'
   AND l.charttime < c.index_time
   AND COALESCE(l.storetime,l.charttime) <= c.index_time
   AND l.valuenum IS NOT NULL
  JOIN lab_map m ON m.itemid=l.itemid
), wide AS (
  SELECT definition,subject_id,
         MAX(valuenum) FILTER (WHERE concept='lactate' AND rn=1) AS lactate,
         MAX(valuenum) FILTER (WHERE concept='albumin' AND rn=1) AS albumin,
         MAX(valuenum) FILTER (WHERE concept='bicarbonate' AND rn=1)
           AS bicarbonate,
         MAX(valuenum) FILTER (WHERE concept='bilirubin' AND rn=1)
           AS bilirubin,
         MAX(valuenum) FILTER (WHERE concept='creatinine' AND rn=1)
           AS creatinine,
         MAX(valuenum) FILTER (WHERE concept='glucose' AND rn=1) AS glucose,
         MAX(valuenum) FILTER (WHERE concept='potassium' AND rn=1)
           AS potassium,
         MAX(valuenum) FILTER (WHERE concept='sodium' AND rn=1) AS sodium,
         MAX(valuenum) FILTER (WHERE concept='bun' AND rn=1) AS bun,
         MAX(valuenum) FILTER (WHERE concept='hemoglobin' AND rn=1)
           AS hemoglobin,
         MAX(valuenum) FILTER (WHERE concept='inr' AND rn=1) AS inr,
         MAX(valuenum) FILTER (WHERE concept='platelet' AND rn=1) AS platelet
  FROM labs
  GROUP BY definition,subject_id
)
SELECT l.definition,l.subject_id,l.careunit_at_a,
       w.lactate,w.albumin,w.bicarbonate,w.bilirubin,w.creatinine,w.glucose,
       w.potassium,w.sodium,w.bun,w.hemoglobin,w.inr,w.platelet
FROM location l
LEFT JOIN wide w USING (definition,subject_id);

CREATE UNIQUE INDEX ON _bc_baseline (definition,subject_id);

ANALYZE _bc_baseline;

DROP TABLE IF EXISTS pg_temp._bc_source_z;

CREATE TEMP TABLE _bc_source_z ON COMMIT PRESERVE ROWS AS
WITH primary_candidate AS (
  SELECT *
  FROM _bc_candidates
  WHERE definition='B_24h_treatment_24h_specimen'
), specimen_flags AS (
  SELECT c.subject_id,
         COUNT(DISTINCT m.micro_specimen_id) FILTER (
           WHERE m.spec_type_desc='BLOOD CULTURE'
         )::int AS blood_specimens_24h,
         COUNT(DISTINCT m.micro_specimen_id) FILTER (
           WHERE m.spec_type_desc='BLOOD CULTURE'
             AND m.charttime >= c.index_time-INTERVAL '6 hours'
         )::int AS blood_specimens_6h,
         BOOL_OR(m.spec_type_desc='URINE') AS urine_collected,
         BOOL_OR(m.spec_type_desc IN (
           'SPUTUM','BRONCHOALVEOLAR LAVAGE','BRONCHIAL WASHINGS','Mini-BAL'
         )) AS respiratory_collected,
         BOOL_OR(m.spec_type_desc IN (
           'SWAB','TISSUE','ABSCESS','FLUID,OTHER','PERITONEAL FLUID',
           'PLEURAL FLUID','JOINT FLUID','BILE'
         )) AS tissue_fluid_collected,
         BOOL_OR(m.spec_type_desc='CSF;SPINAL FLUID') AS csf_collected
  FROM primary_candidate c
  LEFT JOIN mimiciv_hosp.microbiologyevents m
    ON m.subject_id=c.subject_id
   AND (m.hadm_id=c.hadm_id OR m.hadm_id IS NULL)
   AND m.charttime >= c.index_time-INTERVAL '24 hours'
   AND m.charttime < c.index_time
  GROUP BY c.subject_id
), urine_wbc AS (
  SELECT DISTINCT ON (c.subject_id)
         c.subject_id,
         l.valuenum AS urine_wbc_pre_a
  FROM primary_candidate c
  JOIN mimiciv_hosp.labevents l
    ON l.subject_id=c.subject_id
   AND (l.hadm_id=c.hadm_id OR l.hadm_id IS NULL)
   AND l.itemid=51516
   AND l.charttime >= c.index_time-INTERVAL '24 hours'
   AND l.charttime < c.index_time
   AND COALESCE(l.storetime,l.charttime) <= c.index_time
   AND l.valuenum IS NOT NULL
   AND l.valuenum>=0
  ORDER BY c.subject_id,l.charttime DESC,l.storetime DESC NULLS LAST,
           l.labevent_id DESC
)
SELECT c.subject_id,
       c.arm,
       c.positive_culture,
       f.blood_specimens_24h,
       f.blood_specimens_6h,
       f.urine_collected,
       f.respiratory_collected,
       f.tissue_fluid_collected,
       f.csf_collected,
       u.urine_wbc_pre_a
FROM primary_candidate c
JOIN specimen_flags f USING (subject_id)
LEFT JOIN urine_wbc u USING (subject_id);

CREATE UNIQUE INDEX ON _bc_source_z (subject_id);

ANALYZE _bc_source_z;

DROP TABLE IF EXISTS pg_temp._bc_source_feature_view;

CREATE TEMP TABLE _bc_source_feature_view ON COMMIT PRESERVE ROWS AS
SELECT c.definition,
       c.subject_id,
       c.arm,
       c.positive_culture::int AS w_culture_positive,
       w.w_pathogen_class,
       z.z_wbc,
       z.z_anc,
       z.z_alc,
       z.z_neutrophil_percent,
       z.z_bands,
       z.z_immature_granulocytes,
       c.age,
       c.admission_to_a_h,
       c.specimen_to_a_h,
       EXTRACT(HOUR FROM c.index_time)::int AS index_hour,
       EXTRACT(ISODOW FROM c.index_time)::int AS index_weekday,
       b.lactate,
       b.albumin,
       b.bicarbonate,
       b.bilirubin,
       b.creatinine,
       b.glucose,
       b.potassium,
       b.sodium,
       b.bun,
       b.hemoglobin,
       b.inr,
       b.platelet,
       c.gender,
       c.race,
       c.admission_type,
       c.admission_location,
       c.anchor_year_group,
       b.careunit_at_a,
       s.blood_specimens_24h,
       s.blood_specimens_6h,
       s.urine_collected::int AS urine_collected,
       s.respiratory_collected::int AS respiratory_collected,
       s.tissue_fluid_collected::int AS tissue_fluid_collected,
       s.csf_collected::int AS csf_collected,
       s.urine_wbc_pre_a
FROM _bc_candidates c
JOIN _bc_z_rich z USING (definition,subject_id)
JOIN _bc_w_rich w USING (definition,subject_id)
JOIN _bc_baseline b USING (definition,subject_id)
JOIN _bc_source_z s USING (subject_id)
WHERE c.definition='B_24h_treatment_24h_specimen'
  AND z.z_wbc IS NOT NULL;

CREATE UNIQUE INDEX ON _bc_source_feature_view (definition,subject_id);

ANALYZE _bc_source_feature_view;

DROP TABLE IF EXISTS pg_temp._bc_outcome_feature_view;

CREATE TEMP TABLE _bc_outcome_feature_view ON COMMIT PRESERVE ROWS AS
WITH primary_candidate AS (
  SELECT c.subject_id,
         c.hadm_id,
         c.index_time
  FROM _bc_candidates c
  WHERE c.definition='B_24h_treatment_24h_specimen'
), shaped AS (
  SELECT c.subject_id,
         CASE
           WHEN (
             a.deathtime IS NOT NULL
             AND a.deathtime>=c.index_time
             AND a.deathtime<=c.index_time+INTERVAL '28 days'
           ) OR (
             p.dod IS NOT NULL
             AND p.dod::date>=c.index_time::date
             AND p.dod::date<=(c.index_time+INTERVAL '28 days')::date
           ) THEN 0.0
           WHEN a.dischtime>=c.index_time+INTERVAL '28 days' THEN 0.0
           ELSE GREATEST(
             0.0,
             EXTRACT(EPOCH FROM (
               c.index_time+INTERVAL '28 days'-a.dischtime
             ))/86400.0
           )
         END AS y_daoh28,
         CASE
           WHEN (
             a.deathtime IS NOT NULL
             AND a.deathtime>=c.index_time
             AND a.deathtime<=c.index_time+INTERVAL '28 days'
           ) OR (
             p.dod IS NOT NULL
             AND p.dod::date>=c.index_time::date
             AND p.dod::date<=(c.index_time+INTERVAL '28 days')::date
           ) THEN 0 ELSE 1
         END::int AS y_survival28,
         CASE
           WHEN (
             a.deathtime IS NOT NULL
             AND a.deathtime>=c.index_time
             AND a.deathtime<=c.index_time+INTERVAL '90 days'
           ) OR (
             p.dod IS NOT NULL
             AND p.dod::date>=c.index_time::date
             AND p.dod::date<=(c.index_time+INTERVAL '90 days')::date
           ) THEN 0 ELSE 1
         END::int AS y_survival90,
         (
           p.dod IS NOT NULL
           AND p.dod::date=(c.index_time+INTERVAL '28 days')::date
         )::int AS dod_day28_boundary,
         (
           p.dod IS NOT NULL
           AND p.dod::date=(c.index_time+INTERVAL '90 days')::date
         )::int AS dod_day90_boundary
  FROM primary_candidate c
  JOIN mimiciv_hosp.admissions a
    ON a.subject_id=c.subject_id AND a.hadm_id=c.hadm_id
  JOIN mimiciv_hosp.patients p
    ON p.subject_id=c.subject_id
)
SELECT f.*,
       o.y_daoh28,
       o.y_survival28,
       o.y_survival90,
       o.dod_day28_boundary,
       o.dod_day90_boundary
FROM _bc_source_feature_view f
JOIN shaped o USING (subject_id);

CREATE UNIQUE INDEX ON _bc_outcome_feature_view (definition,subject_id);

ANALYZE _bc_outcome_feature_view;

DROP TABLE IF EXISTS pg_temp._bc_survival90_feature_view;

CREATE TEMP TABLE _bc_survival90_feature_view ON COMMIT PRESERVE ROWS AS
WITH primary_candidate AS (
  SELECT c.subject_id,
         c.index_time
  FROM _bc_candidates c
  WHERE c.definition='B_24h_treatment_24h_specimen'
), followup AS (
  SELECT c.subject_id,
         c.index_time,
         MIN(a.deathtime) FILTER (
           WHERE a.deathtime IS NOT NULL
             AND a.deathtime>=c.index_time
         ) AS first_exact_deathtime_on_after_t0,
         MAX(a.dischtime) AS last_hospital_discharge
  FROM primary_candidate c
  JOIN mimiciv_hosp.admissions a
    ON a.subject_id=c.subject_id
  GROUP BY c.subject_id,c.index_time
), shaped AS (
  SELECT f.subject_id,
         CASE
           WHEN f.first_exact_deathtime_on_after_t0 IS NOT NULL
             AND f.first_exact_deathtime_on_after_t0<=
                 f.index_time+INTERVAL '90 days'
             THEN EXTRACT(EPOCH FROM (
                    f.first_exact_deathtime_on_after_t0-f.index_time
                  ))/86400.0
           WHEN p.dod IS NOT NULL
             AND p.dod::date>=f.index_time::date
             AND p.dod::date<=(f.index_time+INTERVAL '90 days')::date
             THEN (p.dod::date-f.index_time::date)::double precision
           ELSE 90.0
         END AS survival90_observed_time_days,
         CASE
           WHEN f.first_exact_deathtime_on_after_t0 IS NOT NULL
             AND f.first_exact_deathtime_on_after_t0<=
                 f.index_time+INTERVAL '90 days'
             THEN 1
           WHEN p.dod IS NOT NULL
             AND p.dod::date>=f.index_time::date
             AND p.dod::date<=(f.index_time+INTERVAL '90 days')::date
             THEN 1
           ELSE 0
         END::int AS survival90_death_event,
         0::int AS survival90_coverage_censor_event,
         (
           f.last_hospital_discharge+INTERVAL '365 days'>=
             f.index_time+INTERVAL '90 days'
         )::int AS survival90_nominal_coverage_reaches_horizon,
         1.0::double precision AS survival90_censoring_survival_g,
         CASE
           WHEN f.first_exact_deathtime_on_after_t0 IS NOT NULL
             AND f.first_exact_deathtime_on_after_t0<=
                 f.index_time+INTERVAL '90 days'
             THEN 0
           WHEN p.dod IS NOT NULL
             AND p.dod::date>=f.index_time::date
             AND p.dod::date<=(f.index_time+INTERVAL '90 days')::date
             THEN 0
           ELSE 1
         END::int AS survival90_reconstructed_y
  FROM followup f
  JOIN mimiciv_hosp.patients p
    ON p.subject_id=f.subject_id
)
SELECT o.*,
       s.survival90_observed_time_days,
       s.survival90_death_event,
       s.survival90_coverage_censor_event,
       s.survival90_nominal_coverage_reaches_horizon,
       s.survival90_censoring_survival_g,
       s.survival90_reconstructed_y
FROM _bc_outcome_feature_view o
JOIN shaped s USING (subject_id);

CREATE UNIQUE INDEX ON _bc_survival90_feature_view (definition,subject_id);

ANALYZE _bc_survival90_feature_view;

\o
SELECT * FROM _bc_survival90_feature_view ORDER BY subject_id;
