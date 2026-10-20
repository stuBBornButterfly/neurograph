// one unique key per node type; MERGE on these keys makes every load safe to rerun
CREATE CONSTRAINT dataset_id IF NOT EXISTS FOR (n:Dataset) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT participant_code IF NOT EXISTS FOR (n:Participant) REQUIRE n.code IS UNIQUE;
CREATE CONSTRAINT assessment_id IF NOT EXISTS FOR (n:Assessment) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT session_id IF NOT EXISTS FOR (n:Session) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT recording_id IF NOT EXISTS FOR (n:Recording) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT pipeline_run_id IF NOT EXISTS FOR (n:PipelineRun) REQUIRE n.run_id IS UNIQUE;
CREATE CONSTRAINT feature_value_id IF NOT EXISTS FOR (n:FeatureValue) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT biomarker_name IF NOT EXISTS FOR (n:Biomarker) REQUIRE n.name IS UNIQUE;
CREATE CONSTRAINT band_name IF NOT EXISTS FOR (n:FrequencyBand) REQUIRE n.name IS UNIQUE;
CREATE CONSTRAINT region_name IF NOT EXISTS FOR (n:BrainRegion) REQUIRE n.name IS UNIQUE;
CREATE CONSTRAINT model_id IF NOT EXISTS FOR (n:Model) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT split_name IF NOT EXISTS FOR (n:Split) REQUIRE n.name IS UNIQUE;
CREATE CONSTRAINT metric_id IF NOT EXISTS FOR (n:Metric) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT risk_id IF NOT EXISTS FOR (n:RiskEstimate) REQUIRE n.id IS UNIQUE;
