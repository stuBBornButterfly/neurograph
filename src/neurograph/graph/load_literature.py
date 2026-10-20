# literature graph: (:Paper)-[:REPORTS]->(:Claim)-[:ABOUT]->(:Biomarker), (:Claim)-[:CONCERNS]->(:Condition)
SCHEMA = ["CREATE CONSTRAINT paper_pmid IF NOT EXISTS FOR (p:Paper) REQUIRE p.pmid IS UNIQUE",
          "CREATE CONSTRAINT claim_id IF NOT EXISTS FOR (c:Claim) REQUIRE c.id IS UNIQUE"]
RESET = "MATCH (n) WHERE n:Paper OR n:Claim CALL (n) { DETACH DELETE n } IN TRANSACTIONS OF 1000 ROWS"
PAPERS = """UNWIND $rows AS row
MERGE (p:Paper {pmid: row.pmid})
SET p.title = row.title, p.year = row.year, p.journal = row.journal"""
CLAIMS = """UNWIND $rows AS row
MATCH (p:Paper {pmid: row.pmid})
MERGE (c:Claim {id: row.id})
SET c.text = row.sentence, c.band = row.band, c.measure = row.measure, c.region = row.region,
    c.direction = row.direction, c.hypothesis = row.hypothesis, c.entail = row.entail, c.verified = row.verified
MERGE (p)-[:REPORTS]->(c)
MERGE (d:Condition {name: row.condition})
MERGE (c)-[:CONCERNS]->(d)
WITH c, row
UNWIND row.biomarkers AS name
MATCH (b:Biomarker {name: name})
MERGE (c)-[:ABOUT]->(b)"""
# claims the agent may cite: verified only
CLAIM_QUERY = """MATCH (p:Paper)-[:REPORTS]->(c:Claim {verified: true})
WHERE ($band IS NULL OR c.band = $band) AND ($region IS NULL OR c.region IN [$region, 'global'])
  AND ($measure IS NULL OR c.measure = $measure)
RETURN p.pmid AS pmid, p.year AS year, c.text AS text, c.band AS band, c.measure AS measure,
       c.region AS region, c.direction AS direction, c.entail AS entail
ORDER BY c.entail DESC"""
