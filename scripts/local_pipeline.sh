#!/usr/bin/env bash
# Whole pipeline on one local GPU machine instead of Kaggle (as run on 2026-10-21: RTX 5060 Ti 16 GB, 15 GB RAM).
# Needs: .venv with the packages in requirements*.txt (scikit-learn < 1.8: 1.8 changed StratifiedGroupKFold splits),
# Docker Neo4j 5 (+APOC) on bolt://localhost:7687, Ollama with qwen3.5:9b and gemma2:9b-instruct-q4_K_M (Phase 9),
# and a gitignored .env with NEO4J_URI / NEO4J_USERNAME / NEO4J_PASSWORD and the NEUROGRAPH_* path overrides:
#   NEUROGRAPH_BIDS_DIR=<repo>/data/bids/ds003478  NEUROGRAPH_DERIVATIVES_DIR=<repo>/data/derivatives
#   NEUROGRAPH_INPUT_ROOT=<repo>/data   (stands in for /kaggle/input in the configs' input globs)
# Usage: bash scripts/local_pipeline.sh [first_step]   e.g. "bash scripts/local_pipeline.sh 60"
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; source .env; set +a
source .venv/bin/activate
export MLFLOW_DISABLE_AGENT_HINT=1 TOKENIZERS_PARALLELISM=false
mkdir -p logs
FROM=${1:-10}
step() { echo "== $1 $(date +%T)"; python -m "scripts.$1" "${@:2}" 2>&1 | grep -v -i -e warn -e "it/s" > "logs/$1.log"; tail -n 3 "logs/$1.log"; }

if [ "$FROM" -le 10 ]; then  # Phases 1-5: EEG pipeline and study graph (notebooks 20-50)
  step download_data --include "sub-*"
  step summarize_dataset
  step run_preprocess
  step qc_summary
fi
if [ "$FROM" -le 30 ]; then
  step compute_psd_features
  step build_features
fi
if [ "$FROM" -le 40 ]; then
  step run_baselines; step explain_models; step calibrate_tiers
  step train_eegnet; step channel_subsets; step log_mlflow
fi
if [ "$FROM" -le 50 ]; then
  step build_study_graph; step run_graph_queries
fi
if [ "$FROM" -le 60 ]; then  # Phase 6 + 8: university intake, index, policy + literature graph, agent (notebook 60)
  python -m pytest tests -q -p no:cacheprovider | tail -n 1
  step ingest_pages --refetch
  step build_index
  step build_policy_graph
  step build_literature_graph
  step run_agent_eval
fi
if [ "$FROM" -le 70 ]; then  # Phase 7: QLoRA (notebook 70)
  step make_qa_dataset
  step train_qlora
  step eval_qlora
  step run_agent_eval --adapter "$NEUROGRAPH_DERIVATIVES_DIR/qlora/adapter" --tag _qlora
  step export_gguf
fi
if [ "$FROM" -le 90 ]; then  # Phase 9: benchmark, RAGAS, human-audit sheet; demo Space folder
  step run_benchmark
  step run_ragas
  step make_audit_sheet
  step export_graph_snapshot
  step build_space
fi
