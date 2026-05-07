#!/bin/bash
PYTHONPATH="${PYTHONPATH}:$(realpath "./src")"
export PYTHONPATH

# Navigate to the script's directory (project root)
cd "$(dirname "$0")" || exit

echo "Running base agent tests"
pytest tests/agents/test_base_agent.py
echo "Done..."
echo "==============================================="

echo "Running dummy agent tests"
pytest tests/agents/test_dummy_agent.py
echo "Done..."
echo "==============================================="

echo "Running scraper agent tests"
pytest tests/agents/test_scraper_agent.py
echo "Done..."
echo "==============================================="

echo "Running surveillance data collector tests"
pytest tests/agents/test_surveillance_data_collector.py
echo "Done..."
echo "==============================================="

echo "Running route finder agent tests"
pytest tests/agents/test_route_finder_agent.py
echo "Done..."
echo "==============================================="

echo "Running LangChain LLM wrapper tests"
pytest tests/llm/test_langchain_llm.py
echo "Done..."
echo "==============================================="

echo "Running SurveillanceLLM batch tests"
pytest tests/llm/test_surveillance_llm_batch.py
echo "Done..."
echo "==============================================="

echo "Running analysis chain batch enrichment tests"
pytest tests/chains/test_analysis_chain_batch.py
echo "Done..."
echo "==============================================="

echo "Running io tools tests"
pytest tests/tools/test_io_tools.py
echo "Done..."
echo "==============================================="

echo "Running mapping tools tests"
pytest tests/tools/test_mapping_tools.py
echo "Done..."
echo "==============================================="

echo "Running chart tools tests"
pytest tests/tools/test_chart_tools.py
echo "Done..."
echo "==============================================="

echo "Running memory store tests"
pytest tests/memory/test_memory_store.py
echo "Done..."
echo "==============================================="

echo "Running decorator tests"
pytest tests/utils/test_decorators.py
echo "Done..."
echo "==============================================="

echo "Running db utils tests"
pytest tests/utils/test_db.py
echo "Done..."
echo "==============================================="

echo "Running overpass utils tests"
pytest tests/utils/test_overpass.py
echo "Done..."
echo "==============================================="

echo "Running LangChain configuration tests"
pytest tests/config/test_langchain_config.py
echo "Done..."
echo "==============================================="

echo "Running pipeline configuration tests"
pytest tests/config/test_pipeline_config.py
echo "Done..."
echo "==============================================="

echo "Running Routing tools tests"
pytest tests/tools/test_routing_tools.py
echo "Done..."
echo "==============================================="

echo "Running orchestrator analyzer-skip tests"
pytest tests/orchestration/test_analyzer_skip.py
echo "Done..."
echo "==============================================="

echo "Running orchestrator on-scrape-complete hook tests"
pytest tests/orchestration/test_on_scrape_complete.py
echo "Done..."
echo "==============================================="
