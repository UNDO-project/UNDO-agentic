# An Agentic System for Researching Surveillance Infrastructure

A multi-agent system for analyzing surveillance infrastructure and computing privacy-preserving walking routes in urban environments using OpenStreetMap data. The system operates completely locally without external APIs and provides both CLI and REST API interfaces.

## Overview

The pipeline consists of three main agents:

- **Scraper Agent**: Downloads surveillance camera data from OpenStreetMap via Overpass API
- **Analyzer Agent**: Enriches data using local LLM analysis and generates visualizations
- **Route Finder Agent**: Computes low-surveillance walking routes using k-shortest paths and spatial analysis

**Key Features:**
- **Privacy-focused routing**: Find walking routes that minimize camera exposure
- **Local LLM processing**: No external API calls - complete privacy
- **Dual interface**: Rich CLI and production-ready FastAPI REST API
- **Real-time updates**: WebSocket support for live pipeline progress
- **Intelligent caching**: Agent memory stores results to avoid redundant computation
- **Analysis presets + overrides**: Two scenario presets (`basic`, `full`) layered with per-output toggle overrides via CLI flags or the API `overrides` field
- **Comprehensive visualizations**: Heatmaps, hotspots, route maps, and statistical charts
- **Spatial optimization**: Efficient GeoDataFrame indexing for large camera datasets

## Methodology

The pipeline composes several established methods rather than inventing new ones. Each subsection below lists what the codebase does, the file it lives in, and the canonical reference for that step. Full bibliographic entries are collected in [References](#references).

### Data acquisition

Camera locations are pulled from OpenStreetMap via the Overpass API (`src/agents/surveillance_data_collector.py`, `src/tools/surveillance_data_collector_tools.py`). The query targets `man_made=surveillance` features within a city's bounding box. Provenance: the underlying data is volunteered geographic information licensed under the ODbL.

### LLM enrichment

A LangChain analysis chain (`src/chains/analysis_chain.py`, `src/llm/surveillance_llm.py`, `src/prompts/prompt_template.py`) classifies each camera's privacy impact, sensitivity, and zone type, and produces the per-city report. The model is served locally by Ollama (default: `llama3:latest`). The enrichment layer is interpretive scaffolding.

### Hotspot analysis (four layers)

Each layer answers a different question — they complement rather than replace each other. All four project points to the local UTM zone before any metric computation (`src/tools/geo_projection.py`), so thresholds are isotropic in metres rather than degrees of latitude.

1. **HDBSCAN clusters** (`<city>_hotspots.geojson` + `<city>_hotspot_polygons.geojson`, `src/tools/hotspot_clustering.py`)
   Density-based clustering with locally-adaptive `ε`, computed in UTM metres so a "20-metre cluster" means the same thing at any latitude. See [Campello et al. 2013][campello2013], [McInnes et al. 2017][mcinnes2017].
2. **Planar KDE density surface** (`<city>_heatmap.html` + `<city>_density.geojson`, `src/tools/density_kde.py`)
   FFT-based Gaussian kernel density on a metric grid via [KDEpy][kdepy]; the folium heatmap is *derived* from the surface rather than from folium's opaque built-in interpolation, and the same surface contours into a GeoJSON layer at the 50/75/90/95 percentiles for researcher-grade work. The current implementation is planar — kernels radiate isotropically in 2-D; a network-constrained variant ([Okabe & Sugihara 2012][okabe2012]) is planned. See also [Silverman 1986][silverman1986], [Wand & Jones 1995][wand1995].
3. **Getis-Ord Gi\* hex grid** (`<city>_gi_star.geojson` + `<city>_gi_star.png`, `src/tools/spatial_stats.py`)
   Points are binned to an [H3 hexagonal grid][h3]; a distance-band spatial weights matrix is built on hex centroids in UTM metres ([`libpysal.weights.DistanceBand`][pysal]); per-hex Gi\* z-scores are computed with [`esda.G_Local`][esda] (`star=True`); p-values are corrected with Benjamini–Hochberg FDR; each hex is then classified `hot_99` / `hot_95` / `not_significant` / `cold_95` / `cold_99` — the ArcGIS/QGIS "Hot Spot Analysis" convention. See [Getis & Ord 1992][getis1992], [Ord & Getis 1995][ord1995], [Benjamini & Hochberg 1995][bh1995], [Rey & Anselin 2007][rey2007], [Wolf et al. 2021][wolf2021], [Brodsky 2018][brodsky2018]. Amnesty International's [*Decode Surveillance NYC*][amnesty] is the methodological precedent for crowdsourced camera-mapping research that this statistical layer is meant to support.
4. **Cameras per road-km** (`<city>_density_metrics.json`, `src/tools/density_metrics.py`)
   Stanford Computational Policy Lab's [*Surveilling Surveillance*][stanford2021] (2021) made cameras-per-linear-km of road the canonical headline for cross-city camera prevalence (0.2 in Los Angeles to 0.9 in Seoul). The motivation is that `cameras / km²` is sensitive to how much park, water, or other unbuilt land falls inside a city's polygon — those areas inflate the denominator without contributing cameras to the numerator. Normalising by road length compares infrastructure to infrastructure. This project follows that approach but uses the **OSMnx pedestrian graph** (rather than all roads), since the question we care about is what someone walking past gets exposed to. Reuses the routing agent's cached graph so the metric and routing layer share one OSM download. A secondary `cameras / km²` (convex hull of graph nodes via [SciPy's `ConvexHull`][scipy]) is kept as a sanity check against numbers cited elsewhere.

### Low-surveillance routing

The routing agent (`src/agents/route_finder.py`, `src/tools/routing_tools.py`) finds walking routes that minimise camera exposure between two coordinates.

1. **Pedestrian graph construction** — OSMnx downloads the walkable OSM network and caches it as GraphML at `overpass_data/.graph_cache/<sha>.graphml`. See [Boeing 2017][boeing2017].
2. **Node snapping** — start/end coordinates are snapped to the nearest graph node using `osmnx.distance.nearest_nodes`, with a configurable haversine threshold (default 500 m).
3. **k-shortest path generation** — `networkx.shortest_simple_paths` (Yen's algorithm) enumerates up to `max_candidates` simple paths between the snapped nodes. See [Yen 1971][yen1971], [Hagberg et al. 2008][networkx].
4. **Exposure scoring** — each candidate path is buffered by `buffer_radius_m` (default 50 m), and cameras within the buffer are counted via a [GeoPandas / Shapely][shapely] spatial join. The exposure score is reported in cameras per kilometre.
5. **Route selection** — the path with the minimum exposure score is returned, alongside a comparison against the unconstrained shortest path so the privacy gain is quantified rather than asserted.

### References

<!-- Sorted by topic so the inline citations can be skimmed in context. -->

#### Clustering, density, spatial statistics
[campello2013]: https://doi.org/10.1007/978-3-642-37456-2_14 "Campello, R. J. G. B., Moulavi, D., & Sander, J. (2013). Density-Based Clustering Based on Hierarchical Density Estimates. In Advances in Knowledge Discovery and Data Mining (PAKDD)."
- **[Campello et al. 2013][campello2013]** — Campello, R. J. G. B., Moulavi, D., & Sander, J. (2013). *Density-Based Clustering Based on Hierarchical Density Estimates.* PAKDD. doi:10.1007/978-3-642-37456-2_14

[mcinnes2017]: https://doi.org/10.21105/joss.00205 "McInnes, L., Healy, J., & Astels, S. (2017). hdbscan: Hierarchical density based clustering. JOSS 2(11), 205."
- **[McInnes et al. 2017][mcinnes2017]** — McInnes, L., Healy, J., & Astels, S. (2017). *hdbscan: Hierarchical density based clustering.* Journal of Open Source Software, 2(11), 205. doi:10.21105/joss.00205

[silverman1986]: https://www.routledge.com/Density-Estimation-for-Statistics-and-Data-Analysis/Silverman/p/book/9780412246203 "Silverman, B. W. (1986). Density Estimation for Statistics and Data Analysis. Chapman & Hall."
- **[Silverman 1986][silverman1986]** — Silverman, B. W. (1986). *Density Estimation for Statistics and Data Analysis.* Chapman & Hall.

[wand1995]: https://www.routledge.com/Kernel-Smoothing/Wand-Jones/p/book/9780412552700 "Wand, M. P., & Jones, M. C. (1995). Kernel Smoothing. Chapman & Hall."
- **[Wand & Jones 1995][wand1995]** — Wand, M. P., & Jones, M. C. (1995). *Kernel Smoothing.* Chapman & Hall.

[okabe2012]: https://doi.org/10.1002/9781119967101 "Okabe, A., & Sugihara, K. (2012). Spatial Analysis Along Networks. Wiley."
- **[Okabe & Sugihara 2012][okabe2012]** — Okabe, A., & Sugihara, K. (2012). *Spatial Analysis Along Networks: Statistical and Computational Methods.* Wiley. doi:10.1002/9781119967101

[getis1992]: https://doi.org/10.1111/j.1538-4632.1992.tb00261.x "Getis, A., & Ord, J. K. (1992). The Analysis of Spatial Association by Use of Distance Statistics. Geographical Analysis 24(3)."
- **[Getis & Ord 1992][getis1992]** — Getis, A., & Ord, J. K. (1992). *The Analysis of Spatial Association by Use of Distance Statistics.* Geographical Analysis, 24(3), 189–206.

[ord1995]: https://doi.org/10.1111/j.1538-4632.1995.tb00912.x "Ord, J. K., & Getis, A. (1995). Local Spatial Autocorrelation Statistics: Distributional Issues and an Application. Geographical Analysis 27(4)."
- **[Ord & Getis 1995][ord1995]** — Ord, J. K., & Getis, A. (1995). *Local Spatial Autocorrelation Statistics: Distributional Issues and an Application.* Geographical Analysis, 27(4), 286–306.

[bh1995]: https://doi.org/10.1111/j.2517-6161.1995.tb02031.x "Benjamini, Y., & Hochberg, Y. (1995). Controlling the False Discovery Rate. JRSS B 57(1)."
- **[Benjamini & Hochberg 1995][bh1995]** — Benjamini, Y., & Hochberg, Y. (1995). *Controlling the False Discovery Rate: A Practical and Powerful Approach to Multiple Testing.* Journal of the Royal Statistical Society, Series B, 57(1), 289–300.

#### Networks, routing, geometry
[boeing2017]: https://doi.org/10.1016/j.compenvurbsys.2017.05.004 "Boeing, G. (2017). OSMnx. Computers, Environment and Urban Systems 65."
- **[Boeing 2017][boeing2017]** — Boeing, G. (2017). *OSMnx: New methods for acquiring, constructing, analyzing, and visualizing complex street networks.* Computers, Environment and Urban Systems, 65, 126–139.

[yen1971]: https://doi.org/10.1287/mnsc.17.11.712 "Yen, J. Y. (1971). Finding the k Shortest Loopless Paths in a Network. Management Science 17(11)."
- **[Yen 1971][yen1971]** — Yen, J. Y. (1971). *Finding the k Shortest Loopless Paths in a Network.* Management Science, 17(11), 712–716.

[networkx]: https://www.osti.gov/biblio/960616 "Hagberg, A., Schult, D., & Swart, P. (2008). Exploring network structure, dynamics, and function using NetworkX. Proceedings of SciPy 2008 (LANL/OSTI 960616)."
- **[Hagberg et al. 2008][networkx]** — Hagberg, A., Schult, D., & Swart, P. (2008). *Exploring network structure, dynamics, and function using NetworkX.* In Proceedings of the 7th Python in Science Conference (SciPy 2008).

#### Surveillance-research precedents
[stanford2021]: https://doi.org/10.1145/3461702.3462525 "Sheng, Yao, & Goel (2021). Surveilling Surveillance: Estimating the Prevalence of Surveillance Cameras with Street View Data. AAAI/ACM AIES."
- **[Sheng, Yao, & Goel 2021][stanford2021]** — Sheng, H., Yao, K., & Goel, S. (2021). *Surveilling Surveillance: Estimating the Prevalence of Surveillance Cameras with Street View Data.* Proceedings of the 2021 AAAI/ACM Conference on AI, Ethics, and Society (AIES). doi:10.1145/3461702.3462525

[amnesty]: https://banthescan.amnesty.org/decode/ "Amnesty International (2022). Decode Surveillance NYC."
- **[Amnesty International 2022][amnesty]** — Amnesty International. *Decode Surveillance NYC.* Crowdsourced camera-mapping project, Ban the Scan campaign.



# Installation

## Prerequisites

- Python 3.11
- `uv` package manager

### Install Python 3.11

- For macOS
  - Use HomeBrew package manager. Install HomeBrew following these [instructions](https://brew.sh).
  
    ```commandline
    brew install python@3.11
    ```
- For Ubuntu
  - You can utilize the [Deadsnakes PPA](https://launchpad.net/~deadsnakes/+archive/ubuntu/ppa).
    
    ```commandline
    sudo add-apt-repository ppa:deadsnakes/ppa
    ```
  - Update the package list.
    
    ```commandline
    sudo apt update
    ```
  - Install Python 3.11.
    
    ```commandline
    sudo apt install python3.11
    ```
  - Verify the installation.
    
    ```commandline
    python3.11 --version
    ```
    
### Install `uv`

```commandline
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Create and activate virtual environment

```commandline
uv venv --python 3.11
source .venv/bin/activate
```

### Add dependencies:

```commandline
uv add name-of-dependency
```

### Synchronize dependencies

```commandline
uv sync
```

## Testing:
In order to run the tests from the root project run:

```commandline
bash ./local_test_pipeline.sh
```


## Code formatting

This project uses [.pre-commit](https://pre-commit.com) hooks to ensure universal code formatting.

To install these use:

```commandline
pre-commit install
```

## Ollama client

The application uses Ollama for interacting with LLMs locally.

In order for this to work follow these steps:

1. Create `.env` file at the root of the project. See [`.env-sample`](.env-sample) for the exact naming and properties.

2. Download and install [Ollama](https://ollama.com/download).

3. Open your terminal and execute the following command:

- Download the model:

```commandline
ollama pull llama3:latest
```
- Start Ollama:

```commandline
ollama serve
```

## Usage

The system provides a rich CLI interface for running surveillance analysis:

### Basic Usage

```bash
# Analyze a city with basic settings
python main.py Berlin

# Specify country for disambiguation
python main.py Athens --country GR

# Use a different scenario preset
python main.py Hamburg --scenario full

# Layer per-output toggles on top of a preset (basic + heatmap, no charts)
python main.py Munich --scenario basic --heatmap --no-charts
```

### Analysis Scenarios

Two presets are available:

- `basic` (default): enriched data + summary statistics
- `full`: every output enabled (heatmap, hotspots, charts, stats)

Override individual outputs from either preset using CLI toggle flags
(or the `overrides` field on the API request body):

- `--heatmap` / `--no-heatmap`
- `--hotspots` / `--no-hotspots` (DBSCAN clusters + scatter plot)
- `--charts` / `--no-charts` (privacy pie + zone-sensitivity + sensitivity-reasons)

> **Migration note:** the previous `quick`, `report`, and `mapping` scenarios
> were removed. `quick` was identical to `basic`; `report` and `mapping` are
> now expressed as `basic` plus the relevant `--charts` or
> `--heatmap`/`--hotspots` toggles (or the API `overrides` block).
> Requests using the removed values now return HTTP 422.

### Low-Surveillance Routing

The system can compute privacy-preserving walking routes that minimize exposure to surveillance cameras. Routes are calculated using k-shortest paths algorithms and scored based on camera density within a configurable buffer radius.

**Basic Routing:**

```bash
# Compute a low-surveillance route between two coordinates
python main.py Lund \
  --country SE \
  --enable-routing \
  --start-lat 55.709400 \
  --start-lon 13.194381 \
  --end-lat 55.705962 \
  --end-lon 13.182304
```

**Using Existing Data:**

```bash
# Skip scraping and use cached camera data
python main.py Malmö \
  --country SE \
  --data-path overpass_data/malmö/malmö.json \
  --skip-scrape \
  --enable-routing \
  --start-lat 55.595650 \
  --start-lon 13.022659 \
  --end-lat 55.594801 \
  --end-lon 13.000557
```

**Routing Features:**
- **k-shortest paths**: Evaluates multiple candidate routes (default: 3)
- **Exposure scoring**: Cameras per kilometer metric for route comparison
- **Baseline comparison**: Shows how much safer the route is vs. shortest path
- **Interactive maps**: Folium-based HTML visualizations with route and cameras
- **Graph caching**: OSMnx pedestrian networks cached locally for fast re-computation
- **Result caching**: Routes cached in agent memory for identical requests

**Note:** First-time routing for a city will download the pedestrian network from OSM, which can take several minutes for large cities. Subsequent routes in the same city will be much faster.

### Advanced Options

```bash
# Skip scraping (use existing data)
python main.py Berlin --data-path overpass_data/berlin/berlin.json --skip-scrape

# Skip analysis (scraping only)
python main.py Hamburg --skip-analyze

# Custom output directory
python main.py Paris --output-dir /custom/path

# Verbose logging (helpful for debugging routing performance)
python main.py London --verbose

# Combine routing with full analysis
python main.py Berlin \
  --scenario full \
  --enable-routing \
  --start-lat 52.52 \
  --start-lon 13.40 \
  --end-lat 52.50 \
  --end-lon 13.42
```

### Output Files

The system generates files in `overpass_data/<city>/` organized by function:

**Analysis Outputs:**
- **Enriched JSON** (`<city>_enriched.json`): Original data enhanced with LLM analysis
- **GeoJSON** (`<city>_enriched.geojson`): Geographic data for mapping applications
- **Heatmap** (`<city>_heatmap.html`): Interactive spatial density visualization
- **Hotspots** (`hotspots_<city>.geojson`, `hotspot_plot_<city>.png`): DBSCAN clustering results
- **Statistics** (`stats_chart_<city>.png`): Summary charts and metrics

**Routing Outputs** (in `routes/` subdirectory):
- **Route GeoJSON** (`route_<hash>.geojson`): Route geometry with exposure metrics and nearby camera IDs
- **Route Map** (`route_<hash>.html`): Interactive Folium map with:
  - Low-surveillance route (blue line)
  - Start/end markers (green/red)
  - Camera coverage circles (semi-transparent red)
  - Route metrics tooltip (length, exposure score)

**Cache Files:**
- **OSM Graphs** (`.graph_cache/<hash>.graphml`): Cached pedestrian networks
- **Agent Memory** (`memory.db`): SQLite database storing route and query caches

## FastAPI Web Interface

In addition to the CLI, the system provides a production-ready REST API for programmatic access to all functionality.

### Running the API Server

**Development Mode:**
```bash
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8080
```

**Production Mode:**
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8080 --workers 4
```

**Access Documentation:**
- Swagger UI: `http://localhost:8080/docs`
- ReDoc: `http://localhost:8080/redoc`
- OpenAPI spec: `http://localhost:8080/openapi.json`

### API Features

- **Asynchronous execution**: Long-running jobs processed in background tasks
- **Real-time progress**: WebSocket endpoint for live pipeline updates
- **Task management**: Full CRUD operations on analysis jobs
- **File serving**: Direct access to generated GeoJSON, maps, and visualizations
- **Type safety**: Pydantic validation on all requests and responses
- **Auto-documentation**: Complete OpenAPI spec with interactive examples

### API Endpoints

#### Health & System

```http
GET /health
```
Returns service health status.

**Example Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-12-05T10:30:00Z",
  "service": "Agentic Surveillance Research API"
}
```

```http
GET /version
```
Returns API version information.

#### Pipeline Execution

```http
POST /api/v1/pipeline/run
```
Start a complete pipeline job (scraping + analysis + optional routing).

**Example Request:**
```json
{
  "city": "Berlin",
  "country": "DE",
  "scenario": "basic"
}
```

**With Routing:**
```json
{
  "city": "Lund",
  "country": "SE",
  "scenario": "full",
  "routing_config": {
    "city": "Lund",
    "country": "SE",
    "start_lat": 55.7047,
    "start_lon": 13.1910,
    "end_lat": 55.7058,
    "end_lon": 13.1932
  }
}
```

**Response:**
```json
{
  "task_id": "abc123",
  "status": "pending",
  "message": "Pipeline started for Berlin"
}
```

```http
GET /api/v1/pipeline/{task_id}
```
Get status and results for a pipeline job.

**Response (Running):**
```json
{
  "id": "abc123",
  "type": "pipeline",
  "status": "running",
  "progress": 50,
  "created_at": "2025-12-05T10:30:00Z",
  "started_at": "2025-12-05T10:30:01Z",
  "metadata": {
    "city": "Berlin",
    "scenario": "basic"
  }
}
```

**Response (Completed):**
```json
{
  "id": "abc123",
  "type": "pipeline",
  "status": "completed",
  "progress": 100,
  "result": {
    "city": "Berlin",
    "status": "completed",
    "scrape": { "success": true, "elements_count": 150 },
    "analyze": { "success": true, "element_count": 150 },
    "routing": { "success": true, "length_m": 1523.4, "exposure_score": 2.3 }
  },
  "created_at": "2025-12-05T10:30:00Z",
  "completed_at": "2025-12-05T10:32:15Z"
}
```

```http
POST /api/v1/pipeline/{task_id}/cancel
```
Cancel a running pipeline job.

```http
DELETE /api/v1/pipeline/{task_id}
```
Delete a pipeline job and its results.

#### File Outputs

```http
GET /api/v1/outputs/{city}/geojson?enriched=true
```
Download enriched GeoJSON file for a city.

```http
GET /api/v1/outputs/{city}/map?map_type=heatmap
```
Get interactive HTML heatmap. Options: `heatmap`, `hotspots`.

```http
GET /api/v1/outputs/{city}/route?format=map
```
Get route visualization. Formats: `map` (HTML), `geojson`.

```http
GET /api/v1/outputs/{city}/stats?format=json
```
Get statistics. Formats: `json`, `chart` (PNG).

```http
GET /api/v1/outputs/{city}/list
```
List all available files for a city with metadata.

**Example Response:**
```json
{
  "city": "Berlin",
  "file_count": 8,
  "files": [
    {
      "name": "Berlin_enriched.geojson",
      "path": "/outputs/Berlin_enriched.geojson",
      "size_bytes": 245678,
      "modified": 1733395200.0,
      "type": "application/geo+json"
    }
  ]
}
```

```http
GET /api/v1/outputs/file/{filename}
```
Generic file access by filename.

#### Real-Time Progress (WebSocket)

```http
WS /ws/tasks/{task_id}
```
WebSocket endpoint for real-time pipeline progress updates.

**Example Messages:**

```json
{
  "type": "progress",
  "stage": "scraping",
  "progress": 20,
  "message": "Downloading surveillance data from OpenStreetMap",
  "timestamp": "2025-12-05T10:30:05Z"
}
```

```json
{
  "type": "completed",
  "stage": "completed",
  "progress": 100,
  "message": "Pipeline completed successfully",
  "timestamp": "2025-12-05T10:32:15Z"
}
```

### API Usage Examples

#### Using curl

**Start a pipeline:**
```bash
curl -X POST http://localhost:8080/api/v1/pipeline/run \
  -H "Content-Type: application/json" \
  -d '{
    "city": "Berlin",
    "country": "DE",
    "scenario": "basic"
  }'
```

**Check status:**
```bash
curl http://localhost:8080/api/v1/pipeline/abc123
```

**Download GeoJSON:**
```bash
curl http://localhost:8080/api/v1/outputs/Berlin/geojson > berlin.geojson
```

#### Using Python

```python
import requests
import time

# Start pipeline
response = requests.post(
    "http://localhost:8080/api/v1/pipeline/run",
    json={
        "city": "Athens",
        "country": "GR",
        "scenario": "full",
        "routing_config": {
            "city": "Athens",
            "country": "GR",
            "start_lat": 37.9838,
            "start_lon": 23.7275,
            "end_lat": 37.9755,
            "end_lon": 23.7348
        }
    }
)
task_id = response.json()["task_id"]

# Poll for completion
while True:
    status = requests.get(f"http://localhost:8080/api/v1/pipeline/{task_id}").json()
    print(f"Progress: {status['progress']}%")

    if status["status"] in ["completed", "failed"]:
        break

    time.sleep(2)

# Get results
if status["status"] == "completed":
    results = status["result"]
    print(f"Route length: {results['routing']['length_m']}m")
    print(f"Exposure score: {results['routing']['exposure_score']} cameras/km")
```

#### Using JavaScript/WebSocket

```javascript
// Connect to WebSocket for real-time updates
const ws = new WebSocket('ws://localhost:8080/ws/tasks/abc123');

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log(`${data.stage}: ${data.progress}%`);

  if (data.type === 'completed') {
    console.log('Pipeline finished!');
    ws.close();
  }
};

// Send periodic ping to keep connection alive
setInterval(() => ws.send('ping'), 5000);
```

### Docker Deployment

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install uv
RUN pip install uv

# Copy project files
COPY pyproject.toml .
COPY src ./src

# Install dependencies
RUN uv sync --no-dev

# Expose port
EXPOSE 8080

# Run server
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8080"]
```

**Build and run:**
```bash
docker build -t surveillance-api .
docker run -p 8080:8080 surveillance-api
```

### API Testing

Run comprehensive API test suite:
```bash
bash ./api_local_test_pipeline.sh
```

This runs 67 tests covering:
- Health endpoints (6 tests)
- Pydantic models (10 tests)
- Pipeline endpoints (12 tests)
- Task manager (11 tests)
- WebSocket (8 tests)
- Output file serving (20 tests)

## Architecture

### Agent-Based Design

The system follows a perceive-plan-act pattern with three specialized agents:

1. **SurveillanceDataCollector** (Scraper)
   - Perceives: City name and query parameters
   - Plans: Check cache, determine if scraping needed
   - Acts: Query Overpass API, save results, update cache

2. **SurveillanceAnalyzerAgent** (Analyzer)
   - Perceives: Raw surveillance data from scraper
   - Plans: Select analysis workflow based on scenario
   - Acts: Enrich with LLM, generate visualizations, compute statistics

3. **RouteFinderAgent**
   - Perceives: Start/end coordinates, enriched camera data
   - Plans: Check route cache, determine computation steps
   - Acts: Build graph, generate k-shortest paths, score exposure, select optimal route

### Routing Algorithm

The routing system uses a multi-step approach to find privacy-preserving routes:

1. **Graph Construction**: OSMnx downloads walkable street network from OpenStreetMap
2. **Node Snapping**: Start/end coordinates snapped to nearest graph nodes (500m threshold)
3. **Path Generation**: NetworkX k-shortest simple paths algorithm generates candidate routes
4. **Exposure Scoring**:
   - Buffer each route by configurable radius (default: 50m)
   - Use GeoDataFrame spatial join to count cameras within buffer
   - Calculate exposure as cameras/km
5. **Route Selection**: Choose path with minimum exposure score
6. **Baseline Comparison**: Compare against shortest path to quantify privacy gain

**Performance Optimizations:**
- OSM graphs cached to disk (avoiding repeated downloads)
- Camera GeoDataFrame built once and reused across all candidate paths
- Routes cached in agent memory by (city, coordinates, settings) hash

### Configuration

Route computation can be customized via `src/config/settings.py`:

```python
class RouteSettings:
    max_candidates: int = 3           # Number of alternative paths to evaluate
    buffer_radius_m: float = 50.0     # Camera detection radius in meters
    network_type: str = "walk"        # OSMnx network type
    snap_distance_threshold_m: float = 500.0  # Max distance to snap coordinates
```

## Troubleshooting

### Routing Performance

**Symptom:** First routing attempt for a city takes 10-30+ minutes

**Cause:** OSMnx is downloading the entire pedestrian network from OpenStreetMap

**Solution:**
- Use `--verbose` flag to confirm it's the graph download step
- Be patient - this only happens once per city (results are cached)
- For large cities like Malmö, consider testing with closer coordinates first

**Performance Tips:**
- Test with points 500m-1km apart before trying longer routes
- Use `--data-path` and `--skip-scrape` to skip analysis when testing routes
- Check `overpass_data/.graph_cache/` to see which cities are already cached

### Coordinate Snapping Errors

**Error:** `Cannot snap (lat, lon) to walkable network: nearest node is XXXm away`

**Cause:** Coordinates are not near any walkable paths (e.g., middle of water, private property)

**Solution:**
- Verify coordinates using OpenStreetMap
- Ensure coordinates are on or near streets/sidewalks
- Try coordinates closer to known roads

### Help

```bash
python main.py --help
```

## Contributing

This project uses:
- **uv** for dependency management
- **pytest** for testing
- **pre-commit** hooks for code formatting
- **ruff** for linting

Run cli tests with:
```bash
bash ./cli_local_test_pipeline.sh
```

Run api tests with:
```bash
bash ./api_local_test_pipeline.sh
```
