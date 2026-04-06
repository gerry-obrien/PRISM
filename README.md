# PRISM — Paris Real estate Intelligent Smart Market

## What is PRISM?

Buying property in Paris is hard. Prices vary wildly from one arrondissement to another, listings don't always reflect fair value, and figuring out if you're getting a good deal usually requires either years of experience or an expensive advisor.

PRISM is our attempt to solve that with data. The idea is simple: we built a machine learning model that estimates the fair price of any apartment in Paris based on its features (location, size, floor, energy rating, etc.). We then compare the model's estimate to the actual asking price and flag whether a listing is **overpriced**, **fairly priced**, or a **bargain**.

On top of that, we built a web interface where a potential buyer can:
- Browse and filter 1,000 Paris property listings with instant valuation labels
- Estimate the value of their own apartment based on its features
- Chat with an AI-powered investment advisor that knows the market data and can help think through investment decisions

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                        PRISM                            │
│                                                         │
│  ┌──────────────┐    ┌──────────────┐    ┌───────────┐  │
│  │  ML Pipeline │    │   Backend    │    │ Frontend  │  │
│  │              │    │   (FastAPI)  │    │(Streamlit)│  │
│  │ generate_data│───▶│              │◀───│           │  │
│  │ train model  │    │ /api/listings│    │ Explorer  │  │
│  │              │    │ /api/predict │    │ Advisor   │  │
│  │ W&B + MLflow │    │ /api/chat   │    │ Auth      │  │
│  └──────────────┘    └──────┬───────┘    └───────────┘  │
│                             │                           │
│                      ┌──────▼───────┐                   │
│                      │   Ollama     │                   │
│                      │  (Local LLM) │                   │
│                      └──────────────┘                   │
└─────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
prism/
├── docker-compose.yml          # Orchestrates all services
├── .dockerignore
├── .env.example                # Environment variables template
├── data/                       # Generated CSVs
├── ml/
│   ├── Dockerfile
│   ├── generate_data.py        # Creates synthetic datasets
│   ├── train.py                # Trains, evaluates, predicts listings
│   ├── requirements.txt
│   ├── artifacts/              # Saved model + metrics
│   ├── plots/                  # Evaluation charts
│   └── mlflow-artifacts/       # MLflow artifact store
├── backend/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── app/
│       ├── main.py             # FastAPI entry point
│       ├── models/             # Request/response schemas
│       ├── routers/            # API endpoints (listings, predict, chatbot)
│       └── services/           # Model loading, inference, chatbot logic
├── frontend/
│   ├── Dockerfile
│   ├── app.py                  # Streamlit UI
│   ├── data.py                 # Data loading helpers
│   └── components.py           # Sidebar filter components
├── mlruns/                     # MLflow run metadata
└── wandb/                      # W&B local data
```

---

## How to Run It

You have two options: **Docker** (recommended, one command) or **manual setup** (step by step).

---

### Option A — Docker (easier)

Docker bundles everything so you don't have to install dependencies manually. You just need Docker Desktop installed and make it run beofre starting to build.

**Prerequisites:**
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running
- [Ollama](https://ollama.com) installed on your machine (for the chatbot — see the Ollama section below)

**1. Clone and configure:**

```bash
git clone https://github.com/gerry-obrien/PRISM.git
cd PRISM
cp .env.example .env
# Edit .env if needed (W&B API key see Weight and Biases section later if needed, MLflow URI)
```

**2. Launch everything:**

```bash
docker compose up --build
```

This builds and starts three containers in order: first the ML pipeline (generates data + trains the model), then the backend (FastAPI), then the frontend (Streamlit). The whole process takes a few minutes the first time.

**3. Open the app:**

- Frontend UI: [http://localhost:8501](http://localhost:8501)
- API docs (Swagger): [http://localhost:8000/docs](http://localhost:8000/docs)

To stop everything: `docker compose down`

---

### Option B — Manual Setup (without Docker)

You need Python 3.10 or higher.

**1. Clone and set up a virtual environment:**

```bash
git clone https://github.com/gerry-obrien/PRISM.git
cd PRISM
```

macOS / Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows (PowerShell):
```bash
py -m venv .venv
.venv\Scripts\Activate.ps1
```

**2. Generate the data:**

```bash
pip install -r ml/requirements.txt
python ml/generate_data.py
```

This creates two files in `data/`:
- `training_data.csv` — 100,000 past Paris property sales (2015–2024)
- `listings.csv` — 1,000 properties currently for sale, with no price estimate yet

The script uses a fixed seed so the output is always identical.

**3. Configure environment variables:**

```bash
cp .env.example .env
```

Open `.env` and set your tracking server. The shared team server (`https://mlflow.lewagon.ai`) is the default — you shouldn't need to change anything unless you want a local server.

**Running a local MLflow server instead (optional):**

If you prefer not to use the shared server, update `.env` to `MLFLOW_TRACKING_URI=http://127.0.0.1:5000` and start a local server:

```bash
mlflow server --backend-store-uri ./mlruns --default-artifact-root ./ml/mlflow-artifacts --host 127.0.0.1 --port 5000 --workers 1
```

The local UI is available at [http://127.0.0.1:5000](http://127.0.0.1:5000).

**4. Train the model:**

```bash
python ml/train.py
```

This trains a Random Forest on the training data, evaluates it, then runs predictions on all 1,000 listings.

You'll see something like this:
```
R² score:                    ~0.83
Mean Absolute Error (MAE):   ~€113,000
Mean Abs % Error (MAPE):     ~20%
```

The 20% error comes from the way we generate the data — it attempts to imitate random variance in real housing market prices.

**Where outputs are saved:**

| Output | Location | Purpose |
| --- | --- | --- |
| Trained model | `ml/artifacts/model.pkl` | Loaded by the API to serve predictions |
| Evaluation metrics | `ml/artifacts/metrics.json` | Local copy of R², MAE, MAPE |
| Evaluation plots | `ml/plots/` | 4 evaluation charts (gitignored) |
| Listing predictions | `data/listings_predicted.csv` | Listings with estimated prices and valuation labels |

**5. Start the API:**

```bash
pip install -r backend/requirements.txt
uvicorn app.main:app --reload --port 8000 --app-dir backend
```

Once it says "Application startup complete", go to [http://localhost:8000/docs](http://localhost:8000/docs) to see all the endpoints.

**6. Start the frontend:**

In a separate terminal (with the venv activated):

```bash
streamlit run frontend/app.py
```

The UI opens at [http://localhost:8501](http://localhost:8501).

---

## Setting Up Ollama (LLM Chatbot)

The Investment Advisor tab in the frontend uses a local LLM through [Ollama](https://ollama.com). This is optional — the rest of the app (Property Explorer, API, predictions) works fine without it.

**Install Ollama:**

Download and install from [ollama.com](https://ollama.com). On Windows, it runs as a background service after installation. On macOS/Linux, you may need to run `ollama serve` in a separate terminal.

**If you're on WSL (Windows Subsystem for Linux):** Install Ollama on the Windows side (not inside WSL). The Windows install auto-detects your GPU and WSL can access it through `localhost:11434` since they share the network.

**Pull the model:**

```bash
ollama pull llama3.2:1b
```

This downloads a ~1.3 GB model. We went with a small local model to keep it free and dependency-light — no API keys, no cloud costs.

**Verify it works:**

```bash
curl http://localhost:11434/api/tags
```

You should see `llama3.2:1b` in the list.

**Docker note:** When running with Docker, Ollama stays on the host machine (not in a container). The backend container reaches it through `host.docker.internal:11434` — this is handled automatically by the `docker-compose.yml` environment variable.

---

## What Each Part Does

### The Data (`ml/generate_data.py`)

We use synthetic data instead of real transaction records. Each property's price comes from a formula that combines things like location (arrondissement), size, floor, elevator, energy rating, building condition, and whether it's a new build — all with realistic Paris market weightings. On top of that, every price gets multiplied by a random noise factor of about ±20%, because that's how housing markets actually work.

### The Model (`ml/train.py`)

A Random Forest regressor, with a scikit-learn Pipeline so the preprocessing (scaling, encoding) is bundled with the model itself.

We picked Random Forest because it handles the mix of feature types well (continuous, ordinal, categorical), catches interactions the data has (like high floors being a premium with an elevator but a penalty without one), and gives us feature importance scores we can actually interpret.

The script generates plots — feature importance, actual vs predicted scatter, residual distribution, and error broken down by arrondissement — so you can see at a glance where the model does well and where it struggles.

### The API (`backend/`)

A FastAPI server that sits between the model and the frontend.

| Method | Endpoint | What it does |
| --- | --- | --- |
| `GET` | `/api/listings` | Browse listings with filters (arrondissement, price, DPE, valuation, etc.) |
| `GET` | `/api/listings/{id}` | Get one listing |
| `GET` | `/api/listings/stats/summary` | Aggregate stats across all listings |
| `POST` | `/api/predict` | Send property features, get a price estimate |
| `GET` | `/api/predict/metrics` | Model performance metrics |
| `POST` | `/api/chat` | Send a message to the investment advisor chatbot |

### The Investment Advisor (`backend/app/services/chatbot.py`)

A conversational chatbot that helps users explore the listings data and think through investment decisions. It connects to a local LLM running through Ollama and feeds it a pre-computed summary of the market data — averages per arrondissement, valuation breakdowns, top deals — so it can answer questions without seeing the full dataset.

The chatbot can answer things like which arrondissements have the most undervalued properties, what a reasonable rental yield looks like for a given area, or roughly when you'd break even on an investment. It uses simple assumptions for rental yield (~3.5% gross in Paris), annual price growth (~1.5%), and typical charges (~25% of rent) to give ballpark estimates.

The flow is: user types in Streamlit → Streamlit calls `POST /api/chat` → FastAPI passes the message to the chatbot service → the service builds a data summary from the listings, constructs a system prompt, and sends everything to Ollama → the LLM response comes back through the chain.

### The Frontend (`frontend/`)

A Streamlit UI with two tabs:

- **Property Explorer** — filter listings by arrondissement, price range, area, number of rooms, and energy rating. Results are displayed in a table with a map showing property locations.
- **Investment Advisor** — chat interface where users can ask questions about the market data, find undervalued listings, or get rough investment return estimates. Conversation history is preserved within the session.

The app also includes user authentication (login/logout) and the ability for users to estimate the value of their own apartment.

---

## Experiment Tracking

### Weights & Biases (W&B)

Metrics are always logged to W&B. To sync to the cloud dashboard:

1. Create an account at [wandb.ai](https://wandb.ai)
2. Add your API key to `.env`: `WANDB_API_KEY=your_key_here`
3. W&B will automatically sync on the next training run

Without an API key, W&B runs in offline mode and stores data locally in `wandb/`.

### MLflow

Each training and retraining run logs parameters, metrics, and evaluation plots to MLflow.

**What gets logged:**

| Item | Where to find it |
| --- | --- |
| Run parameters (model type, test size, etc.) | Experiment → Run → Parameters |
| Metrics (R², MAE, MAPE) | Experiment → Run → Metrics |
| Evaluation plots (4 PNG charts) | Experiment → Run → Artifacts → plots/ |
| Model registry entry | Models tab → prism-model |

### Known Issues with MLflow + Le Wagon Server

We ran into two compatibility issues between the MLflow Python client (2.13+) and the Le Wagon shared server (older MLflow version):

**1. Model file upload is disabled.** The Le Wagon server returns HTTP 413 (Request Entity Too Large) when attempting to upload the model file, even after compressing it and reducing `max_depth` to bring the size to ~4.5 MB. The server has a small nginx `client_max_body_size` limit. To enable model uploads, the server admin would need to increase this limit. Until then, the model lives locally at `ml/artifacts/model.pkl` and is also stored via W&B.

**2. Model registry uses a lower-level API call.** The high-level `mlflow.register_model()` triggers a request to `/api/2.0/mlflow/logged-models/search` — a newer endpoint that doesn't exist on the Le Wagon server (returns 404). We use `MlflowClient.create_model_version()` instead, which calls the older `/model-versions/create` endpoint that the server supports. Each training or retraining run registers a new version under `prism-model` in the Models tab.

---

## Quick API Examples

**Estimate a price:**

```bash
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "area_sqm": 65,
    "num_rooms": 3,
    "arrondissement": 11,
    "property_type": "Apartment",
    "year": 2024,
    "floor": 4,
    "has_elevator": true,
    "dpe_rating": "D",
    "building_condition": "Good",
    "is_new_build": false
  }'
```

**Find undervalued properties in the 18th:**

```bash
curl "http://localhost:8000/api/listings?arrondissement=18&valuation=Undervalued&sort_by=price_delta_pct&sort_order=asc"
```

**Ask the investment advisor:**

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What are the best deals in the 11th arrondissement?",
    "history": []
  }'
```

---

## What We Didn't Do (and Why)

We had to make choices about what to focus on. Here's what we intentionally left out and the reasoning behind it:

- **Prefect / Airflow orchestration** — We looked into adding a proper orchestration pipeline for retraining, but for a project of this scale it felt overkill. Our retraining flow is a single script that runs sequentially. Adding a DAG orchestrator would have added infrastructure complexity without real benefit given the simplicity of our pipeline.
- **Full MLflow model serving** — As described in the Known Issues section, the Le Wagon shared MLflow server has size limits and API incompatibilities that prevented us from uploading model artifacts directly. We worked around it by storing the model locally and via W&B, and using the lower-level MLflow API for the registry. A production setup would need a properly configured MLflow server.
- **Cloud storage (S3, BigQuery)** — We considered hosting data and apartment images on AWS S3 or BigQuery, but since we're working with synthetic data and no real images, it didn't make practical sense. The CSV files are generated deterministically from a script, so there's nothing to persist in the cloud.
- **Interactive map** — The current Streamlit map is basic (just dots on a map using lat/long). A more interactive version with clustering, popups per listing, and neighbourhood overlays would have been nice, but Streamlit's map capabilities are limited and we preferred to focus on the ML pipeline and API rather than frontend polish.
- **DPE rental restrictions** — G-rated properties can't be rented out since 2025, F-rated banned from 2028, E-rated from 2034. This is critical info for buy-to-let investors and would be a natural addition to the valuation logic, but we didn't have time to integrate the regulatory layer.
