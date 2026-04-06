# PRISM - Paris Real estate Intelligent Smart Market

PRISM is an app that helps people buying property in Paris figure out if they're getting a good deal. A machine learning model estimates fair prices for 1,000 properties currently on the market, then flags which ones are overpriced and which are bargains. The application comes with a UI (Streamlit for now) to help the customer select their criteria for buying a property. Moreover, the synthetic data on which the model is trained can be used as continuous development. Moreover, the app allows the customer to price its own appartment based on different features, this helps to increase the database of properties listed and for later development we want to provide investment opportunity based on the listed property.

---

## How to run it

You need Python 3.10 or higher.

### 1. Clone and set up

```bash
git clone https://github.com/YOUR_USERNAME/prism.git
cd prism
```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows** — use the `py` launcher (comes with the [python.org](https://www.python.org/downloads/) installer):
```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1         # PowerShell
# or
.venv\Scripts\activate.bat         # Command Prompt
```

### 2. Generate the data

```bash
pip install -r ml/requirements.txt
python ml/generate_data.py
```

This creates two files in `data/`:
- `training_data.csv` - 100,000 past Paris property sales (2015–2024)
- `listings.csv` - 1,000 properties currently for sale, with no price estimate yet

The script uses a fixed seed so the output is always identical.

### 3. Configure MLflow

MLflow tracks every training run — parameters, metrics, and the model artifact.

Copy the example environment file:

**macOS / Linux:**
```bash
cp .env.example .env
```

**Windows (PowerShell):**
```powershell
Copy-Item .env.example .env
```

Open `.env` and set your tracking server. The shared team server is already set as the default — you should not need to change anything.

**Running a local server instead (optional):**

If you prefer not to use the shared server, update `.env` to `MLFLOW_TRACKING_URI=http://127.0.0.1:5000` and start a local server in a separate terminal:

**macOS / Linux:**
```bash
mlflow server --backend-store-uri ./mlruns --default-artifact-root ./ml/mlflow-artifacts --host 127.0.0.1 --port 5000 --workers 1
```

**Windows (PowerShell):**
```powershell
mlflow server --backend-store-uri ./mlruns --default-artifact-root ./ml/mlflow-artifacts --host 127.0.0.1 --port 5000 --workers 1
```

The local UI is available at [http://127.0.0.1:5000](http://127.0.0.1:5000).

### 4. Train the model

```bash
python ml/train.py
```

This trains a Random Forest on the training data, tests it, then runs it against all 1,000 listings to fill in estimated prices.

You'll see something like this in the terminal:

```
  R² score:                    ~0.83
  Mean Absolute Error (MAE):   ~€113,000
  Mean Abs % Error (MAPE):     ~20%
```

The 20% error comes from the way we generate the data which attempts to imitate random variance in housing market prices.

**Where outputs are saved:**

| Output | Location | Purpose |
|--------|----------|---------|
| Trained model | `ml/artifacts/model.pkl` | Loaded by the API to serve predictions |
| Evaluation metrics | `ml/artifacts/metrics.json` | Local copy of R², MAE, MAPE |
| Evaluation plots | `ml/plots/` | Local copies of the 4 charts (gitignored) |
| Listing predictions | `data/listings_predicted.csv` | Listings with estimated prices and valuation labels |

**What gets logged to MLflow (`https://mlflow.lewagon.ai`):**

| Item | Where to find it |
|------|-----------------|
| Run parameters (model type, test size, etc.) | Experiment → Run → Parameters |
| Metrics (R², MAE, MAPE) | Experiment → Run → Metrics |
| Evaluation plots (4 PNG charts) | Experiment → Run → Artifacts → plots/ |
| Model registry entry | Models tab → prism-model |

> **Note on model uploads and registry (debugged with Claude Code):**
>
> Two compatibility issues were discovered between the MLflow Python client (2.13+) and the Le Wagon shared server (older MLflow version):
>
> **1. Model file upload (`model.pkl`) is disabled.** The Le Wagon server returns HTTP 413 (Request Entity Too Large) when attempting to upload the model file, even after compressing it and reducing `max_depth` to bring the size to ~4.5 MB. The server has a small nginx `client_max_body_size` limit. To enable model uploads, the server admin needs to increase this limit. Until then, the model lives locally at `ml/artifacts/model.pkl` and is also stored via W&B.
>
> **2. Model registry uses the lower-level `MlflowClient.create_model_version()` instead of `mlflow.register_model()`.** The high-level `register_model()` call triggers a follow-up request to `/api/2.0/mlflow/logged-models/search` — a newer endpoint that doesn't exist on the Le Wagon server (returns 404). The lower-level `create_model_version()` uses the older `/model-versions/create` endpoint which the server supports. Each training or retraining run registers a new version under `prism-model` in the Models tab.

**What gets logged to W&B:**

Metrics are always logged to W&B. To also sync the model artifact and view results in the W&B cloud dashboard:

1. Create an account at [wandb.ai](https://wandb.ai)
2. Add your API key to `.env`: `WANDB_API_KEY=your_key_here`
3. W&B will automatically sync on the next run

Without an API key, W&B runs in offline mode and stores data locally in `wandb/`.

### 5. Start the API

```bash
pip install -r backend/requirements.txt
uvicorn app.main:app --reload --port 8000 --app-dir backend
```

Once it says "Application startup complete", go to [http://localhost:8000/docs](http://localhost:8000/docs) to see all the endpoints and try them.

### 6. Install Ollama (for the Investment Advisor chatbot)

The Investment Advisor tab in the frontend uses a local LLM through [Ollama](https://ollama.com). Download and install Ollama from their website, then pull a model:

```bash
ollama pull llama3.2:1b
```

On Windows, Ollama runs as a background service after installation so you don't need to start it manually. On macOS/Linux you may need to run `ollama serve` in a separate terminal.

The rest of the app (Property Explorer, API, predictions) works fine without Ollama - only the chatbot tab requires it.

---

## Project structure

```
prism/
├── data/               # Generated CSVs (gitignored - reproducible from scripts)
├── ml/
│   ├── generate_data.py        # Creates the synthetic datasets
│   ├── train.py                # Trains the model, evaluates, predicts listings
│   ├── artifacts/              # Saved model + metrics (gitignored)
│   ├── plots/                  # Evaluation charts (gitignored)
│   └── mlflow-artifacts/       # MLflow artifact store (gitignored)
├── mlruns/             # MLflow run metadata (gitignored - created by mlflow server)
├── backend/
│   └── app/
│       ├── main.py         # FastAPI entry point
│       ├── models/         # Request/response schemas
│       ├── routers/        # API endpoints (listings, predict, chatbot)
│       └── services/       # Model loading + inference + chatbot logic
└── frontend/
    ├── app.py          # Streamlit UI (Property Explorer + Investment Advisor)
    ├── data.py         # Data loading
    └── components.py   # Sidebar filters
```
---

## What each part does

### The data (ml/generate_data.py)

We use synthetic data instead of real transaction records. Each property's price comes from a formula that combines things like location (arrondissement), size, floor, elevator, energy rating, building condition, and whether it's a new build - all with realistic Paris market weightings. On top of that, every price gets multiplied by a random noise factor of about ±20%, because that's how housing markets actually work.

### The model (ml/train.py)

A Random Forest regressor, with a scikit-learn Pipeline so the preprocessing (scaling, encoding) is bundled with the model itself.

We picked Random Forest because it handles the mix of feature types well (continuous, ordinal, categorical), catches interactions the data has (like high floors being a premium with an elevator but a penalty without one), and gives us feature importance scores we can actually interpret.

The script generates plots - feature importance, actual vs predicted scatter, residual distribution, and error broken down by arrondissement - so you can see at a glance where the model does well and where it struggles.

### The API (backend/)

A FastAPI server that sits between the model and the frontend.

Main endpoints:

| Method | Endpoint | What it does |
|--------|----------|--------------|
| `GET` | `/api/listings` | Browse listings with filters (arrondissement, price, DPE, valuation, etc.) |
| `GET` | `/api/listings/{id}` | Get one listing |
| `GET` | `/api/listings/stats/summary` | Aggregate stats across all listings |
| `POST` | `/api/predict` | Send property features, get a price estimate |
| `GET` | `/api/predict/metrics` | How well the model performs |
| `POST` | `/api/chat` | Send a message to the investment advisor chatbot |

### The Investment Advisor (backend/app/services/chatbot.py)

A conversational chatbot that helps users explore the listings data and think through investment decisions. It connects to a local LLM running through Ollama and feeds it a summary of the market data - averages per arrondissement, valuation breakdowns, top deals so it can answer questions without seeing the full dataset.

The chatbot can answer things like which arrondissements have the most undervalued properties, what a reasonable rental yield looks like for a given area, or roughly when you'd break even on an investment. It uses simple assumptions for rental yield (~3.5% gross in Paris), annual price growth (~1.5%), and typical charges (~25% of rent) to give ballpark investment estimates.

The flow is: user types in Streamlit → Streamlit calls `POST /api/chat` → FastAPI passes the message to the chatbot service → the service builds a data summary from the listings, constructs a system prompt, and sends everything to Ollama → the LLM response comes back through the chain.

We went with Ollama + a local model (llama3.2:1b) to keep it free and dependency-light: no API keys, no cloud costs. The tradeoff is that whoever runs the app needs Ollama installed locally.

---

## Quick API examples

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

### The Front-End

In order to launch the front-end (Streamlit), you first need to start the API with Uvicorn. Then, from your terminal at the root of the repository, run:

```bash
streamlit run frontend/app.py
```

### How the Front-End works ?

It uses synthetically created data stored in the data folder of the repository, and it pulls from functions in the data.py and components.py files located in the frontend folder.

It presents a simple Streamlit user interface for a prospective home buyer. With the sidebar on the left, the user can modify the attributes of the property according to their preferences. After submitting the specifications, the displayed dataframe is instantly filtered to reflect the selected attributes.

This version of the interface also includes a simple map that uses the longitude and latitude of the properties to display the locations of the filtered results.

The frontend has two tabs:
- **Property Explorer** - the filtering interface with the sidebar and map described above.
- **Investment Advisor** - a chat interface where users can ask questions about the market data, find undervalued listings, or get rough investment return estimates. Conversation history is preserved within the session.
