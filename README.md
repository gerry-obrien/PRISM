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

### 3. Train the model

```bash
python ml/train.py
```

This trains a Random Forest on the training data, tests it, then runs it against all 1,000 listings to fill in estimated prices. It saves:
- `ml/artifacts/model.pkl` - the trained model
- `ml/artifacts/metrics.json` - evaluation numbers
- `ml/plots/` - charts showing how well the model performs
- `data/listings_predicted.csv` - listings with estimated prices and over/undervalued labels

You'll see something like this in the terminal:

```
  R² score:                    ~0.83
  Mean Absolute Error (MAE):   ~€113,000
  Mean Abs % Error (MAPE):     ~20%
```

The 20% error comes from the way we generate the data which attempts to imitate random variance in housing market prices.

### 4. Start the API

```bash
pip install -r backend/requirements.txt
uvicorn app.main:app --reload --port 8000 --app-dir backend
```

Once it says "Application startup complete", go to [http://localhost:8000/docs](http://localhost:8000/docs) to see all the endpoints and try them.

---

## Project structure

```
prism/
├── data/               # Generated CSVs (gitignored - reproducible from scripts)
├── ml/
│   ├── generate_data.py    # Creates the synthetic datasets
│   ├── train.py            # Trains the model, evaluates, predicts listings
│   ├── artifacts/          # Saved model + metrics (gitignored)
│   └── plots/              # Evaluation charts (gitignored)
└── backend/
    └── app/
        ├── main.py         # FastAPI entry point
        ├── models/         # Request/response schemas
        ├── routers/        # API endpoints
        └── services/       # Model loading + inference
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

### The Front-End

In order to launch the front-end (Streamlit), you first need to start the API with Uvicorn. Then, from your terminal at the root of the repository, run:

```bash
streamlit run frontend/app.py
```

### How the Front-End works ?

It uses synthetically created data stored in the data folder of the repository, and it pulls from functions in the data.py and components.py files located in the frontend folder.

It presents a simple Streamlit user interface for a prospective home buyer. With the sidebar on the left, the user can modify the attributes of the property according to their preferences. After submitting the specifications, the displayed dataframe is instantly filtered to reflect the selected attributes.

This version of the interface also includes a simple map that uses the longitude and latitude of the properties to display the locations of the filtered results.


## What's next

This is Phase 1. Here's what's coming and some ideas we're exploring.

### Phase 2 - Continuous integration and development (CI/CD) and deployment

- **Re-training on new data** - The idea is to develop a retraining method that mimics real-life situations where additional data becomes available over time. We want to be able to append new synthetic data representing more recent listings. Then, the model should be retrained and its performance compared to the previous version.

This implementation requires several steps to ensure a continuous development pipeline:

-- **Add Weight and Biases pipeline** Create a W&B connection to our training in other to monitor the training performances of the models

-- **Adding a MLFLOW pipeline** - Create a MLFLOW server to host the "in production" model and keep the other in archive.

-- **Add an orchestration pipeline** - Create a Prefect or Airflow pipeline to handle the workflow or retraining and model in production through different conditions.

- **Cloud implementation** We would like to add cloud features such as hosting our database on BigQuery or AWS (S3 if images are required). Finally, we plan to dockerize the codebase and deploy it to the cloud.

- **MCP server** - Wrap the API as MCP tools so an AI assistant can conversationally search listings, estimate prices, and check regulatory eligibility on behalf of a user.

- **Investment simulator** - A calculator for the Jeanbrun device that shows projected amortization, rental yield, and tax impact over the 9-year commitment period based on a specific property and the user's tax bracket.

- **Neighbourhood insights** - Enrich each arrondissement with data such as average rental yield, metro station density, school ratings, and crime statistics.

### Phase 3 - Frontend

We would like to improve our front-end, either by continuing to use Streamlit while adding new features, or by switching to a JavaScript-based solution (e.g., Vibecode) and providing the full prompt used.

Addtional features would be:

- **User Registration** authentification and possibillity to save previous searches from the user.

- **PTZ eligibility and additional law oriented features** such as Financing help presentation in the UI for potential first time buyer, or presentation of fiscal amortization plan in case it can finance the loan to buy (Jeanbrun device).

- **DPE rental restrictions** - G-rated properties can't be rented out since 2025, F-rated banned from 2028, E-rated from 2034. The 2026 reform also changed the electricity conversion coefficient, reclassifying ~850,000 properties. Critical info for buy-to-let buyers.

- **Adding fake visuals for each properties**

- **LLM as an assistant** - Add a LLM/Chatbot in the UI to help the customer navigates through the app and ask how to use it or opinion on some properties.


All these ideas might not be implemented in the end but are what we think the project should look like.
