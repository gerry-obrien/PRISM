# PRISM - Paris Real estate Intelligent Smart Market

An app that helps people buying property in Paris figure out if they're getting a good deal. A machine learning model estimates fair prices for 1,000 properties currently on the market, then flags which ones are overpriced and which are bargains.

---

## How to run it

You need Python 3.10 or higher.

### 1. Clone and set up

```bash
git clone https://github.com/YOUR_USERNAME/prism.git
cd prism
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
.venv\Scripts\activate           # Windows
```

### 2. Generate the data

```bash
pip install -r ml/requirements.txt
cd ml
python generate_data.py
```

This creates two files in `data/`:
- `training_data.csv` - 100,000 past Paris property sales (2015–2024)
- `listings.csv` - 1,000 properties currently for sale, with no price estimate yet

The script uses a fixed seed so the output is always identical.

### 3. Train the model

```bash
python train.py
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
cd ../backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
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

## What's next

This is Phase 1. Here's what's coming and some ideas we're exploring.

### Phase 2 - Frontend

A React app with a Leaflet map showing all 1,000 listings as colour-coded pins (green for undervalued, red for overvalued, yellow for fair). Clicking a pin opens a listing card with the DPE gauge, asking price vs estimated price, and property details. Filters for arrondissement, price range, number of rooms, DPE rating, and valuation status.

### Phase 3 - User accounts and buyer quiz

User registration and login (SQLite). After signing in, users complete a multi-step quiz about their buyer profile - are they a first-time buyer, what's their household income, are they buying to live in or to rent out, how much do they care about energy efficiency. Based on the answers, the app tells them:

- **PTZ eligibility** - Paris is zone A bis. The prêt à taux zéro is for first-time buyers below certain income thresholds, and can finance up to 50% of the purchase for the most modest households.
- **Jeanbrun device** - Uses fiscal amortisation instead of tax credits, no geographic zoning, but requires a 9-year rental commitment with capped rents. Relevant for anyone buying as an investment.
- **DPE rental restrictions** - G-rated properties can't be rented out since 2025, F-rated banned from 2028, E-rated from 2034. The 2026 reform also changed the electricity conversion coefficient, reclassifying ~850,000 properties. Critical info for buy-to-let buyers.

Users can also save favourite listings and come back to them later.

### Phase 4 - Containerisation and deployment

Dockerfiles for each service, a `docker-compose.yml` to run everything with one command, and Kubernetes manifests for production deployment. Includes a CronJob for periodic model retraining.

### Other potential ideas we're considering

- **MCP server** - Wrapping the API as MCP tools so an AI assistant could conversationally search listings, estimate prices, and check regulatory eligibility on behalf of a user.
- **Model comparison dashboard** - Training multiple models (Linear Regression as a baseline, XGBoost as a contender) and letting users see how they compare, which would demonstrate the diminishing returns of model complexity when there's a hard noise ceiling.
- **Investment simulator** - A calculator for the Jeanbrun device that shows projected amortisation, rental yield, and tax impact over the 9-year commitment period based on a specific property and the user's tax bracket.
- **Neighbourhood insights** - Enriching each arrondissement with data like average rental yield, metro station density, school ratings, and crime stats.
