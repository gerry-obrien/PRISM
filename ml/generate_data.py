"""
generate_data.py — Synthetic Paris Housing Market Data Generator
================================================================
Produces two CSV files for a property-price prediction pipeline:
  • data/training_data.csv  (100 000 rows — historical sales 2015-2024)
  • data/listings.csv        (1 000 rows  — current properties for sale)

Run:  python generate_data.py

Author's note for students
--------------------------
Every section below is commented to explain *what* it does and *why*.
The goal is to create data that behaves like the real Paris property market:
prices are mostly explainable by location, size, condition, etc., but there
is always a large chunk of unexplained variance (negotiation skill, exact
view from windows, seller urgency…). That unexplained part is injected via
a log-normal noise multiplier so that a regression model tops out around
R² ≈ 0.65–0.75 — a realistic ceiling for housing data.
"""

import os
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 0. REPRODUCIBILITY
# ---------------------------------------------------------------------------
# Setting a fixed seed means running this script twice produces identical CSVs.
np.random.seed(42)


# ===========================================================================
# 1. REFERENCE TABLES — Paris arrondissement knowledge
# ===========================================================================
# These encode *qualitative* knowledge about Paris real estate as smooth
# numeric relationships.  They are NOT hard-coded price tables — they feed
# into a formula with plenty of internal variance and noise on top.

# --- 1a. Price-per-m² baseline by arrondissement (approximate €/m² in 2021)
# Source intuition: publicly available Notaires de Paris price maps.
# These are rough centres of each arrondissement's distribution; every
# individual property will deviate substantially.
PRICE_PER_SQM_BASELINE = {
    1: 12_800, 2: 11_500, 3: 11_800, 4: 13_000,
    5: 11_200, 6: 14_000, 7: 13_500, 8: 11_800,
    9: 10_500, 10: 9_800,  11: 10_000, 12: 9_200,
    13: 8_800,  14: 9_600,  15: 9_800,  16: 11_000,
    17: 9_900,  18: 8_500,  19: 7_600,  20: 7_800,
}

# --- 1b. Approximate geographic centres of each arrondissement
# Used only for scattering lat/lon on a map — not fed to the model.
ARROND_CENTRES = {
    1:  (48.8606, 2.3376),  2:  (48.8682, 2.3414),
    3:  (48.8637, 2.3615),  4:  (48.8540, 2.3565),
    5:  (48.8462, 2.3502),  6:  (48.8495, 2.3323),
    7:  (48.8566, 2.3150),  8:  (48.8744, 2.3106),
    9:  (48.8769, 2.3372),  10: (48.8762, 2.3599),
    11: (48.8599, 2.3785),  12: (48.8396, 2.3876),
    13: (48.8322, 2.3561),  14: (48.8331, 2.3264),
    15: (48.8421, 2.2990),  16: (48.8637, 2.2769),
    17: (48.8875, 2.3089),  18: (48.8925, 2.3444),
    19: (48.8864, 2.3824),  20: (48.8638, 2.3985),
}

# --- 1c. Listing stock distribution for the 1 000 active listings
# Outer/larger arrondissements carry more active stock.
LISTING_WEIGHTS = {
    1: 15, 2: 18, 3: 20, 4: 18, 5: 35, 6: 25,
    7: 30, 8: 30, 9: 45, 10: 55, 11: 65, 12: 55,
    13: 70, 14: 55, 15: 95, 16: 65, 17: 65, 18: 75,
    19: 80, 20: 80,
}

# --- 1d. Realistic Parisian street name pool (≈200 names)
# Mixing real and plausible-sounding names so addresses look authentic
# without being traceable to specific real buildings.
STREET_PREFIXES = [
    "Rue", "Rue", "Rue", "Rue",        # Rue is by far the most common
    "Avenue", "Avenue",
    "Boulevard", "Boulevard",
    "Impasse", "Passage", "Place", "Allée",
]

STREET_NAMES = [
    # Classic & historical
    "de Rivoli", "du Faubourg Saint-Honoré", "de la Paix", "Saint-Honoré",
    "de Rennes", "de Sèvres", "du Bac", "de Grenelle", "de Vaugirard",
    "Saint-Jacques", "Mouffetard", "des Écoles", "Monge", "Lacépède",
    "Oberkampf", "de la Roquette", "de Charonne", "Saint-Maur",
    "de Belleville", "des Pyrénées", "de Ménilmontant", "de Bagnolet",
    "de Tolbiac", "Nationale", "de Patay", "Bobillot",
    "de la Convention", "Lecourbe", "du Commerce", "Cambronne",
    "Daguerre", "d'Alésia", "Didot", "Raymond Losserand",
    "Lepic", "des Abbesses", "Caulaincourt", "Lamarck",
    "de Crimée", "de Flandre", "Manin", "Laumière",
    "de Clichy", "des Batignolles", "Cardinet", "de Tocqueville",
    "de Passy", "de la Pompe", "de Longchamp", "Raynouard",
    "de la Bastille", "de Lyon", "de Bercy", "Traversière",
    "Réaumur", "de Turbigo", "du Temple", "Volta",
    "des Francs-Bourgeois", "de Turenne", "du Roi de Sicile",
    "des Archives", "Rambuteau", "Beaubourg",
    # Grands boulevards & avenues
    "des Champs-Élysées", "Montaigne", "George V", "Marceau",
    "Haussmann", "de l'Opéra", "des Italiens", "Poissonnière",
    "de Magenta", "du Faubourg Saint-Denis", "du Château d'Eau",
    "de Strasbourg", "Saint-Martin", "de la République",
    "Voltaire", "de la Nation", "du Trône",
    "Daumesnil", "de Reuilly", "Michel Bizot", "de Picpus",
    "de Choisy", "d'Ivry", "d'Italie", "des Gobelins",
    "du Maine", "du Montparnasse", "Edgar Quinet", "Delambre",
    "Émile Zola", "de Lourmel", "Saint-Charles", "Beaugrenelle",
    "Kléber", "Victor Hugo", "Foch", "Henri Martin",
    "de Villiers", "de Lévis", "Nollet", "des Dames",
    "Ordener", "Marcadet", "Championnet", "du Mont-Cenis",
    "de Meaux", "Armand Carrel", "de Joinville", "Jaurès",
    "des Amandiers", "de la Mare", "Sorbier", "des Envierges",
    # Extra variety
    "Étienne Marcel", "Montorgueil", "du Louvre", "de Valois",
    "du Quatre-Septembre", "Vivienne", "des Petits-Champs",
    "de Bretagne", "du Vertbois", "Pastourelle", "de Picardie",
    "des Rosiers", "du Pont Louis-Philippe", "Saint-Paul",
    "Descartes", "Thouin", "Clovis", "du Cardinal Lemoine",
    "de Médicis", "Guynemer", "d'Assas", "de Fleurus",
    "Saint-Dominique", "de l'Université", "de Varenne", "de Bourgogne",
    "de Miromesnil", "du Faubourg Saint-Honoré", "de Penthièvre",
    "de la Chaussée d'Antin", "de Maubeuge", "de la Tour d'Auvergne",
    "du Faubourg Poissonnière", "d'Hauteville", "de Paradis",
    "Bichat", "de Lancry", "de Marseille", "des Vinaigriers",
    "de la Fontaine au Roi", "Jean-Pierre Timbaud", "du Chemin Vert",
    "Erard", "de Prague", "Crozatier", "de Cotte",
    "du Moulin des Prés", "Vergniaud", "de la Butte aux Cailles",
    "Pernety", "du Château", "des Plantes", "Brézin",
    "de Javel", "Fondary", "Violet", "de la Croix Nivert",
    "Chardon Lagache", "d'Auteuil", "Michel-Ange", "Molitor",
    "de la Jonquière", "Truffaut", "Brochant", "Legendre",
    "Ramey", "Custine", "Clignancourt", "des Poissonniers",
    "Petit", "de l'Ourcq", "Riquet", "Curial",
    "de Fontarabie", "des Orteaux", "des Maraîchers", "Pelleport",
]


# ===========================================================================
# 2. HELPER FUNCTIONS
# ===========================================================================

def generate_arrondissements(n, weights=None):
    """
    Sample arrondissements 1–20.

    For training data we use a roughly uniform distribution so the model
    sees enough examples from every arrondissement.
    For listings we use `weights` that mirror real active-stock patterns
    (more listings in outer arrondissements).
    """
    arronds = np.arange(1, 21)
    if weights is not None:
        probs = np.array([weights[a] for a in arronds], dtype=float)
        probs /= probs.sum()
    else:
        # Slightly weighted toward outer arrondissements even in training
        # to loosely reflect reality, but not as extreme as listings.
        probs = np.array([3, 3, 3, 3, 4, 3, 4, 4, 5, 5,
                          6, 5, 6, 5, 8, 6, 6, 7, 7, 7], dtype=float)
        probs /= probs.sum()
    return np.random.choice(arronds, size=n, p=probs)


def generate_year(n, is_listing=False):
    """
    Sample the year of sale / listing.

    Training data spans 2015–2024 with a roughly uniform distribution
    (slightly more volume in 2019–2022, the boom years).
    Listings are all 2024 — they represent current stock.
    """
    if is_listing:
        return np.full(n, 2024, dtype=int)
    # Weights: a gentle bump during peak years
    year_weights = {
        2015: 8, 2016: 9, 2017: 9, 2018: 10, 2019: 11,
        2020: 9, 2021: 12, 2022: 11, 2023: 10, 2024: 11,
    }
    years = np.array(list(year_weights.keys()))
    probs = np.array(list(year_weights.values()), dtype=float)
    probs /= probs.sum()
    return np.random.choice(years, size=n, p=probs)


def year_price_index(years):
    """
    Temporal price index: captures the 2015→2021 rise and 2022→2024 softening.

    Returns a multiplier centred near 1.0 for 2021 (the peak year).
    The curve is smooth — modelled as a polynomial fit to approximate
    the real Paris price trajectory.

    Approximate real trajectory (index, 2021 = 1.00):
      2015: 0.82   2016: 0.85   2017: 0.88   2018: 0.92   2019: 0.96
      2020: 0.97   2021: 1.00   2022: 0.99   2023: 0.96   2024: 0.94
    """
    index_map = {
        2015: 0.82, 2016: 0.85, 2017: 0.88, 2018: 0.92, 2019: 0.96,
        2020: 0.97, 2021: 1.00, 2022: 0.99, 2023: 0.96, 2024: 0.94,
    }
    return np.array([index_map[y] for y in years])


def generate_property_type(n, arrondissements):
    """
    Assign property type: Apartment or House.

    Houses are rare in Paris (≈5 % overall) and almost nonexistent in the
    central arrondissements. They are more common in the 16th, 19th, 20th
    and other outer areas where small houses / pavillons still exist.
    """
    # Probability of 'House' by arrondissement
    house_prob = {
        1: 0.005, 2: 0.005, 3: 0.008, 4: 0.008, 5: 0.01,
        6: 0.01,  7: 0.015, 8: 0.01,  9: 0.01,  10: 0.01,
        11: 0.02, 12: 0.04, 13: 0.04, 14: 0.03, 15: 0.03,
        16: 0.08, 17: 0.03, 18: 0.04, 19: 0.06, 20: 0.06,
    }
    probs = np.array([house_prob[a] for a in arrondissements])
    is_house = np.random.random(n) < probs
    return np.where(is_house, "House", "Apartment")


def generate_area(n, property_types):
    """
    Generate built area in square metres.

    Apartments: log-normal distribution centred around 50 m², with a long
    right tail (some large Haussmannian apartments reach 200+ m²).
    Houses: generally larger, centred around 100 m².
    """
    areas = np.zeros(n)
    apt_mask = property_types == "Apartment"
    house_mask = ~apt_mask

    # Apartments: log-normal with median ≈ 48 m²
    n_apt = apt_mask.sum()
    areas[apt_mask] = np.random.lognormal(mean=np.log(48), sigma=0.50, size=n_apt)
    areas[apt_mask] = np.clip(areas[apt_mask], 9, 350)  # studio min 9 m²

    # Houses: log-normal with median ≈ 95 m²
    n_house = house_mask.sum()
    if n_house > 0:
        areas[house_mask] = np.random.lognormal(mean=np.log(95), sigma=0.40, size=n_house)
        areas[house_mask] = np.clip(areas[house_mask], 30, 500)

    return np.round(areas, 1)


def generate_num_rooms(areas):
    """
    Derive number of rooms from area.

    In France, 'pièces principales' roughly follows:
      < 20 m²  → 1 room (studio)
      20–35 m² → 1–2 rooms
      35–55 m² → 2–3 rooms
      55–80 m² → 3–4 rooms
      80–120 m²→ 4–5 rooms
      > 120 m² → 5–7 rooms

    We add randomness: the base is area / 22 (roughly one room per 22 m²)
    with Gaussian noise, clamped to [1, 9].
    """
    base_rooms = areas / 22.0
    noise = np.random.normal(0, 0.4, size=len(areas))
    rooms = np.round(base_rooms + noise).astype(int)
    return np.clip(rooms, 1, 9)


def generate_is_new_build(n, years):
    """
    New builds are rare in Paris (≈3 % of transactions) and almost all
    appear from 2019 onwards — major new-build programmes (Batignolles,
    Bercy-Charenton, etc.) delivered mainly in that period.
    """
    probs = np.where(years >= 2019, 0.04, 0.005)
    return np.random.random(n) < probs


def generate_building_condition(n, is_new_build):
    """
    New builds are always in Good condition.
    Existing stock: roughly 55 % Good, 30 % Average, 15 % Poor.
    """
    conditions = np.full(n, "Good", dtype=object)
    existing = ~is_new_build
    n_existing = existing.sum()
    cond_draw = np.random.random(n_existing)
    conditions[existing] = np.where(
        cond_draw < 0.55, "Good",
        np.where(cond_draw < 0.85, "Average", "Poor")
    )
    return conditions


def generate_dpe_rating(n, is_new_build, building_condition):
    """
    Energy Performance Certificate (DPE) rating A–G.

    New builds: almost always A or B (current regulations require it).
    Existing stock: the distribution depends on condition.
      Good condition  → skewed toward C/D
      Average         → skewed toward D/E
      Poor            → skewed toward E/F/G (passoires thermiques)
    """
    ratings = np.empty(n, dtype=object)

    # --- New builds: A (60 %) or B (35 %) or C (5 %)
    new_mask = is_new_build
    n_new = new_mask.sum()
    if n_new > 0:
        r = np.random.random(n_new)
        ratings[new_mask] = np.where(r < 0.60, "A",
                            np.where(r < 0.95, "B", "C"))

    # --- Existing stock
    existing = ~is_new_build
    for cond, probs in [
        ("Good",    [0.03, 0.08, 0.25, 0.35, 0.18, 0.08, 0.03]),
        ("Average", [0.01, 0.03, 0.12, 0.28, 0.30, 0.18, 0.08]),
        ("Poor",    [0.00, 0.01, 0.05, 0.15, 0.25, 0.30, 0.24]),
    ]:
        mask = existing & (building_condition == cond)
        n_mask = mask.sum()
        if n_mask > 0:
            ratings[mask] = np.random.choice(
                ["A", "B", "C", "D", "E", "F", "G"],
                size=n_mask, p=probs,
            )
    return ratings


def generate_floor_and_elevator(n, property_types, is_new_build):
    """
    Floor number and elevator presence.

    Houses are always floor 0 with no elevator.
    Apartments:
      - Parisian buildings are typically 5–7 storeys (some up to 8–9).
      - Distribution: ground floor ≈ 12 %, floors 1–3 most common, 4–6
        progressively rarer, 7+ very rare.
      - Elevator: new builds almost always have one (95 %).
        Older buildings: ≈60 % have an elevator.
      - Without elevator, we cap the floor at 6 (nobody walks up 8 flights).
      - Ground-floor apartments are less likely to have an elevator entry
        (some ground-floor units are in small buildings without one).
    """
    floors = np.zeros(n, dtype=int)
    elevators = np.zeros(n, dtype=bool)

    house_mask = property_types == "House"
    apt_mask = ~house_mask
    n_apt = apt_mask.sum()

    if n_apt > 0:
        # Floor distribution for apartments
        floor_probs = [0.12, 0.18, 0.20, 0.18, 0.14, 0.10, 0.05, 0.02, 0.01]
        floor_choices = np.arange(len(floor_probs))
        apt_floors = np.random.choice(floor_choices, size=n_apt, p=floor_probs)
        floors[apt_mask] = apt_floors

        # Elevator
        apt_new = is_new_build[apt_mask]
        elev_prob = np.where(apt_new, 0.95, 0.60)
        # Ground-floor apartments in older buildings are less likely to
        # be in a building with elevator
        ground_and_old = (apt_floors == 0) & (~apt_new)
        elev_prob[ground_and_old] = 0.40
        apt_elev = np.random.random(n_apt) < elev_prob
        elevators[apt_mask] = apt_elev

        # Constraint: no elevator → cap floor at 6
        no_elev_high = apt_mask.copy()
        no_elev_high[apt_mask] = (~apt_elev) & (apt_floors > 6)
        floors[no_elev_high] = np.random.randint(3, 7, size=no_elev_high.sum())

    return floors, elevators


def generate_address(n, arrondissements):
    """
    Build synthetic but realistic-looking French street addresses.

    Format: "{number} {prefix} {name}, {postcode} Paris"
    The postcode encodes the arrondissement: 75001 … 75020.
    We draw from a large pool of street prefixes and names so that
    repetition is minimal across 100 000+ rows.
    """
    numbers = np.random.randint(1, 151, size=n)
    prefix_idx = np.random.randint(0, len(STREET_PREFIXES), size=n)
    name_idx = np.random.randint(0, len(STREET_NAMES), size=n)
    postcodes = 75000 + arrondissements

    addresses = [
        f"{numbers[i]} {STREET_PREFIXES[prefix_idx[i]]} {STREET_NAMES[name_idx[i]]}, {postcodes[i]} Paris"
        for i in range(n)
    ]
    return addresses


def generate_lat_lon(arrondissements):
    """
    Scatter GPS coordinates around each arrondissement's centre.

    A Gaussian std of ≈0.008° (roughly 600–900 m) keeps points within
    the arrondissement while adding realistic spread.  These are for
    map display only — they must NOT influence price.
    """
    lats = np.zeros(len(arrondissements))
    lons = np.zeros(len(arrondissements))
    for i, a in enumerate(arrondissements):
        clat, clon = ARROND_CENTRES[a]
        lats[i] = clat + np.random.normal(0, 0.008)
        lons[i] = clon + np.random.normal(0, 0.008)
    return np.round(lats, 6), np.round(lons, 6)


# ===========================================================================
# 3. PRICE MODEL — the explainable component + noise
# ===========================================================================

def compute_price(area, arrond, year, floor, has_elevator, dpe,
                  condition, is_new, prop_type):
    """
    Compute realistic sale prices.

    price = explainable_component × noise_multiplier

    The explainable component combines:
      1. Base price/m² for the arrondissement (with its own micro-variance)
      2. Area with a slight super-linear premium for large properties
      3. Year index (temporal trend)
      4. Floor premium / penalty (modulated by elevator)
      5. DPE adjustment
      6. Condition adjustment
      7. New-build premium
      8. House premium (houses with gardens are rare and prized)

    The noise multiplier (log-normal, σ=0.20) injects ≈±20 % unexplained
    variance — the negotiation / view / urgency factors that never appear
    in the data.
    """
    n = len(area)

    # --- 1. Base price per m² with micro-variance per property
    # Even within one arrondissement, streets differ a lot.
    base_psm = np.array([PRICE_PER_SQM_BASELINE[a] for a in arrond], dtype=float)
    micro_var = np.random.normal(1.0, 0.10, size=n)  # ±10 % street-level variance
    base_psm *= micro_var

    # --- 2. Area with super-linear premium
    # In Paris, large apartments are disproportionately expensive because
    # they are rare. We model this as area^1.05 instead of area^1.0.
    effective_area = area ** 1.05

    # --- 3. Temporal trend
    yr_idx = year_price_index(year)

    # --- 4. Floor effect (for apartments only)
    # Ground floor: -8 % penalty (noise, security, light)
    # Floors 1-2: neutral
    # Floors 3-5: +2 % per floor above 2
    # Floors 6+: +3 % per floor above 5 (rare high floors, views)
    # BUT without elevator, high floors LOSE their premium and floors 4+
    # actually get a penalty (nobody wants to climb).
    floor_mult = np.ones(n)
    is_apt = prop_type == "Apartment"

    # Ground floor penalty
    ground = is_apt & (floor == 0)
    floor_mult[ground] = 0.92

    # Mid-floor premium (floors 3-5)
    for f in range(3, 6):
        mask = is_apt & (floor == f)
        floor_mult[mask] = 1.0 + 0.02 * (f - 2)

    # High floor premium (6+)
    high = is_apt & (floor >= 6)
    floor_mult[high] = 1.0 + 0.06 + 0.03 * (floor[high] - 5)

    # No elevator: floors 4+ become a penalty instead of a premium
    no_elev_high = is_apt & (~has_elevator) & (floor >= 4)
    floor_mult[no_elev_high] = 1.0 - 0.02 * (floor[no_elev_high] - 3)

    # No elevator floors 1-3: slight discount vs buildings with elevator
    no_elev_low = is_apt & (~has_elevator) & (floor >= 1) & (floor <= 3)
    floor_mult[no_elev_low] *= 0.97

    # --- 5. DPE adjustment
    # A and B: premium (energy efficient, modern insulation)
    # D: neutral reference
    # F and G: heavy discount (passoires thermiques — hard to sell/rent)
    dpe_map = {"A": 1.08, "B": 1.04, "C": 1.01, "D": 1.00,
               "E": 0.97, "F": 0.92, "G": 0.87}
    dpe_mult = np.array([dpe_map[d] for d in dpe])

    # --- 6. Building condition
    cond_map = {"Good": 1.00, "Average": 0.92, "Poor": 0.80}
    cond_mult = np.array([cond_map[c] for c in condition])

    # --- 7. New-build premium (≈15–20 % in Paris)
    new_mult = np.where(is_new, np.random.uniform(1.14, 1.22, size=n), 1.0)

    # --- 8. House premium (scarce in Paris → prized)
    house_mult = np.where(prop_type == "House",
                          np.random.uniform(1.08, 1.18, size=n), 1.0)

    # --- Combine explainable component
    explainable = (base_psm * effective_area * yr_idx * floor_mult *
                   dpe_mult * cond_mult * new_mult * house_mult)

    # --- 9. Inject unexplained noise (the core ML-challenge ingredient)
    # Log-normal with underlying σ = 0.20 → multiplicative noise with
    # mean ≈ 1.0 and roughly ±20 % spread.
    noise = np.random.lognormal(mean=-0.02, sigma=0.20, size=n)
    # (mean=-0.02 offsets the log-normal's inherent right skew so that
    #  the expected value of the multiplier is ≈ 1.0)

    price = explainable * noise

    # Sanity floor: no property below 30 000 €
    price = np.maximum(price, 30_000)

    return np.round(price, 2)


# ===========================================================================
# 4. MAIN GENERATION PIPELINE
# ===========================================================================

def generate_dataset(n, is_listing=False):
    """
    Orchestrate all generators to produce one complete DataFrame.

    Parameters
    ----------
    n : int           — number of rows
    is_listing : bool — if True, adds listing_id and estimated_price_eur
    """
    print(f"  Generating {'listings' if is_listing else 'training'} data ({n:,} rows)…")

    # --- Core columns
    arrond = generate_arrondissements(
        n, weights=LISTING_WEIGHTS if is_listing else None
    )
    year = generate_year(n, is_listing=is_listing)
    prop_type = generate_property_type(n, arrond)
    area = generate_area(n, prop_type)
    num_rooms = generate_num_rooms(area)
    is_new = generate_is_new_build(n, year)
    condition = generate_building_condition(n, is_new)
    dpe = generate_dpe_rating(n, is_new, condition)
    floor, elevator = generate_floor_and_elevator(n, prop_type, is_new)
    address = generate_address(n, arrond)
    lat, lon = generate_lat_lon(arrond)

    # --- Price
    price = compute_price(
        area, arrond, year, floor, elevator, dpe,
        condition, is_new, prop_type,
    )

    # --- Assemble DataFrame
    df = pd.DataFrame({
        "price_eur": price,
        "area_sqm": area,
        "num_rooms": num_rooms,
        "arrondissement": arrond,
        "property_type": prop_type,
        "year": year,
        "floor": floor,
        "has_elevator": elevator,
        "dpe_rating": dpe,
        "building_condition": condition,
        "is_new_build": is_new,
        "address": address,
        "latitude": lat,
        "longitude": lon,
    })

    # --- Extra columns for listings
    if is_listing:
        df.insert(0, "listing_id", [f"LST-{i+1:05d}" for i in range(n)])
        df["estimated_price_eur"] = np.nan

    return df


# ===========================================================================
# 5. VALIDATION SUMMARY
# ===========================================================================

def print_summary(df_train, df_list):
    """
    Print a diagnostic summary so we can eyeball that the data looks right.
    """
    sep = "=" * 72
    print(f"\n{sep}")
    print("VALIDATION SUMMARY")
    print(sep)

    for label, df in [("TRAINING DATA", df_train), ("LISTINGS", df_list)]:
        print(f"\n--- {label} ({len(df):,} rows) ---")

        # Price per m²
        df_tmp = df.copy()
        df_tmp["price_per_sqm"] = df_tmp["price_eur"] / df_tmp["area_sqm"]

        agg = df_tmp.groupby("arrondissement").agg(
            mean_price=("price_eur", "mean"),
            mean_psm=("price_per_sqm", "mean"),
            count=("price_eur", "count"),
        ).round(0)
        print("\n  Arrond  |    Count |  Mean Price (€) | Mean €/m²")
        print("  --------+----------+-----------------+----------")
        for a, row in agg.iterrows():
            print(f"    {a:2d}    | {row['count']:>8.0f} | {row['mean_price']:>15,.0f} | {row['mean_psm']:>8,.0f}")

        # DPE distribution
        dpe_pct = df["dpe_rating"].value_counts(normalize=True).sort_index() * 100
        print("\n  DPE distribution:")
        for rating, pct in dpe_pct.items():
            print(f"    {rating}: {pct:5.1f}%")

        # Property type distribution
        type_pct = df["property_type"].value_counts(normalize=True) * 100
        print("\n  Property type distribution:")
        for ptype, pct in type_pct.items():
            print(f"    {ptype}: {pct:5.1f}%")

    print(f"\n{sep}")
    print("Generation complete.")
    print(sep)


# ===========================================================================
# 6. ENTRY POINT
# ===========================================================================

if __name__ == "__main__":
    print("Paris Housing Market — Synthetic Data Generator")
    print("=" * 50)

    # Resolve data directory relative to project root
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "data")
    os.makedirs(DATA_DIR, exist_ok=True)

    # Generate both datasets
    df_training = generate_dataset(100_000, is_listing=False)
    df_listings = generate_dataset(1_000, is_listing=True)

    # Save to CSV
    df_training.to_csv(os.path.join(DATA_DIR, "training_data.csv"), index=False)
    df_listings.to_csv(os.path.join(DATA_DIR, "listings.csv"), index=False)
    print(f"\n  Saved: {DATA_DIR}/training_data.csv")
    print(f"  Saved: {DATA_DIR}/listings.csv")

    # Print validation summary
    print_summary(df_training, df_listings)
