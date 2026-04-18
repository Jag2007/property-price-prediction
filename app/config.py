import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "processed" / "real_estate_clean.csv"
REG_MODEL_PATH = BASE_DIR / "models" / "xgb_regression_model.joblib"
REG_SCALER_PATH = BASE_DIR / "models" / "regression_scaler.joblib"
CLF_MODEL_PATH = BASE_DIR / "models" / "xgb_classification_model.joblib"
CLF_SCALER_PATH = BASE_DIR / "models" / "classification_scaler.joblib"
HOSTED_APP_URL = "https://property-price-prediction-real-estate.streamlit.app/"

FURNISH_MAP = {"Unfurnished": 0, "Semi-furnished": 1, "Fully-furnished": 2}
NEIGHBORHOODS = ["Downtown", "IT Hub", "Industrial", "Residential", "Suburban"]
INT_COLUMNS = {"Bedrooms", "Bathrooms", "Age_of_Property", "Floor_Number"}
GRADE_LABELS = {0: "0 - Avoid", 1: "1 - Hold", 2: "2 - Buy"}
ADVISORY_LABELS = {0: "Avoid", 1: "Hold", 2: "Buy"}
ADVISORY_DESCRIPTIONS = {
    0: "The model sees this as the weakest investment class. Treat it as high risk unless other real-world checks strongly support it.",
    1: "The model sees this as a moderate investment class. Review the details before deciding because the result is not clearly negative or positive.",
    2: "The model sees this as the strongest investment class. It looks favorable in the training-data pattern, but still needs real-world verification.",
}
ADVISORY_COLORS = {
    0: {"text": "#9F1D1D", "background": "#FFF0F0", "border": "#D92D20"},
    1: {"text": "#8A5B00", "background": "#FFF7E6", "border": "#D99A00"},
    2: {"text": "#176B3A", "background": "#ECFDF3", "border": "#12B76A"},
}

FEATURE_COLUMNS_FALLBACK = [
    "Total_Square_Footage",
    "Bedrooms",
    "Bathrooms",
    "Age_of_Property",
    "Floor_Number",
    "Furnishing_Status",
    "Distance_to_City_Center_km",
    "Proximity_to_Public_Transport_km",
    "Crime_Index",
    "Air_Quality_Index",
    "Neighborhood_Growth_Rate_%",
    "Price_per_SqFt",
    "Annual_Property_Tax",
    "Estimated_Rental_Yield_%",
    "Neighborhood_IT Hub",
    "Neighborhood_Industrial",
    "Neighborhood_Residential",
    "Neighborhood_Suburban",
]

RAW_NUMERIC_COLUMNS = [
    "Total_Square_Footage",
    "Bedrooms",
    "Bathrooms",
    "Age_of_Property",
    "Floor_Number",
    "Distance_to_City_Center_km",
    "Proximity_to_Public_Transport_km",
    "Crime_Index",
    "Air_Quality_Index",
    "Neighborhood_Growth_Rate_%",
    "Price_per_SqFt",
    "Annual_Property_Tax",
    "Estimated_Rental_Yield_%",
]

LABELS = {
    "Total_Square_Footage": "Total Square Footage",
    "Bedrooms": "Bedrooms",
    "Bathrooms": "Bathrooms",
    "Age_of_Property": "Age of Property (years)",
    "Floor_Number": "Floor Number",
    "Distance_to_City_Center_km": "Distance to City Center (km)",
    "Proximity_to_Public_Transport_km": "Proximity to Public Transport (km)",
    "Crime_Index": "Crime Index",
    "Air_Quality_Index": "Air Quality Index",
    "Neighborhood_Growth_Rate_%": "Neighbourhood Growth Rate (%)",
    "Price_per_SqFt": "Price per Sq Ft",
    "Annual_Property_Tax": "Annual Property Tax",
    "Estimated_Rental_Yield_%": "Estimated Rental Yield (%)",
}

PROMPT_EXAMPLE = (
    "Example: 1450 sqft 3 BHK 2 bathrooms, 6 years old, 8th floor, "
    "semi furnished, IT Hub, 4.5 km from city center, 0.7 km from metro, "
    "crime index 32, AQI 88, growth 9%, price per sqft 7200, "
    "annual tax 85000, rental yield 4.2%."
)

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.3-70b-versatile"


# Load local .env values so local demos can use Groq/email without Streamlit secrets.
def load_local_env():
    env_path = BASE_DIR / ".env"
    if not env_path.exists():
        return

    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


# Ignore empty placeholder values copied from setup documentation.
def is_real_config_value(value):
    if not value:
        return False
    value = str(value).strip()
    lowered = value.lower()
    return not (lowered.startswith("replace-with") or lowered.startswith("your-"))


# Read one setting from .env, normal environment variables, or Streamlit secrets.
def config_value(secrets, key):
    load_local_env()
    value = os.getenv(key)
    if is_real_config_value(value):
        return value

    try:
        value = secrets.get(key)
    except Exception:
        value = None
    return value if is_real_config_value(value) else None
