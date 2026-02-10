# model/schema.py

SCHEMA = {
    "Year": {"type": int, "min": 2000, "max": 2030},
    "Prevalence Rate (%)": {"type": float, "min": 0, "max": 100},
    "Incidence Rate (%)": {"type": float, "min": 0, "max": 100},
    "Mortality Rate (%)": {"type": float, "min": 0, "max": 100},
    "Population Affected": {"type": int, "min": 0},
    "Healthcare Access (%)": {"type": float, "min": 0, "max": 100},
    "Doctors per 1000": {"type": float, "min": 0},
    "Hospital Beds per 1000": {"type": float, "min": 0},
    "Average Treatment Cost (USD)": {"type": float, "min": 0},
    "Recovery Rate (%)": {"type": float, "min": 0, "max": 100},
    "DALYs": {"type": float, "min": 0},
    "Improvement in 5 Years (%)": {"type": float, "min": -100, "max": 100},
    "Per Capita Income (USD)": {"type": float, "min": 0},
    "Education Index": {"type": float, "min": 0, "max": 1},
    "Urbanization Rate (%)": {"type": float, "min": 0, "max": 100}
}
