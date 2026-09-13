import re
import requests
from datetime import datetime

OPENMETEO_URL = "https://api.open-meteo.com/v1/forecast"
CHEMB_CAPACITY_MCM = 3645.0

def fetch_chembarambakkam_level():
    try:
        resp = requests.get(
            "https://numerical.co.in/numerons/collection/5e127ba3545c9d1c18f23221",
            timeout=8,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        resp.raise_for_status()
        text = resp.text
        match = re.search(
            r"Chembarambakkam Lake.*?At\s+([\d.]+)%\s+of its full capacity",
            text,
            re.DOTALL,
        )
        if match:
            pct = float(match.group(1))
            mcm = round((pct / 100) * CHEMB_CAPACITY_MCM, 1)
            return {
                "pct": round(pct, 2),
                "mcm": mcm,
                "source": "numerical.co.in / CMWSSB",
                "fetched_at": datetime.now().strftime("%d %b %Y %H:%M"),
            }
        return None
    except Exception:
        return None

def fetch_rainfall_for(lat, lon):
    try:
        params = {
            "latitude": lat,
            "longitude": lon,
            "daily": "precipitation_sum",
            "timezone": "Asia/Kolkata",
            "past_days": 7,
            "forecast_days": 1,
        }
        resp = requests.get(OPENMETEO_URL, params=params, timeout=8)
        resp.raise_for_status()
        data = resp.json()
        dates = data["daily"]["time"]
        daily_mm = [v if v is not None else 0.0 for v in data["daily"]["precipitation_sum"]]
        return {
            "rain_today": round(daily_mm[-1], 1),
            "rain_3": round(sum(daily_mm[-3:]), 1),
            "rain_5": round(sum(daily_mm[-5:]), 1),
            "rain_7": round(sum(daily_mm[-7:]), 1),
            "dates": dates[-7:],
            "daily_mm": daily_mm[-7:],
            "fetched_at": datetime.now().strftime("%d %b %Y %H:%M"),
        }
    except Exception:
        return None