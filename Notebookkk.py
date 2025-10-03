"""
WeatherWise: Hybrid parsing, flexible timeframes, hourly support, and multi-city comparison.

This module is self-contained and designed to be executed directly (import and call).
It keeps the existing app structure intact by providing new, additive capabilities.

Key features:
- Hybrid parsing: Granite (via Ollama) + optional spaCy/transformers NER + regex fallback
- Intent classification: lightweight rule-based with optional sklearn/transformers support
- Flexible timeframes: natural phrases mapped to ISO date/date-range with optional hour_range
- Hourly support: fetch and answer using hourly data when hour_range present
- Multi-city comparison: structured comparison output for multiple locations

External services:
- Weather data is fetched from wttr.in (no API key required)
- Granite model is accessed through Ollama if available (optional)

Usage (examples):
    from Notebookkk import ask_weather
    print(ask_weather("Compare tomorrow's weather in Karachi and Sukkur"))

    print(ask_weather("Between 3-4 pm tomorrow in London, will it rain?"))

Configuration via environment variables:
- OLLAMA_HOST (default: http://localhost:11434)
- GRANITE_MODEL (default: granite3.2)
"""

from __future__ import annotations

import os
import json
import math
import re
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import requests


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
GRANITE_MODEL = os.getenv("GRANITE_MODEL", "granite3.2")


# -----------------------------------------------------------------------------
# Optional NLP dependencies (best-effort import with graceful fallbacks)
# -----------------------------------------------------------------------------
_spacy_nlp = None
try:
    import spacy  # type: ignore

    try:
        # Try to load a small English model if present
        _spacy_nlp = spacy.load("en_core_web_sm")
    except Exception:
        _spacy_nlp = None
except Exception:
    _spacy_nlp = None

_hf_ner = None
try:
    from transformers import pipeline  # type: ignore

    try:
        _hf_ner = pipeline("token-classification", model="dslim/bert-base-NER", grouped_entities=True)
    except Exception:
        _hf_ner = None
except Exception:
    _hf_ner = None

_sklearn_available = False
try:
    # We will set up a tiny on-the-fly rule-based intent model; sklearn optional
    import sklearn  # type: ignore

    _sklearn_available = True
except Exception:
    _sklearn_available = False


# -----------------------------------------------------------------------------
# Granite via Ollama helper
# -----------------------------------------------------------------------------
def ollama_chat(prompt: str, system: Optional[str] = None, model: Optional[str] = None,
                host: Optional[str] = None, json_mode: bool = False, temperature: float = 0.2) -> str:
    model = model or GRANITE_MODEL
    host = host or OLLAMA_HOST
    url = f"{host}/api/chat"
    headers = {"Content-Type": "application/json"}
    messages: List[Dict[str, Any]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    body: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if json_mode:
        body["format"] = "json"

    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=60) as resp:
        raw = json.loads(resp.read().decode("utf-8"))
    if "message" in raw and isinstance(raw["message"], dict):
        return raw["message"].get("content", "")
    return raw.get("response", "")


# -----------------------------------------------------------------------------
# Data structures
# -----------------------------------------------------------------------------
@dataclass
class ParsedQuery:
    locations: List[str]
    attributes: List[str]
    intent: str  # e.g., forecast | comparison | clothing | travel | general
    # Flexible timeframe
    date: Optional[str] = None  # ISO date YYYY-MM-DD
    date_range: Optional[Tuple[str, str]] = None  # (start_iso, end_iso)
    hour_range: Optional[Tuple[int, int]] = None  # (start_hour_0_23, end_hour_0_23)
    original_question: str = ""


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
_CITY_LIKE = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b")
_KNOWN_ATTRS = {"temperature", "precipitation", "humidity", "wind", "pressure", "general"}


def _extract_locations_regex(text: str) -> List[str]:
    # Basic heuristic: title-cased tokens/groups
    cands = _CITY_LIKE.findall(text)
    # Filter out common stop words that are title-cased in sentences
    stop = {"What", "Will", "Show", "How", "In", "On", "At", "From", "To", "And", "Or", "The", "Of"}
    cities = [c for c in cands if c not in stop]
    # Deduplicate while preserving order
    seen = set()
    result: List[str] = []
    for c in cities:
        lc = c.lower()
        if lc not in seen:
            seen.add(lc)
            result.append(lc)
    return result


def _extract_locations_spacy(text: str) -> List[str]:
    if not _spacy_nlp:
        return []
    doc = _spacy_nlp(text)
    cities: List[str] = []
    for ent in doc.ents:
        if ent.label_ in {"GPE", "LOC"}:
            cities.append(ent.text.strip())
    # Dedupe lowercased
    out: List[str] = []
    seen: set = set()
    for c in cities:
        lc = c.lower()
        if lc not in seen:
            seen.add(lc)
            out.append(lc)
    return out


def _extract_locations_hf(text: str) -> List[str]:
    if not _hf_ner:
        return []
    res = _hf_ner(text)
    cities: List[str] = []
    for r in res:
        label = r.get("entity_group", "")
        if label in {"LOC", "PER", "ORG", "MISC"}:  # NER is noisy; keep liberal and filter later
            val = r.get("word", "").strip()
            if val:
                cities.append(val)
    out: List[str] = []
    seen: set = set()
    for c in cities:
        lc = c.lower()
        if lc not in seen:
            seen.add(lc)
            out.append(lc)
    return out


def _extract_attributes(text: str) -> List[str]:
    q = text.lower()
    attrs: List[str] = []
    if any(w in q for w in ["temperature", "temp", "hot", "cold", "warm", "cool"]):
        attrs.append("temperature")
    if any(w in q for w in ["rain", "precip", "precipitation", "snow", "storm", "drizzle"]):
        attrs.append("precipitation")
    if any(w in q for w in ["humidity", "humid", "moisture"]):
        attrs.append("humidity")
    if any(w in q for w in ["wind", "breeze", "gust"]):
        attrs.append("wind")
    if any(w in q for w in ["pressure", "barometric"]):
        attrs.append("pressure")
    if not attrs:
        attrs.append("general")
    return [a for a in attrs if a in _KNOWN_ATTRS]


def _classify_intent(text: str) -> str:
    q = text.lower()
    # Lightweight rule-based intents; extendable to sklearn/transformers if available
    if any(w in q for w in ["compare", "vs", "versus", "comparison"]):
        return "comparison"
    if any(w in q for w in ["pack", "wear", "clothing", "outfit", "dress"]):
        return "clothing"
    if any(w in q for w in ["travel", "trip", "visit", "fly", "drive"]):
        return "travel"
    return "forecast"


_TIME_KEYWORDS = {
    "today": 0,
    "tomorrow": 1,
}


def _parse_timeframe(text: str) -> Tuple[Optional[str], Optional[Tuple[str, str]], Optional[Tuple[int, int]]]:
    """Parse natural time expressions into (date, date_range, hour_range).

    Supports examples like:
    - "next Saturday evening"
    - "from 3–4 pm tomorrow"
    - "in the first week of October"
    Falls back to today if ambiguous.
    """
    q = text.lower()
    now = datetime.now()

    # Hour range like "3-4 pm" or "15:00-16:00"
    hour_range: Optional[Tuple[int, int]] = None
    m = re.search(r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\s*[\-–to]+\s*(\d{1,2})(?::(\d{2}))?\s*(am|pm)?", q)
    if m:
        h1 = int(m.group(1))
        h2 = int(m.group(4))
        ampm1 = m.group(3)
        ampm2 = m.group(6)
        if ampm1:
            if ampm1.lower() == "pm" and h1 != 12:
                h1 += 12
            if ampm1.lower() == "am" and h1 == 12:
                h1 = 0
        if ampm2:
            if ampm2.lower() == "pm" and h2 != 12:
                h2 += 12
            if ampm2.lower() == "am" and h2 == 12:
                h2 = 0
        hour_range = (max(0, min(23, h1)), max(0, min(23, h2)))

    # Specific keywords today/tomorrow
    for kw, offset in _TIME_KEYWORDS.items():
        if kw in q:
            d = (now + timedelta(days=offset)).date().isoformat()
            return d, None, hour_range

    # Next weekday like "next Saturday"
    weekdays = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
    m = re.search(r"next\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)", q)
    if m:
        target_idx = weekdays.index(m.group(1))
        cur_idx = now.weekday()
        delta = (target_idx - cur_idx) % 7
        delta = delta + 7 if delta == 0 else delta
        d = (now + timedelta(days=delta)).date().isoformat()
        return d, None, hour_range

    # First week of a named month (very rough heuristic)
    m = re.search(r"first\s+week\s+of\s+(january|february|march|april|may|june|july|august|september|october|november|december)", q)
    if m:
        month_name = m.group(1)
        month_map = {name: i + 1 for i, name in enumerate(["january","february","march","april","may","june","july","august","september","october","november","december"])}
        month = month_map[month_name]
        year = now.year if month >= now.month else now.year + 1
        start = datetime(year, month, 1).date()
        end = (start + timedelta(days=6))
        return None, (start.isoformat(), end.isoformat()), hour_range

    # Default: today
    return now.date().isoformat(), None, hour_range


def _merge_locations(*lists: List[str]) -> List[str]:
    out: List[str] = []
    seen: set = set()
    for lst in lists:
        for item in lst:
            lc = item.lower()
            if lc and lc not in seen:
                seen.add(lc)
                out.append(lc)
    return out


def parse_query_hybrid(question: str) -> ParsedQuery:
    # 1) Try Granite structured parsing
    locations_llm: List[str] = []
    attributes_llm: List[str] = []
    intent_llm: Optional[str] = None
    date_llm: Optional[str] = None
    date_range_llm: Optional[Tuple[str, str]] = None
    hour_range_llm: Optional[Tuple[int, int]] = None

    try:
        system = (
            "Extract structured weather query as strict JSON. Keys: "
            "locations (array of strings), attributes (subset of ['temperature','precipitation','humidity','wind','pressure','general']), "
            "intent (one of 'forecast','comparison','clothing','travel','general'), "
            "date (ISO date or null), date_range ([startISO,endISO] or null), hour_range ([startHour,endHour] or null)."
        )
        prompt = (
            "Question:\n" + question + "\n"+
            "Return JSON with keys: locations, attributes, intent, date, date_range, hour_range."
        )
        raw = ollama_chat(prompt, system=system, model=GRANITE_MODEL, host=OLLAMA_HOST, json_mode=True, temperature=0.1)
        txt = raw.strip()
        if txt.startswith("```"):
            txt = txt.strip("`")
            txt = txt.split("\n", 1)[1] if "\n" in txt else txt
        parsed = json.loads(txt)
        # Normalize
        locations_llm = [str(x).strip().lower() for x in (parsed.get("locations") or [])]
        attributes_llm = [a for a in (parsed.get("attributes") or []) if a in _KNOWN_ATTRS] or ["general"]
        intent_llm = (parsed.get("intent") or "forecast").strip().lower()
        d = parsed.get("date")
        if d:
            date_llm = str(d)
        dr = parsed.get("date_range")
        if isinstance(dr, list) and len(dr) == 2:
            date_range_llm = (str(dr[0]), str(dr[1]))
        hr = parsed.get("hour_range")
        if isinstance(hr, list) and len(hr) == 2:
            hour_range_llm = (int(hr[0]), int(hr[1]))
    except Exception:
        pass

    # 2) NER + regex extraction
    locations_regex = _extract_locations_regex(question)
    locations_spacy = _extract_locations_spacy(question)
    locations_hf = _extract_locations_hf(question)
    locations = _merge_locations(locations_llm, locations_spacy, locations_hf, locations_regex)
    if not locations:
        # Simple fallback: look for common cities in the question
        for city in ["london","paris","new york","tokyo","sydney","moscow","berlin","madrid","rome","amsterdam","karachi","sukkur"]:
            if city in question.lower() and city not in locations:
                locations.append(city)

    attributes = attributes_llm or _extract_attributes(question)
    intent = intent_llm or _classify_intent(question)

    # 3) Flexible time parsing (fallback if LLM didn't provide)
    date = date_llm
    date_range = date_range_llm
    hour_range = hour_range_llm
    if not any([date, date_range]):
        date, date_range, hr = _parse_timeframe(question)
        if hr and not hour_range:
            hour_range = hr

    return ParsedQuery(
        locations=locations or [""],
        attributes=attributes,
        intent=intent,
        date=date,
        date_range=date_range,
        hour_range=hour_range,
        original_question=question,
    )


# -----------------------------------------------------------------------------
# Weather data access (wttr.in)
# -----------------------------------------------------------------------------
def _safe_int(val: Any, default: int = 0) -> int:
    try:
        return int(val)
    except Exception:
        return default


def _safe_float(val: Any, default: float = 0.0) -> float:
    try:
        return float(val)
    except Exception:
        return default


def fetch_weather(location: str, forecast_days: int = 5) -> Dict[str, Any]:
    url = f"https://wttr.in/{location}?format=j1"
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        return {"success": False, "location": location, "error": f"Fetch error: {e}"}

    try:
        current = data.get("current_condition", [{}])[0] or {}
        current_weather = {
            "temperature": _safe_int(current.get("temp_C")),
            "feels_like": _safe_int(current.get("FeelsLikeC")),
            "humidity": _safe_int(current.get("humidity")),
            "description": ((current.get("weatherDesc") or [{"value": "N/A"}])[0]).get("value", "N/A"),
            "precipitation": _safe_float(current.get("precipMM")),
            "wind_speed": _safe_int(current.get("windspeedKmph")),
            "pressure": _safe_int(current.get("pressure")),
        }

        forecast: List[Dict[str, Any]] = []
        weather_days = data.get("weather") or []
        for i in range(min(forecast_days, len(weather_days))):
            day = weather_days[i] or {}
            snow_cm = _safe_float(day.get("totalSnow_cm"))
            precip_mm = _safe_float(day.get("totalprecip_mm"))
            if precip_mm == 0 and day.get("hourly"):
                try:
                    hrs = day.get("hourly") or []
                    precip_mm = sum(_safe_float(h.get("precipMM")) for h in hrs) / max(len(hrs), 1)
                except Exception:
                    precip_mm = 0.0
            hourly = day.get("hourly") or []
            # humidity
            try:
                avg_humidity = _safe_int(day.get("avghumidity"))
            except Exception:
                avg_humidity = 0
            if not avg_humidity and hourly:
                hv = [_safe_int(h.get("humidity")) for h in hourly]
                avg_humidity = _safe_int(sum(hv) / max(len(hv), 1)) if hv else 0

            forecast.append({
                "date": day.get("date", ""),
                "max_temp": _safe_int(day.get("maxtempC")),
                "min_temp": _safe_int(day.get("mintempC")),
                "avg_temp": _safe_int(day.get("avgtempC"), default=(_safe_int(day.get("maxtempC")) + _safe_int(day.get("mintempC"))) // 2),
                "precipitation": snow_cm + precip_mm,
                "humidity": avg_humidity,
                "description": ((hourly[0].get("weatherDesc") if hourly else [{"value": "N/A"}]) or [{"value": "N/A"}])[0].get("value", "N/A"),
                "hourly": hourly,
            })

        return {"success": True, "location": location, "current": current_weather, "forecast": forecast}
    except Exception as e:
        return {"success": False, "location": location, "error": f"Parsing error: {e}"}


# -----------------------------------------------------------------------------
# Response generation
# -----------------------------------------------------------------------------
def _select_day_index(forecast: List[Dict[str, Any]], date_iso: Optional[str]) -> int:
    if not date_iso:
        return 0
    for idx, day in enumerate(forecast):
        if day.get("date") == date_iso:
            return idx
    return 0


def _hourly_summary(hourlies: List[Dict[str, Any]], start_h: int, end_h: int) -> Dict[str, Any]:
    start_h = max(0, min(23, start_h))
    end_h = max(0, min(23, end_h))
    if end_h < start_h:
        start_h, end_h = end_h, start_h

    # wttr.in times are in steps like 0, 300, 600 ... representing minutes from midnight
    def slot_to_hour(slot: Any) -> int:
        try:
            return max(0, min(23, int(slot) // 100))
        except Exception:
            return 0

    sel = [h for h in hourlies if start_h <= slot_to_hour(h.get("time", 0)) <= end_h]
    if not sel:
        return {"avg_temp": None, "total_precip": 0.0, "desc": "N/A"}
    temps = [_safe_int(h.get("tempC")) for h in sel]
    precs = [_safe_float(h.get("precipMM")) for h in sel]
    desc = ((sel[0].get("weatherDesc") or [{"value": "N/A"}])[0]).get("value", "N/A")
    return {
        "avg_temp": sum(temps) / max(len(temps), 1) if temps else None,
        "total_precip": sum(precs),
        "desc": desc,
    }


def _format_city_header(city: str) -> str:
    return city.title()


def _format_hour_range(hr: Optional[Tuple[int, int]]) -> str:
    if not hr:
        return ""
    s, e = hr
    return f" between {s:02d}:00–{e:02d}:00"


def generate_response(parsed: ParsedQuery, city_to_weather: Dict[str, Dict[str, Any]]) -> str:
    intent = parsed.intent
    cities = parsed.locations

    # Comparison intent across multiple cities
    if intent == "comparison" and len(cities) >= 2:
        lines: List[str] = []
        lines.append("Comparison:")
        for city in cities[:5]:
            wd = city_to_weather.get(city, {})
            if not wd or not wd.get("success"):
                lines.append(f"- {city.title()}: unavailable")
                continue
            date_idx = _select_day_index(wd["forecast"], parsed.date)
            day = wd["forecast"][date_idx]
            lines.append(
                f"- {city.title()}: {day['min_temp']}–{day['max_temp']}°C, {day['description']}, {day['precipitation']:.1f} mm"
            )
        return "\n".join(lines)

    # Hourly-specific answer
    if parsed.hour_range:
        city = cities[0]
        wd = city_to_weather.get(city, {})
        if not wd or not wd.get("success"):
            return f"Sorry, I couldn't get weather data for {city}. {wd.get('error','Please try again.') if wd else ''}"
        date_idx = _select_day_index(wd["forecast"], parsed.date)
        day = wd["forecast"][date_idx]
        hr = parsed.hour_range
        summary = _hourly_summary(day.get("hourly", []), hr[0], hr[1])
        t = summary.get("avg_temp")
        p = summary.get("total_precip", 0.0)
        d = summary.get("desc", "N/A")
        date_label = parsed.date or day.get("date") or "today"
        return (
            f"{_format_city_header(city)}{_format_hour_range(parsed.hour_range)} on {date_label}: "
            f"around {t:.0f}°C with {d.lower()} ({p:.1f} mm)." if t is not None else
            f"{_format_city_header(city)}{_format_hour_range(parsed.hour_range)} on {date_label}: {d.lower()} ({p:.1f} mm)."
        )

    # Default forecast answer for a single city
    city = cities[0]
    wd = city_to_weather.get(city, {})
    if not wd or not wd.get("success"):
        return f"Sorry, I couldn't get weather data for {city}. {wd.get('error','Please try again.') if wd else ''}"
    date_idx = _select_day_index(wd["forecast"], parsed.date)
    day = wd["forecast"][date_idx]
    lines: List[str] = []
    lines.append(f"Weather for {city.title()} on {day.get('date','today')}:")
    lines.append(f"- Temperature: {day['min_temp']}–{day['max_temp']}°C (avg {day['avg_temp']}°C)")
    lines.append(f"- Conditions: {day['description']}")
    lines.append(f"- Precipitation: {day['precipitation']:.1f} mm")
    lines.append(f"- Humidity: {day['humidity']}%")
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
def ask_weather(question: str) -> str:
    parsed = parse_query_hybrid(question)

    # Fetch data for all mentioned cities
    city_to_weather: Dict[str, Dict[str, Any]] = {}
    for city in parsed.locations[:5]:  # cap to avoid excess calls
        if not city:
            continue
        city_to_weather[city] = fetch_weather(city, forecast_days=5)

    # Generate response
    return generate_response(parsed, city_to_weather)


__all__ = [
    "ask_weather",
    "parse_query_hybrid",
    "fetch_weather",
    "generate_response",
    "ParsedQuery",
]

import json
import urllib.request
import requests
import matplotlib.pyplot as plt
import pyinputplus as pyip
import re
import os

# Ollama / Granite configuration
OLLAMA_HOST = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
GRANITE_MODEL = os.getenv('GRANITE_MODEL', 'granite3.2')

# Configure matplotlib for better display
plt.style.use('default')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

print("✅ WeatherWise - Weather-Aware Chatbot")
print("📦 All required packages imported successfully!")
print("🤖 LLM (Ollama) host:", OLLAMA_HOST)
print("🧠 Granite model:", GRANITE_MODEL)
print("🚀 Ready to start building your weather chatbot!")


def ollama_chat(prompt, system=None, model=None, host=None, json_mode=False, temperature=0.2):
    print("def ollama chat called")
    """
    Call the Ollama chat API and return the response text.

    Args:
        prompt (str): User prompt
        system (str): Optional system prompt
        model (str): Model name (defaults to GRANITE_MODEL)
        host (str): Ollama host (defaults to OLLAMA_HOST)
        json_mode (bool): If True, request JSON-safe output
        temperature (float): Sampling temperature

    Returns:
        str: Model response text
    """
    model = model or GRANITE_MODEL
    host = host or OLLAMA_HOST
    url = f"{host}/api/chat"
    headers = {"Content-Type": "application/json"}
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    body = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    
    if json_mode:
        body["format"] = "json"

    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=60) as resp:
        raw = json.loads(resp.read().decode("utf-8"))
    # Ollama returns either 'message': {'content': ...} or 'response'
    if "message" in raw and isinstance(raw["message"], dict):
        return raw["message"].get("content", "")
    return raw.get("response", "")


def get_weather_data(location, forecast_days=5):
    print("def get_weather_data called")
    
    """
    Retrieve weather data for a specified location using wttr.in API.

    Args:
        location (str): City or location name
        forecast_days (int): Number of days to forecast (1-5)

    Returns:
        dict: Weather data including current conditions and forecast
    """
    try:
        # Make API request to wttr.in
        url = f"https://wttr.in/{location}?format=j1"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        # Extract current weather
        current = data['current_condition'][0]
        current_weather = {
            'temperature': int(current['temp_C']),
            'feels_like': int(current['FeelsLikeC']),
            'humidity': int(current['humidity']),
            'description': current['weatherDesc'][0]['value'],
            'precipitation': float(current['precipMM']),
            'wind_speed': int(current['windspeedKmph']),
            'pressure': int(current['pressure'])
        }
        
        # Extract forecast data
        forecast = []
        weather_days = data.get('weather') or []
        for i in range(min(forecast_days, len(weather_days))):
            day_data = weather_days[i] or {}
            # Handle precipitation data safely
            snow_cm = float(day_data.get('totalSnow_cm', 0) or 0)
            precip_mm = float(day_data.get('totalprecip_mm', 0) or 0)
            if precip_mm == 0 and 'hourly' in day_data:
                try:
                    precip_mm = sum(float(h.get('precipMM', 0) or 0) for h in (day_data.get('hourly') or [])) / max(len(day_data.get('hourly') or []), 1)
                except Exception:
                    precip_mm = 0.0
            precipitation = snow_cm + precip_mm
            
            # Compute average humidity safely (fallback to averaging hourly humidity)
            avg_humidity = None
            try:
                if 'avghumidity' in day_data and day_data.get('avghumidity') not in (None, ""):
                    avg_humidity = int(day_data.get('avghumidity'))
            except Exception:
                avg_humidity = None
            if avg_humidity is None:
                hourly = day_data.get('hourly', []) or []
                humidity_values = []
                for h in hourly:
                    try:
                        humidity_values.append(int(h.get('humidity', 0) or 0))
                    except Exception:
                        continue
                avg_humidity = int(sum(humidity_values) / len(humidity_values)) if humidity_values else 0
            
            # Safe description extraction
            description = 'N/A'
            try:
                first_hour = (day_data.get('hourly') or [{}])[0]
                first_desc = (first_hour.get('weatherDesc') or [{'value': 'N/A'}])[0]
                description = first_desc.get('value', 'N/A')
            except Exception:
                pass
            
            # Safe temps
            def to_int_safe(x, default=0):
                try:
                    return int(x)
                except Exception:
                    return default
            max_temp = to_int_safe(day_data.get('maxtempC'))
            min_temp = to_int_safe(day_data.get('mintempC'))
            avg_temp = to_int_safe(day_data.get('avgtempC'), default=(max_temp + min_temp)//2 if (max_temp or min_temp) else 0)
            
            forecast.append({
                'date': day_data.get('date', f'Day {i+1}'),
                'max_temp': max_temp,
                'min_temp': min_temp,
                'avg_temp': avg_temp,
                'precipitation': float(f"{precipitation:.2f}"),
                'humidity': avg_humidity,
                'description': description
            })
        
        return {
            'location': location,
            'current': current_weather,
            'forecast': forecast,
            'success': True
        }
        
    except requests.exceptions.RequestException as e:
        return {
            'location': location,
            'error': f"Failed to fetch weather data: {str(e)}",
            'success': False
        }
    except KeyError as e:
        return {
            'location': location,
            'error': f"Invalid location or data format error: {str(e)}",
            'success': False
        }
    except Exception as e:
        return {
            'location': location,
            'error': f"Unexpected error: {str(e)}",
            'success': False
        }

# %% [markdown]
# ## 📊 Visualisation Functions

# %%
# Define create_temperature_visualisation() and create_precipitation_visualisation() here
def create_temperature_visualisation(weather_data, output_type='display'):
    print("create_temparature_visualisation called")
    
    """
    Create visualisation of temperature data.

    Args:
        weather_data (dict): The processed weather data
        output_type (str): Either 'display' to show in notebook or 'figure' to return the figure

    Returns:
        If output_type is 'figure', returns the matplotlib figure object
        Otherwise, displays the visualisation in the notebook
    """
    if not weather_data.get('success', False):
        print(f"Error: {weather_data.get('error', 'Unknown error')}")
        return None
    
    # Prepare data for plotting
    dates = []
    max_temps = []
    min_temps = []
    avg_temps = []
    
    # Add current day data
    current = weather_data['current']
    dates.append('Today')
    max_temps.append(current['temperature'])
    min_temps.append(current['temperature'])
    avg_temps.append(current['temperature'])
    
    # Add forecast data
    for day in weather_data['forecast']:
        dates.append(day['date'])
        max_temps.append(day['max_temp'])
        min_temps.append(day['min_temp'])
        avg_temps.append(day['avg_temp'])
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    plt.plot(dates, max_temps, 'r-o', label='Max Temperature', linewidth=2, markersize=6)
    plt.plot(dates, min_temps, 'b-o', label='Min Temperature', linewidth=2, markersize=6)
    plt.plot(dates, avg_temps, 'g-o', label='Average Temperature', linewidth=2, markersize=6)
    
    plt.title(f'Temperature Forecast for {weather_data["location"]}', fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Temperature (°C)', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if output_type == 'figure':
        return plt.gcf()
    else:
        plt.show()
        return None


# %%

def create_precipitation_visualisation(weather_data, output_type='display'):
    print("create_precipitation_visualisation called")
    
    """
    Create visualisation of precipitation data.

    Args:
        weather_data (dict): The processed weather data
        output_type (str): Either 'display' to show in notebook or 'figure' to return the figure

    Returns:
        If output_type is 'figure', returns the matplotlib figure object
        Otherwise, displays the visualisation in the notebook
    """
    if not weather_data.get('success', False):
        print(f"Error: {weather_data.get('error', 'Unknown error')}")
        return None
    
    # Prepare data for plotting
    dates = []
    precipitation = []
    
    # Add current day data
    current = weather_data['current']
    dates.append('Today')
    precipitation.append(current['precipitation'])
    
    # Add forecast data
    for day in weather_data['forecast']:
        dates.append(day['date'])
        precipitation.append(day['precipitation'])
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    bars = plt.bar(dates, precipitation, color='skyblue', edgecolor='navy', alpha=0.7)
    
    # Add value labels on top of bars
    for bar, value in zip(bars, precipitation):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{value:.1f}mm', ha='center', va='bottom', fontweight='bold')
    
    plt.title(f'Precipitation Forecast for {weather_data["location"]}', fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Precipitation (mm)', fontsize=12)
    plt.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if output_type == 'figure':
        return plt.gcf()
    else:
        plt.show()
        return None

# %% [markdown]
# ## 🤖 Natural Language Processing

# %%
# 🤖 LLM-powered parsing (Granite via Ollama) with fallback
import json as _json


def _rule_based_parse(question):
    print("_rule_based_parse called")
    
    import re as _re
    q = question.lower()
    location_patterns = [
        r"\bin\s+([a-zA-Z\s]+?)(?:\s|$|,|\.|\?)",
        r"weather\s+for\s+([a-zA-Z\s]+?)(?:\s|$|,|\.|\?)",
        r"([a-zA-Z\s]+?)\s+weather",
        r"forecast\s+for\s+([a-zA-Z\s]+?)(?:\s|$|,|\.|\?)",
    ]
    location = None
    for pat in location_patterns:
        m = _re.search(pat, q)
        if m:
            location = m.group(1).strip()
            break
    if not location:
        for city in ['london','paris','new york','tokyo','sydney','moscow','berlin','madrid','rome','amsterdam']:
            if city in q:
                location = city
                break
    attributes = []
    if any(w in q for w in ['temperature','temp','hot','cold','warm','cool']): attributes.append('temperature')
    if any(w in q for w in ['rain','precipitation','rainfall','snow','storm']): attributes.append('precipitation')
    if any(w in q for w in ['humidity','humid','moisture']): attributes.append('humidity')
    if any(w in q for w in ['wind','breeze','gust']): attributes.append('wind')
    if any(w in q for w in ['pressure','barometric']): attributes.append('pressure')
    if not attributes: attributes.append('general')
    timeframe = 'today'
    if any(w in q for w in ['tomorrow','next day']): timeframe = 'tomorrow'
    elif any(w in q for w in ['week','7 days','seven days']): timeframe = 'week'
    elif any(w in q for w in ['3 days','three days','next 3 days']): timeframe = '3days'
    elif any(w in q for w in ['5 days','five days','next 5 days']): timeframe = '5days'
    return {'location': location, 'attributes': attributes, 'timeframe': timeframe, 'original_question': question}


def parse_weather_question(question):
    print("_parse_weather_question called")
    
    """Use Granite (via Ollama) to parse question; fallback to rule-based on error."""
    try:
        system = (
            "You extract weather query parameters and return ONLY strict JSON. "
            "Keys: location (string|null), attributes (array from ['temperature','precipitation','humidity','wind','pressure','general']), "
            "timeframe (one of 'today','tomorrow','3days','5days','week')."
        )
        prompt = (
            "Question:\n" + question + "\n"+
            "Return JSON: {\"location\":...,\"attributes\":...,\"timeframe\":...}"
        )
        raw = ollama_chat(prompt, system=system, model=GRANITE_MODEL, host=OLLAMA_HOST, json_mode=True, temperature=0.1)
        txt = raw.strip()
        if txt.startswith('```'):
            txt = txt.strip('`')
            txt = txt.split('\n',1)[1] if '\n' in txt else txt
        parsed = _json.loads(txt)
        location = parsed.get('location') or None
        if isinstance(location, str): location = location.strip()
        allowed = {'temperature','precipitation','humidity','wind','pressure','general'}
        attributes = [a for a in (parsed.get('attributes') or []) if isinstance(a, str) and a in allowed]
        if not attributes: attributes = ['general']
        timeframe = parsed.get('timeframe') or 'today'
        if timeframe not in ['today','tomorrow','3days','5days','week']: timeframe = 'today'
        return {'location': location, 'attributes': attributes, 'timeframe': timeframe, 'original_question': question}
    except Exception:
        return _rule_based_parse(question)


# %% [markdown]
# ## 🧭 User Interface

# %%
# 🤖 LLM-powered response generation with fallback

def _rule_based_response(parsed_question, weather_data):
    print("_rule_based_response called")
    
    # reuse existing deterministic logic by calling the earlier function if present
    return generate_weather_response(parsed_question, weather_data)  # will be overwritten below


def generate_weather_response(parsed_question, weather_data):
    print("_generate_weather_response called")
    
    """
    Prefer Granite via Ollama to compose the answer; fallback to deterministic response.
    """
    try:
        if not weather_data.get('success', False):
            return f"Sorry, I couldn't get weather data for {parsed_question.get('location') or 'that location'}. {weather_data.get('error', 'Please try again.')}"
        system = (
            "You are WeatherWise, a friendly weather assistant. You will be given parsed user intent and weather data. "
            "Produce a concise, friendly answer in natural language, with bullet points for key facts."
        )
        import json as _json
        payload = {
            'parsed_question': parsed_question,
            'weather_data': weather_data,
        }
        prompt = (
            "Use the following JSON as context and answer the user's weather question.\n" +
            _json.dumps(payload, ensure_ascii=False)
        )
        answer = ollama_chat(prompt, system=system, model=GRANITE_MODEL, host=OLLAMA_HOST, json_mode=False, temperature=0.4)
        if isinstance(answer, str) and answer.strip():
            return answer.strip()
        raise ValueError('Empty LLM response')
    except Exception:
        # Fallback: deterministic response using the previous rule-based generator body (copied here)
        if not weather_data.get('success', False):
            return f"Sorry, I couldn't get weather data for {parsed_question['location'] or 'that location'}. {weather_data.get('error', 'Please try again.')}"
        location = weather_data['location']
        current = weather_data['current']
        forecast = weather_data['forecast']
        attributes = parsed_question['attributes']
        timeframe = parsed_question['timeframe']
        response_parts = []
        response_parts.append(f"Here's the weather information for {location.title()}:")
        if timeframe == 'today':
            response_parts.append(f"\n🌤️ **Current Conditions:**")
            response_parts.append(f"• Temperature: {current['temperature']}°C (feels like {current['feels_like']}°C)")
            response_parts.append(f"• Description: {current['description']}")
            response_parts.append(f"• Humidity: {current['humidity']}%")
            response_parts.append(f"• Precipitation: {current['precipitation']}mm")
            response_parts.append(f"• Wind Speed: {current['wind_speed']} km/h")
            response_parts.append(f"• Pressure: {current['pressure']} mb")
        elif timeframe == 'tomorrow' and len(forecast) > 0:
            tomorrow = forecast[0]
            response_parts.append(f"\n🌅 **Tomorrow's Forecast:**")
            response_parts.append(f"• High: {tomorrow['max_temp']}°C, Low: {tomorrow['min_temp']}°C")
            response_parts.append(f"• Description: {tomorrow['description']}")
            response_parts.append(f"• Precipitation: {tomorrow['precipitation']}mm")
            response_parts.append(f"• Humidity: {tomorrow['humidity']}%")
        else:
            days_to_show = 1
            if timeframe == '3days':
                days_to_show = min(3, len(forecast))
            elif timeframe == '5days':
                days_to_show = min(5, len(forecast))
            elif timeframe == 'week':
                days_to_show = min(7, len(forecast))
            response_parts.append(f"\n📅 **{days_to_show}-Day Forecast:**")
            for i in range(days_to_show):
                day = forecast[i]
                response_parts.append(f"• {day['date']}: {day['min_temp']}°C - {day['max_temp']}°C, {day['description']}, {day['precipitation']}mm rain")
        if 'temperature' in attributes:
            if timeframe == 'today':
                response_parts.append(f"\n🌡️ **Temperature Details:**")
                response_parts.append(f"Current temperature is {current['temperature']}°C, but it feels like {current['feels_like']}°C.")
            else:
                response_parts.append(f"\n🌡️ **Temperature Trend:**")
                temps = [day['avg_temp'] for day in forecast[:days_to_show]]
                avg_temp = sum(temps) / len(temps)
                response_parts.append(f"Average temperature over the period: {avg_temp:.1f}°C")
        if 'precipitation' in attributes:
            if timeframe == 'today':
                response_parts.append(f"\n🌧️ **Precipitation:**")
                response_parts.append(f"Current precipitation: {current['precipitation']}mm")
            else:
                response_parts.append(f"\n🌧️ **Precipitation Forecast:**")
                total_precip = sum(day['precipitation'] for day in forecast[:days_to_show])
                response_parts.append(f"Total expected precipitation: {total_precip:.1f}mm over {days_to_show} days")
        if 'humidity' in attributes:
            if timeframe == 'today':
                response_parts.append(f"\n💧 **Humidity:**")
                response_parts.append(f"Current humidity: {current['humidity']}%")
            else:
                response_parts.append(f"\n💧 **Humidity Forecast:**")
                humidities = [day['humidity'] for day in forecast[:days_to_show]]
                avg_humidity = sum(humidities) / len(humidities)
                response_parts.append(f"Average humidity: {avg_humidity:.1f}%")
        response_parts.append(f"\n💡 **Recommendation:**")
        if current['temperature'] > 25:
            response_parts.append("It's quite warm! Stay hydrated and wear light clothing.")
        elif current['temperature'] < 10:
            response_parts.append("It's chilly! Bundle up and stay warm.")
        else:
            response_parts.append("Pleasant weather! Great for outdoor activities.")
        if current['precipitation'] > 5:
            response_parts.append("Don't forget your umbrella - there's significant precipitation expected!")
        return "\n".join(response_parts)


# %%
# Define menu functions using pyinputplus or ipywidgets here
def display_welcome():
    """Display welcome message and instructions."""
    print("=" * 60)
    print("🌦️  Welcome to WeatherWise - Your Weather-Aware Chatbot!  🌦️")
    print("=" * 60)
    print("Ask me anything about the weather in natural language!")
    print("Examples:")
    print("• 'What's the weather like in London?'")
    print("• 'Will it rain tomorrow in Paris?'")
    print("• 'Show me the temperature forecast for New York for the next 3 days'")
    print("• 'How humid is it in Tokyo?'")
    print("=" * 60)

def get_user_question():
    """Get weather question from user."""
    return pyip.inputStr("🌤️  Ask me about the weather: ", blank=False)

def display_menu():
    """Display main menu options."""
    print("\n" + "=" * 40)
    print("📋 Main Menu")
    print("=" * 40)
    print("1. Ask a weather question")
    print("2. Exit")
    print("=" * 40)

def run_chatbot():
    """Main chatbot loop."""
    display_welcome()
    
    while True:
        display_menu()
        try:
            choice = input("Choose an option: ").strip()
            
            if choice == '1':
                try:
                    # Get user question
                    question = input("🌤️  Ask me about the weather: ").strip()
                    if not question:
                        print("❌ Please enter a weather question.")
                        continue
                        
                    print(f"\n🤔 Processing: '{question}'")
                    print("-" * 50)
                    
                    # Parse the question
                    parsed = parse_weather_question(question)
                    
                    # Check if location was found
                    if not parsed['location']:
                        print("❌ I couldn't identify a location in your question.")
                        print("Please include a city name, like 'What's the weather in London?'")
                        continue
                    
                    # Get weather data
                    print(f"🌍 Fetching weather data for {parsed['location']}...")
                    weather_data = get_weather_data(parsed['location'])
                    
                    # Generate and display response
                    response = generate_weather_response(parsed, weather_data)
                    print(response)
                    
                    # Show visualizations if data is available
                    if weather_data.get('success', False):
                        print("\n📊 Generating visualizations...")
                        
                        # Show temperature chart
                        if 'temperature' in parsed['attributes'] or 'general' in parsed['attributes']:
                            create_temperature_visualisation(weather_data)
                        
                        # Show precipitation chart
                        if 'precipitation' in parsed['attributes'] or 'general' in parsed['attributes']:
                            create_precipitation_visualisation(weather_data)
                    
                    print("\n" + "=" * 50)
                    
                except KeyboardInterrupt:
                    print("\n\n👋 Goodbye! Thanks for using WeatherWise!")
                    break
                except Exception as e:
                    print(f"\n❌ An error occurred: {str(e)}")
                    print("Please try again with a different question.")
            
            elif choice == '2':
                print("\n👋 Thank you for using WeatherWise! Have a great day! 🌤️")
                break
            else:
                print("❌ Invalid choice. Please enter 1 or 2.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye! Thanks for using WeatherWise!")
            break
        except Exception as e:
            print(f"\n❌ An error occurred: {str(e)}")
            print("Please try again.")

# %% [markdown]
# ## 🧩 Main Application Logic

# %%
# Duplicate removed: using the LLM+fallback version of generate_weather_response defined earlier in the notebook.

# %% [markdown]
# ## 🧪 Testing and Examples

# %%
# Include sample input/output for each function

# Test the weather data function
print("🧪 Testing Weather Data Function")
print("=" * 40)
test_weather = get_weather_data("London", 3)
if test_weather['success']:
    print(f"✅ Successfully fetched weather for {test_weather['location']}")
    print(f"Current temperature: {test_weather['current']['temperature']}°C")
    print(f"Forecast days: {len(test_weather['forecast'])}")
else:
    print(f"❌ Error: {test_weather['error']}")

print("\n" + "=" * 40)

# Test the question parsing function
print("🧪 Testing Question Parsing")
print("=" * 40)
test_questions = [
    "What's the weather like in Paris?",
    "Will it rain tomorrow in Tokyo?",
    "Show me temperature for New York for next 3 days",
    "How humid is it in Sydney?"
]

for question in test_questions:
    parsed = parse_weather_question(question)
    print(f"Question: '{question}'")
    print(f"Parsed: Location='{parsed['location']}', Attributes={parsed['attributes']}, Timeframe='{parsed['timeframe']}'")
    print()

print("=" * 40)

# Test the response generation
print("🧪 Testing Response Generation")
print("=" * 40)
if test_weather['success']:
    test_parsed = parse_weather_question("What's the temperature in London?")
    response = generate_weather_response(test_parsed, test_weather)
    print("Sample response:")
    print(response[:200] + "..." if len(response) > 200 else response)
else:
    print("Skipping response test due to weather data error")

print("\n" + "=" * 40)
print("🎯 Ready to run the chatbot! Use run_chatbot() to start.")
print("=" * 40)

# %%
run_chatbot()


# %% [markdown]
# ## 🗂️ AI Prompting Log (Optional)
# Add markdown cells here summarising prompts used or link to AI conversations in the `ai-conversations/` folder.

# %%
# 🚀 Run the WeatherWise Chatbot
def run_chat():
    """Simple function to run the chatbot - same as run_chatbot()"""
    run_chatbot()

# Or run a quick demo
print("🎯 Quick Demo - WeatherWise Chatbot")
print("=" * 50)
print("To start the interactive chatbot, run: run_chat() or run_chatbot()")
print("=" * 50)


run_chatbot()


