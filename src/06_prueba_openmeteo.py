import requests

URL = "https://archive-api.open-meteo.com/v1/archive"
params = {
    "latitude": 37.39, "longitude": -5.98,
    "start_date": "2024-01-01", "end_date": "2024-01-02",
    "hourly": "temperature_2m", "models": "era5_land", "timezone": "GMT",
}

r = requests.get(URL, params=params, timeout=60)
if not r.ok:
    print("Error", r.status_code, r.text)
else:
    d = r.json()
    print("Celda:", d["latitude"], d["longitude"], "| elevación:", d["elevation"])
    print("Nº de horas:", len(d["hourly"]["time"]))
    print(list(zip(d["hourly"]["time"][:5], d["hourly"]["temperature_2m"][:5])))