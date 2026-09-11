import os
from dotenv import load_dotenv

load_dotenv()
token = os.getenv("ESIOS_TOKEN")
print("Token cargado:", token is not None)
print("Longitud:", len(token) if token else 0)