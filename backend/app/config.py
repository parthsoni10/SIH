import os
from pathlib import Path

# Load .env file using python-dotenv
BASE_DIR = Path(__file__).resolve().parent.parent
env_file = BASE_DIR / ".env"

try:
    from dotenv import load_dotenv
    if env_file.exists():
        load_dotenv(dotenv_path=env_file)
except ImportError:
    pass

def resolve_path(env_val: str, default_path: Path) -> Path:
    if not env_val:
        return default_path
    p = Path(env_val)
    if p.is_absolute():
        return p
    return BASE_DIR / p

class Settings:
    PROJECT_NAME: str = os.getenv("PROJECT_NAME", "AI-Based Fake Identity & Document Screening System")
    RISK_MODEL_PATH: Path = resolve_path(os.getenv("RISK_MODEL_PATH", ""), BASE_DIR / "app" / "models" / "risk_model.pkl")
    UPLOAD_DIR: Path = resolve_path(os.getenv("UPLOAD_DIR", ""), BASE_DIR / "uploads")
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", os.getenv("GOOGLE_API_KEY", os.getenv("MISTRAL_API_KEY", "")))
    MISTRAL_API_KEY: str = os.getenv("MISTRAL_API_KEY", "")
    MAX_IMAGE_EDGE: int = int(os.getenv("MAX_IMAGE_EDGE", "2000"))
    DATABASE_URL: str = os.getenv("DATABASE_URL", f"sqlite:///{BASE_DIR / 'screening.db'}")

settings = Settings()

# Ensure uploads directory exists
settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
