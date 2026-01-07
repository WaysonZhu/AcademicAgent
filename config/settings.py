import os
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()

class Settings:
    # Redis Configuration
    REDIS_HOST = os.getenv("REDIS_HOST", "127.0.0.1")
    REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
    REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "redis.123456")
    REDIS_DB = int(os.getenv("REDIS_DB", 0))

    # API Configuration
    API_BASE_URL = "https://ai4scholar.net/graph/v1/paper"
    AI4SCHOLAR_API_KEY = os.getenv("AI4SCHOLAR_API_KEY", "")

    # LLM Configuration (Qwen-Max)
    DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
    MODEL_NAME = "qwen-max"

    # Project Paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    LOG_DIR = os.path.join(BASE_DIR, "logs")
    DATA_DIR = os.path.join(BASE_DIR, "data")

    # Ensure directories exist
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

settings = Settings()