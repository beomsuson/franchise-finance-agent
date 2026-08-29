import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

# 배포판은 정보공개서 데이터가 읽기 전용·소규모라 별도 클라우드 DB 없이
# franchise_agent/franchise.db(SQLite, 로컬 MySQL 뷰를 그대로 덤프한 파일)를 그대로 번들해서 쓴다.
_SQLITE_PATH = Path(__file__).resolve().parent.parent / "franchise.db"
DATABASE_URL = f"sqlite:///{_SQLITE_PATH}"
