from pydantic import BaseSettings, SecretStr


class Settings(BaseSettings):
    llm_api_key: SecretStr
    llm_base_url: str | None = None
    llm_model: str

    class Config:
        env_prefix = "LLM_"
        env_file = ".env"
