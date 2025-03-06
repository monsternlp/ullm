import datetime
import json
from pathlib import Path
from typing import Literal, Optional

from peewee import BooleanField, CharField, DateTimeField
from peewee import Model as DatabaseModel
from playhouse.db_url import connect
from pydantic import model_validator
from pydantic_settings import BaseSettings
from redis import StrictRedis

from .base import LanguageModel, RemoteLanguageModel

DEFAULT_DB_URL = "sqlite:///" + str(Path.home() / ".ullm.db")
DEFAULT_REDIS_URL = "redis://localhost:6379/0"


class HubConfig(BaseSettings):
    class Config:
        env_prefix = "ULLM_"
        env_file = ".env"
        env_file_encoding = "utf8"
        extra = "ignore"

    HUB_BACKEND: Optional[Literal["redis", "rds"]] = None
    HUB_DB_URL: Optional[str] = None
    HUB_REDIS_PREFIX: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def set_default_db_url(cls, values):
        backend = values.get("HUB_BACKEND", "rds")
        if not values.get("HUB_DB_URL"):
            if backend == "redis":
                values["HUB_DB_URL"] = DEFAULT_REDIS_URL
            elif backend == "rds":
                values["HUB_DB_URL"] = DEFAULT_DB_URL

        if not values.get("HUB_REDIS_PREFIX") and backend == "redis":
            values["HUB_REDIS_PREFIX"] = "/ullm/model_hub/"

        values["HUB_BACKEND"] = backend
        return values


HUB_CONFIG = HubConfig()


class LanguageModelTable(DatabaseModel):
    remote = BooleanField(index=True)
    model_id = CharField(primary_key=True)
    model_name = CharField(index=True)
    model_config = CharField()
    created = DateTimeField(default=datetime.datetime.utcnow)

    class Meta:
        db_table = "language_model"


class ModelHub:
    def __init__(self, hub_backend=None, hub_db_url=None, hub_redis_prefix=None):
        if hub_backend or hub_db_url:
            config = HubConfig(
                HUB_BACKEND=hub_backend, HUB_DB_URL=hub_db_url, HUB_REDIS_PREFIX=hub_redis_prefix
            )
        else:
            config = HUB_CONFIG

        self.hub_backend = config.HUB_BACKEND
        self.hub_db_url = config.HUB_DB_URL
        self.hub_redis_prefix = config.HUB_REDIS_PREFIX
        if self.hub_backend == "rds":
            self.database = connect(self.hub_db_url)
            self.database.bind([LanguageModelTable])
            if not self.database.table_exists(LanguageModelTable._meta.table_name):
                with self.database.connection_context():
                    self.database.create_tables([LanguageModelTable])
        else:
            self.database = StrictRedis.from_url(self.hub_db_url)

    def register_model(self, model: LanguageModel, model_id: str):
        if self.hub_backend == "rds":
            with self.database.connection_context():
                LanguageModelTable.create(
                    remote=isinstance(model, RemoteLanguageModel),
                    model_id=model_id,
                    model_name=model.model,
                    model_config=model.config.model_dump_json(),
                )
        else:
            data = dict(
                remote=isinstance(model, RemoteLanguageModel),
                model_id=model_id,
                model_name=model.model,
                model_config=model.config.model_dump_json(),
                created=str(datetime.datetime.utcnow()),
            )
            key = self.hub_redis_prefix + "models"
            self.database.hset(key, model_id, json.dumps(data, ensure_ascii=False))

    def list_models(self):
        models = []
        if self.hub_backend == "rds":
            with self.database.connection_context():
                for model in LanguageModelTable.select():
                    models.append(model)
        else:
            key = self.hub_redis_prefix + "models"
            for model_id, model_data in self.database.hgetall(key).items():
                models.append(LanguageModelTable(**json.loads(model_data)))

        return models

    def get_model(self, model_id: str):
        model = None
        if self.hub_backend == "rds":
            with self.database.connection_context():
                model_record = LanguageModelTable.get_or_none(
                    LanguageModelTable.model_id == model_id
                )
                if model_record:
                    model = LanguageModel.from_config(json.loads(model_record.model_config))
        else:
            key = self.hub_redis_prefix + "models"
            model_data = self.database.hget(key, model_id)
            if model_data:
                model_data = json.loads(model_data)
                model = LanguageModel.from_config(json.loads(model_data["model_config"]))

        return model
