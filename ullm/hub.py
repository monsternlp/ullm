import datetime
import json
import os
from pathlib import Path

from peewee import BooleanField, CharField, DateTimeField
from peewee import Model as DatabaseModel
from playhouse.db_url import connect

from .base import LanguageModel, RemoteLanguageModel


class LanguageModelTable(DatabaseModel):
    remote = BooleanField(index=True)
    model_id = CharField(primary_key=True)
    model_name = CharField(index=True)
    model_config = CharField()
    created = DateTimeField(default=datetime.datetime.utcnow)

    class Meta:
        db_table = "language_model"


DEFAULT_DB_URL = "sqlite:///" + str(Path.home() / ".ullm.db")


class ModelHub:
    def __init__(self, db_url=None):
        db_url = db_url or os.environ.get("ULLM_DB_URL") or DEFAULT_DB_URL
        self.database = connect(db_url)
        self.database.bind([LanguageModelTable])
        if not self.database.table_exists(LanguageModelTable._meta.table_name):
            with self.database.connection_context():
                self.database.create_tables([LanguageModelTable])

    def register_model(self, model: LanguageModel, model_id: str):
        with self.database.connection_context():
            LanguageModelTable.create(
                remote=isinstance(model, RemoteLanguageModel),
                model_id=model_id,
                model_name=model.model,
                model_config=model.config.model_dump_json(),
            )

    def list_models(self):
        models = []
        with self.database.connection_context():
            for model in LanguageModelTable.select():
                models.append(model)

        return models

    def get_model(self, model_id: str):
        model = None
        with self.database.connection_context():
            model_record = LanguageModelTable.get_or_none(LanguageModelTable.model_id == model_id)
            if model_record:
                model = LanguageModel.from_config(json.loads(model_record.model_config))

        return model
