import dotenv
import os

from dagster_duckdb import DuckDBResource
from dagster import (
    AssetSelection,
    ScheduleDefinition,
    Definitions,
    define_asset_job,
    load_assets_from_modules
)
from . import assets

atp_job = define_asset_job("atp_job", selection=AssetSelection.all())

atp_schedule = ScheduleDefinition(
    job=atp_job,
    cron_schedule="0 * * * *"
)

all_assets = load_assets_from_modules([assets])

defs = Definitions(
    assets=all_assets,
    jobs=[atp_job],
    resources={
        "duckdb": DuckDBResource(
            database="atp.duckdb"
        )
    },
    schedules=[atp_schedule]
)