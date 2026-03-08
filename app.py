from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any
import pandas as pd
import numpy as np

app = FastAPI(title="District Heating Graph API")

CP_WATER = 4180.0


class CleanMergeRequest(BaseModel):
    substation_id: str
    substation_rows: List[Dict[str, Any]]
    outdoor_rows: List[Dict[str, Any]]
    building_rows: List[Dict[str, Any]]


def coerce_numeric(df: pd.DataFrame, cols):
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = (
                df[c].astype(str)
                .str.replace(",", ".", regex=False)
                .replace({"nan": np.nan, "None": np.nan, "": np.nan})
            )
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def find_col(df: pd.DataFrame, candidates):
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in df.columns:
            return cand
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/clean-merge-physics")
def clean_merge_physics(req: CleanMergeRequest):
    sub_df = pd.DataFrame(req.substation_rows)
    outdoor_df = pd.DataFrame(req.outdoor_rows)
    buildings_df = pd.DataFrame(req.building_rows)

    if sub_df.empty:
        return {
            "status": "ok",
            "substation_id": req.substation_id,
            "row_count": 0,
            "rows": []
        }

    ts_col = find_col(sub_df, ["timestamp", "time", "datetime", "date"])
    if ts_col is None:
        return {"status": "error", "message": "No timestamp column in substation_rows"}

    rename_map = {ts_col: "timestamp"}

    for target, candidates in {
        "supply_temp": ["supply_temp", "SupplyTemp", "supply"],
        "return_temp": ["return_temp", "ReturnTemp", "return"],
        "flow_rate": ["flow_rate", "FlowRate", "flow"],
        "heat_rate": ["heat_rate", "HeatRate", "heat"],
        "outdoor_temp": ["outdoor_temp"]
    }.items():
        c = find_col(sub_df, candidates)
        if c is not None:
            rename_map[c] = target

    sub_df = sub_df.rename(columns=rename_map)
    sub_df["timestamp"] = pd.to_datetime(sub_df["timestamp"], errors="coerce").dt.floor("h")

    for c in ["supply_temp", "return_temp", "flow_rate", "heat_rate"]:
        if c not in sub_df.columns:
            sub_df[c] = np.nan

    sub_df = coerce_numeric(sub_df, ["supply_temp", "return_temp", "flow_rate", "heat_rate", "outdoor_temp"])
    sub_df["substation_id"] = str(req.substation_id)

    if not outdoor_df.empty:
        ots_col = find_col(outdoor_df, ["timestamp", "time", "datetime", "date"])
        otemp_col = find_col(outdoor_df, ["outdoor_temp", "temperature", "temp", "Temp"])

        if ots_col is None or otemp_col is None:
            return {"status": "error", "message": "Outdoor rows need timestamp and temperature columns"}

        outdoor_df = outdoor_df.rename(columns={ots_col: "timestamp", otemp_col: "outdoor_temp"})
        outdoor_df["timestamp"] = pd.to_datetime(outdoor_df["timestamp"], errors="coerce").dt.floor("h")
        outdoor_df = coerce_numeric(outdoor_df, ["outdoor_temp"])
        outdoor_df = (
            outdoor_df[["timestamp", "outdoor_temp"]]
            .dropna(subset=["timestamp"])
            .sort_values("timestamp")
            .drop_duplicates(subset=["timestamp"], keep="last")
            .reset_index(drop=True)
        )

    if not buildings_df.empty:
        id_col = find_col(buildings_df, ["ID", "id", "substation_id", "sid"])
        if id_col is None:
            return {"status": "error", "message": "Building rows need an ID column"}

        rename_build = {id_col: "building_id"}

        for target, candidates in {
            "building_name": ["Name", "name", "building_name"],
            "area_m2": ["area", "area_m2", "Area"],
            "year_built": ["year_built", "YearBuilt", "year"],
            "building_type": ["Building type", "building_type", "type"]
        }.items():
            c = find_col(buildings_df, candidates)
            if c is not None:
                rename_build[c] = target

        buildings_df = buildings_df.rename(columns=rename_build)

        for c in ["building_name", "area_m2", "year_built", "building_type"]:
            if c not in buildings_df.columns:
                buildings_df[c] = np.nan

        buildings_df["building_id"] = buildings_df["building_id"].astype(str).str.strip()
        buildings_df = coerce_numeric(buildings_df, ["area_m2", "year_built"])
        buildings_df = buildings_df[
            ["building_id", "building_name", "area_m2", "year_built", "building_type"]
        ].drop_duplicates(subset=["building_id"], keep="first")

    sub_df = sub_df.drop(columns=["outdoor_temp"], errors="ignore")

    if not outdoor_df.empty:
        df = sub_df.merge(outdoor_df, on="timestamp", how="left")
        df = df.sort_values("timestamp").reset_index(drop=True)
        df["outdoor_temp"] = df["outdoor_temp"].interpolate(limit_direction="both").ffill().bfill()
    else:
        df = sub_df.copy()
        df["outdoor_temp"] = np.nan

    df["deltaT"] = df["supply_temp"] - df["return_temp"]
    df["heat_rate_calc"] = np.where(
        df["flow_rate"].notna() & df["deltaT"].notna(),
        (df["flow_rate"] * CP_WATER * df["deltaT"]) / 1000.0,
        df["heat_rate"]
    )

    if not buildings_df.empty:
        df = df.merge(buildings_df, left_on="substation_id", right_on="building_id", how="left")
    else:
        df["building_id"] = df["substation_id"]
        df["building_name"] = np.nan
        df["area_m2"] = np.nan
        df["year_built"] = np.nan
        df["building_type"] = np.nan

    df = df[
        [
            "timestamp",
            "substation_id",
            "supply_temp",
            "return_temp",
            "flow_rate",
            "heat_rate",
            "outdoor_temp",
            "deltaT",
            "heat_rate_calc",
            "building_id",
            "building_name",
            "area_m2",
            "year_built",
            "building_type"
        ]
    ].sort_values(["substation_id", "timestamp"]).reset_index(drop=True)

    df["timestamp"] = df["timestamp"].astype(str)

    return {
        "status": "ok",
        "substation_id": req.substation_id,
        "row_count": int(len(df)),
        "rows": df.replace({np.nan: None}).to_dict(orient="records")
    }
