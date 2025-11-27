"""Drought and Water Budget Engine with 5-year synthetic data."""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from dataclasses import dataclass

# ============ SYNTHETIC DATA GENERATOR (5 YEARS) ============

def generate_5year_data(start_year: int = 2019) -> pd.DataFrame:
    """
    Generate 5 years of realistic Chennai/Tamil Nadu hydro-meteorological data.
    Includes known drought (2019) and flood (2015, 2021, 2023) patterns.
    """
    dates = pd.date_range(start=f"{start_year}-01-01", periods=365*5, freq='D')

    data = []
    for date in dates:
        month = date.month
        year = date.year
        day_of_year = date.dayofyear

        # === RAINFALL (mm) ===
        # Chennai: NE Monsoon (Oct-Dec), SW Monsoon (Jun-Sep)
        if month in [10, 11, 12]:  # NE Monsoon - peak
            base_rain = np.random.exponential(8)
            # Add extreme events for known flood years
            if year == 2021 and month == 11 and 7 <= date.day <= 12:
                base_rain = np.random.uniform(80, 200)  # 2021 floods
            elif year == 2023 and month == 11 and 10 <= date.day <= 15:
                base_rain = np.random.uniform(100, 350)  # Cyclone Michaung
        elif month in [6, 7, 8, 9]:  # SW Monsoon - moderate
            base_rain = np.random.exponential(3)
        else:  # Dry season
            base_rain = np.random.exponential(0.5) if np.random.random() > 0.7 else 0

        # Drought year adjustment (2019 was dry)
        if year == 2019:
            base_rain *= 0.4

        rainfall = max(0, base_rain)

        # === TEMPERATURE (°C) ===
        # Chennai: 25-40°C range
        if month in [4, 5, 6]:  # Summer
            temp = np.random.normal(35, 2)
        elif month in [11, 12, 1]:  # Winter
            temp = np.random.normal(26, 2)
        else:
            temp = np.random.normal(30, 2)

        # === EVAPOTRANSPIRATION (mm/day) ===
        # Higher in summer, lower in monsoon
        if month in [4, 5, 6]:
            et = np.random.normal(6.5, 0.5)
        elif month in [10, 11, 12]:
            et = np.random.normal(3.5, 0.5)
        else:
            et = np.random.normal(5.0, 0.5)

        # === SOIL MOISTURE (%) ===
        # Based on cumulative rainfall - ET balance
        if month in [10, 11, 12] and rainfall > 10:
            soil_moisture = np.random.uniform(60, 90)
        elif month in [4, 5, 6]:
            soil_moisture = np.random.uniform(15, 35)
        else:
            soil_moisture = np.random.uniform(30, 55)

        # Drought adjustment
        if year == 2019:
            soil_moisture *= 0.6

        # === RESERVOIR STORAGE (%) ===
        # Poondi, Cholavaram, Red Hills, Chembarambakkam
        if month in [11, 12, 1]:  # Post-monsoon
            reservoir = np.random.uniform(70, 95)
        elif month in [5, 6, 7]:  # Pre-monsoon low
            reservoir = np.random.uniform(20, 45)
        else:
            reservoir = np.random.uniform(40, 65)

        # Drought year - critically low
        if year == 2019 and month in [5, 6, 7]:
            reservoir = np.random.uniform(5, 15)

        # === GROUNDWATER LEVEL (m below ground) ===
        # Higher value = deeper = worse
        if month in [11, 12, 1]:
            gw_level = np.random.uniform(3, 8)
        elif month in [5, 6, 7]:
            gw_level = np.random.uniform(12, 20)
        else:
            gw_level = np.random.uniform(8, 14)

        if year == 2019:
            gw_level += 5  # Drought - much deeper

        # === SPI (Standardized Precipitation Index) ===
        # -2 to +2 scale: <-1.5 severe drought, >1.5 flood
        if rainfall > 50:
            spi = np.random.uniform(1.0, 2.5)
        elif rainfall > 20:
            spi = np.random.uniform(0.5, 1.5)
        elif rainfall > 5:
            spi = np.random.uniform(-0.5, 0.5)
        elif rainfall > 0:
            spi = np.random.uniform(-1.5, -0.5)
        else:
            spi = np.random.uniform(-2.5, -1.0)

        if year == 2019:
            spi -= 0.8  # Push towards drought

        # === NDVI (Vegetation Index) ===
        # 0-1 scale, higher = healthier vegetation
        if month in [10, 11, 12, 1] and rainfall > 5:
            ndvi = np.random.uniform(0.5, 0.8)
        elif month in [4, 5, 6]:
            ndvi = np.random.uniform(0.15, 0.35)
        else:
            ndvi = np.random.uniform(0.3, 0.5)

        if year == 2019:
            ndvi *= 0.7

        data.append({
            'date': date,
            'year': year,
            'month': month,
            'rainfall_mm': round(rainfall, 1),
            'temperature_c': round(temp, 1),
            'et_mm': round(et, 2),
            'soil_moisture_pct': round(soil_moisture, 1),
            'reservoir_pct': round(reservoir, 1),
            'groundwater_m': round(gw_level, 1),
            'spi': round(spi, 2),
            'ndvi': round(ndvi, 2)
        })

    return pd.DataFrame(data)


# ============ WATER BUDGET CALCULATOR ============

@dataclass
class WaterBudget:
    """Water budget for a region."""
    total_supply_mcm: float  # Million Cubic Meters
    total_demand_mcm: float
    deficit_mcm: float
    surplus_mcm: float
    reservoir_storage_mcm: float
    groundwater_available_mcm: float
    rainfall_contribution_mcm: float
    et_loss_mcm: float
    irrigation_demand_mcm: float
    domestic_demand_mcm: float


def calculate_water_budget(
    area_ha: float,
    rainfall_mm: float,
    et_mm: float,
    reservoir_pct: float,
    crop_area_ha: float,
    crop_water_need_mm: float
) -> WaterBudget:
    """Calculate water budget for a region."""

    # Convert to MCM (Million Cubic Meters)
    rainfall_mcm = (rainfall_mm / 1000) * (area_ha * 10000) / 1e6
    et_loss_mcm = (et_mm / 1000) * (area_ha * 10000) / 1e6

    # Reservoir capacity (assume 4 reservoirs, total 800 MCM capacity)
    reservoir_capacity_mcm = 800
    reservoir_storage_mcm = reservoir_capacity_mcm * (reservoir_pct / 100)

    # Groundwater (rough estimate based on area)
    groundwater_mcm = area_ha * 0.05  # 0.05 MCM per hectare

    # Demands
    irrigation_demand_mcm = (crop_water_need_mm / 1000) * (crop_area_ha * 10000) / 1e6
    domestic_demand_mcm = area_ha * 0.01  # Rough estimate

    total_supply = rainfall_mcm + reservoir_storage_mcm + groundwater_mcm
    total_demand = irrigation_demand_mcm + domestic_demand_mcm + et_loss_mcm

    deficit = max(0, total_demand - total_supply)
    surplus = max(0, total_supply - total_demand)

    return WaterBudget(
        total_supply_mcm=round(total_supply, 2),
        total_demand_mcm=round(total_demand, 2),
        deficit_mcm=round(deficit, 2),
        surplus_mcm=round(surplus, 2),
        reservoir_storage_mcm=round(reservoir_storage_mcm, 2),
        groundwater_available_mcm=round(groundwater_mcm, 2),
        rainfall_contribution_mcm=round(rainfall_mcm, 2),
        et_loss_mcm=round(et_loss_mcm, 2),
        irrigation_demand_mcm=round(irrigation_demand_mcm, 2),
        domestic_demand_mcm=round(domestic_demand_mcm, 2)
    )


# ============ DROUGHT ASSESSMENT ============

@dataclass
class DroughtAssessment:
    """Drought assessment result."""
    drought_type: str  # meteorological, agricultural, hydrological
    severity: str  # none, mild, moderate, severe, extreme
    spi_value: float
    weeks_to_impact: int
    affected_area_pct: float
    crop_stress_level: str
    recommended_actions: List[str]


def assess_drought(
    spi: float,
    soil_moisture: float,
    reservoir_pct: float,
    ndvi: float,
    current_month: int
) -> DroughtAssessment:
    """Assess drought conditions and provide recommendations."""

    # Determine drought type and severity
    if spi < -2.0:
        severity = "extreme"
        drought_type = "meteorological"
    elif spi < -1.5:
        severity = "severe"
        drought_type = "meteorological"
    elif spi < -1.0:
        severity = "moderate"
        drought_type = "agricultural"
    elif spi < -0.5:
        severity = "mild"
        drought_type = "agricultural"
    else:
        severity = "none"
        drought_type = "none"

    # Check hydrological drought
    if reservoir_pct < 20:
        drought_type = "hydrological"
        if reservoir_pct < 10:
            severity = "extreme"

    # Estimate weeks to impact
    if severity == "extreme":
        weeks_to_impact = 2
    elif severity == "severe":
        weeks_to_impact = 4
    elif severity == "moderate":
        weeks_to_impact = 6
    else:
        weeks_to_impact = 8

    # Affected area
    if severity == "extreme":
        affected_area = 80
    elif severity == "severe":
        affected_area = 60
    elif severity == "moderate":
        affected_area = 40
    elif severity == "mild":
        affected_area = 20
    else:
        affected_area = 0

    # Crop stress
    if ndvi < 0.2:
        crop_stress = "critical"
    elif ndvi < 0.3:
        crop_stress = "severe"
    elif ndvi < 0.4:
        crop_stress = "moderate"
    elif ndvi < 0.5:
        crop_stress = "mild"
    else:
        crop_stress = "none"

    # Recommendations based on severity and season
    actions = []
    if severity in ["extreme", "severe"]:
        actions.append("Declare drought emergency")
        actions.append("Release contingency water from reservoirs")
        actions.append("Activate crop insurance claims process")
        actions.append("Advise farmers to stop water-intensive crops")
    elif severity == "moderate":
        actions.append("Issue drought advisory")
        actions.append("Reduce irrigation allocation by 30%")
        actions.append("Promote drought-resistant crop varieties")
    elif severity == "mild":
        actions.append("Monitor conditions closely")
        actions.append("Prepare contingency plans")

    # Season-specific
    if current_month in [6, 7, 8]:  # Kharif sowing
        actions.append("Recommend short-duration crops (millets, pulses)")
    elif current_month in [10, 11]:  # Rabi sowing
        actions.append("Delay sowing if water insufficient")

    return DroughtAssessment(
        drought_type=drought_type,
        severity=severity,
        spi_value=spi,
        weeks_to_impact=weeks_to_impact,
        affected_area_pct=affected_area,
        crop_stress_level=crop_stress,
        recommended_actions=actions
    )


# ============ FARMER DECISION SUPPORT ============

@dataclass
class FarmerDecision:
    """Decision option for a farmer."""
    option_id: str
    title: str
    description: str
    investment_required: float
    expected_return: float
    risk_level: str
    water_required_mm: float
    probability_success: float


def generate_farmer_options(
    current_crop: str,
    crop_stage_days: int,
    water_available_mm: float,
    drought_severity: str,
    area_ha: float
) -> List[FarmerDecision]:
    """Generate decision options for a farmer based on conditions."""

    options = []

    # Crop water requirements (mm for full season)
    crop_water = {
        'paddy': 1200,
        'sugarcane': 1800,
        'cotton': 700,
        'groundnut': 500,
        'millets': 350,
        'pulses': 300
    }

    remaining_water_need = crop_water.get(current_crop, 600) * (1 - crop_stage_days/120)

    if drought_severity in ['severe', 'extreme']:
        # Option 1: Continue with reduced yield
        options.append(FarmerDecision(
            option_id="A",
            title="Continue with reduced yield",
            description=f"Continue {current_crop} cultivation with available water",
            investment_required=15000 * area_ha,
            expected_return=25000 * area_ha * 0.4,  # 40% yield
            risk_level="High",
            water_required_mm=remaining_water_need,
            probability_success=0.40
        ))

        # Option 2: Stop and claim insurance
        options.append(FarmerDecision(
            option_id="B",
            title="Stop cultivation, claim insurance",
            description="Declare crop loss and initiate PMFBY claim",
            investment_required=0,
            expected_return=25000 * area_ha,  # Insurance payout
            risk_level="Low",
            water_required_mm=0,
            probability_success=0.85
        ))

        # Option 3: Switch to short-duration crop
        options.append(FarmerDecision(
            option_id="C",
            title="Switch to drought-resistant crop",
            description="Plant millets/pulses (60-75 days)",
            investment_required=8000 * area_ha,
            expected_return=18000 * area_ha,
            risk_level="Medium",
            water_required_mm=300,
            probability_success=0.70
        ))

    elif drought_severity == 'moderate':
        # Option 1: Continue with supplemental irrigation
        options.append(FarmerDecision(
            option_id="A",
            title="Continue with supplemental irrigation",
            description="Purchase water/use borewells",
            investment_required=20000 * area_ha,
            expected_return=40000 * area_ha * 0.7,
            risk_level="Medium",
            water_required_mm=remaining_water_need,
            probability_success=0.65
        ))

        # Option 2: Reduce area
        options.append(FarmerDecision(
            option_id="B",
            title="Reduce cultivated area by 40%",
            description="Focus water on 60% of field",
            investment_required=12000 * area_ha * 0.6,
            expected_return=40000 * area_ha * 0.6 * 0.85,
            risk_level="Low",
            water_required_mm=remaining_water_need * 0.6,
            probability_success=0.80
        ))

    else:  # Mild or none
        options.append(FarmerDecision(
            option_id="A",
            title="Continue normal cultivation",
            description="No changes needed",
            investment_required=10000 * area_ha,
            expected_return=40000 * area_ha,
            risk_level="Low",
            water_required_mm=remaining_water_need,
            probability_success=0.85
        ))

    return options


# ============ ADMINISTRATOR DECISION SUPPORT ============

@dataclass
class AdminDecision:
    """Decision support for administrators."""
    total_cultivated_ha: float
    saveable_area_ha: float
    partial_save_ha: float
    loss_area_ha: float
    water_required_mcm: float
    water_available_mcm: float
    compensation_required_cr: float
    crops_saved_value_cr: float
    net_benefit_cr: float
    priority_villages: List[str]
    recommended_actions: List[str]


def generate_admin_triage(
    total_area_ha: float,
    water_available_mcm: float,
    drought_severity: str,
    reservoir_pct: float
) -> AdminDecision:
    """Generate administrative triage for drought management."""

    # Water requirement per hectare (MCM)
    water_per_ha = 0.008  # 8000 cubic meters per hectare

    total_water_needed = total_area_ha * water_per_ha
    water_deficit_pct = max(0, (total_water_needed - water_available_mcm) / total_water_needed * 100)

    if drought_severity == 'extreme':
        saveable_pct = 0.30
        partial_pct = 0.20
        loss_pct = 0.50
    elif drought_severity == 'severe':
        saveable_pct = 0.50
        partial_pct = 0.25
        loss_pct = 0.25
    elif drought_severity == 'moderate':
        saveable_pct = 0.70
        partial_pct = 0.20
        loss_pct = 0.10
    else:
        saveable_pct = 0.90
        partial_pct = 0.08
        loss_pct = 0.02

    saveable_ha = total_area_ha * saveable_pct
    partial_ha = total_area_ha * partial_pct
    loss_ha = total_area_ha * loss_pct

    # Financial calculations
    compensation_per_ha = 25000  # Rs per hectare
    crop_value_per_ha = 40000  # Rs per hectare

    compensation_required = loss_ha * compensation_per_ha / 1e7  # In crores
    crops_saved_value = saveable_ha * crop_value_per_ha / 1e7
    net_benefit = crops_saved_value - compensation_required

    # Priority villages (demo)
    priority_villages = [
        "Kanchipuram Block - Zone 1",
        "Thiruvallur Block - Zone 3",
        "Chengalpattu Block - Zone 2"
    ]

    # Recommended actions
    actions = []
    if drought_severity in ['severe', 'extreme']:
        actions = [
            f"Release {water_available_mcm * 0.4:.1f} MCM from Poondi reservoir to priority zones",
            f"Issue advisory to {int(loss_ha)} hectares to stop cultivation",
            f"Initiate compensation process: ₹{compensation_required:.1f} Cr required",
            "Activate PMFBY claims for affected farmers",
            "Deploy mobile water tankers to critical villages"
        ]
    elif drought_severity == 'moderate':
        actions = [
            "Reduce irrigation allocation by 30%",
            "Implement rotational water supply",
            "Promote micro-irrigation adoption",
            "Monitor reservoir levels daily"
        ]
    else:
        actions = [
            "Continue normal operations",
            "Maintain buffer stock in reservoirs",
            "Monitor weather forecasts"
        ]

    return AdminDecision(
        total_cultivated_ha=total_area_ha,
        saveable_area_ha=saveable_ha,
        partial_save_ha=partial_ha,
        loss_area_ha=loss_ha,
        water_required_mcm=total_water_needed,
        water_available_mcm=water_available_mcm,
        compensation_required_cr=round(compensation_required, 2),
        crops_saved_value_cr=round(crops_saved_value, 2),
        net_benefit_cr=round(net_benefit, 2),
        priority_villages=priority_villages,
        recommended_actions=actions
    )


# ============ SCENARIO PRESETS ============

SCENARIOS = {
    "2019_drought": {
        "name": "2019 Chennai Water Crisis",
        "description": "Severe hydrological drought, reservoirs at 0.1%",
        "year": 2019,
        "month": 6,
        "conditions": {
            "rainfall_mm": 2.1,
            "reservoir_pct": 0.1,
            "groundwater_m": 25,
            "spi": -2.3,
            "ndvi": 0.18,
            "soil_moisture_pct": 12
        }
    },
    "2021_flood": {
        "name": "2021 Chennai Floods",
        "description": "Heavy NE monsoon, widespread urban flooding",
        "year": 2021,
        "month": 11,
        "conditions": {
            "rainfall_mm": 180,
            "reservoir_pct": 98,
            "groundwater_m": 3,
            "spi": 2.1,
            "ndvi": 0.65,
            "soil_moisture_pct": 92
        }
    },
    "2023_cyclone": {
        "name": "Cyclone Michaung 2023",
        "description": "Record 40cm rainfall in 24 hours",
        "year": 2023,
        "month": 11,
        "conditions": {
            "rainfall_mm": 350,
            "reservoir_pct": 100,
            "groundwater_m": 2,
            "spi": 2.8,
            "ndvi": 0.55,
            "soil_moisture_pct": 98
        }
    },
    "normal_monsoon": {
        "name": "Normal Monsoon Year",
        "description": "Average conditions during NE monsoon",
        "year": 2022,
        "month": 11,
        "conditions": {
            "rainfall_mm": 45,
            "reservoir_pct": 75,
            "groundwater_m": 6,
            "spi": 0.5,
            "ndvi": 0.58,
            "soil_moisture_pct": 65
        }
    },
    "pre_monsoon_stress": {
        "name": "Pre-Monsoon Water Stress",
        "description": "Typical June conditions before SW monsoon",
        "year": 2022,
        "month": 6,
        "conditions": {
            "rainfall_mm": 5,
            "reservoir_pct": 35,
            "groundwater_m": 15,
            "spi": -1.2,
            "ndvi": 0.28,
            "soil_moisture_pct": 25
        }
    }
}
