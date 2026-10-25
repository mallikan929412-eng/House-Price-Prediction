"""
Pydantic Schemas for NammaMane API
----------------------------------
Validates requests and responses, enforces boundaries, and gives clear 422 errors.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, field_validator

VALID_LOCALITIES = [
    "Banashankari", "Bannerghatta Road", "Basavanagudi", "Bellandur",
    "Devanahalli", "Electronic City", "HSR Layout", "Hebbal",
    "Hennur", "Hosur Road", "Indiranagar", "JP Nagar",
    "Jayanagar", "KR Puram", "Kanakapura Road", "Kengeri",
    "Koramangala", "Malleshwaram", "Marathahalli", "RT Nagar",
    "Rajajinagar", "Sarjapur Road", "Vijayanagar", "Whitefield", "Yelahanka"
]

LOCALITY_ZONE_MAP = {
    "Banashankari": "South", "Bannerghatta Road": "South", "Basavanagudi": "South",
    "Bellandur": "East", "Devanahalli": "North", "Electronic City": "South",
    "HSR Layout": "South", "Hebbal": "North", "Hennur": "North",
    "Hosur Road": "South", "Indiranagar": "East", "JP Nagar": "South",
    "Jayanagar": "South", "KR Puram": "East", "Kanakapura Road": "South",
    "Kengeri": "West", "Koramangala": "South", "Malleshwaram": "North",
    "Marathahalli": "East", "RT Nagar": "North", "Rajajinagar": "West",
    "Sarjapur Road": "East", "Vijayanagar": "West", "Whitefield": "East",
    "Yelahanka": "North"
}

class HouseInput(BaseModel):
    locality: str = Field(..., description="Bengaluru locality name")
    zone: Optional[str] = Field(None, description="Zone e.g. South, North, East, West")
    property_type: str = Field("Apartment", description="Apartment, Independent House, or Villa")
    bhk: int = Field(2, ge=1, le=10, description="Bedrooms count (1-10)")
    area_sqft: float = Field(..., gt=100, lt=20000, description="Built-up / super-built-up area in sq.ft")
    bathrooms: Optional[int] = Field(2, ge=1, le=10)
    balconies: Optional[float] = Field(1.0, ge=0.0, le=8.0)
    floor: Optional[int] = Field(3, ge=0, le=60)
    total_floors: Optional[int] = Field(12, ge=1, le=60)
    age_years: Optional[int] = Field(5, ge=0, le=80)
    furnishing: Optional[str] = Field("Semi-Furnished")
    parking_slots: Optional[int] = Field(1, ge=0, le=10)
    lift_available: Optional[str] = Field("Yes")
    gated_community: Optional[str] = Field("Yes")
    amenities_score: Optional[float] = Field(7.0, ge=0.0, le=10.0)
    metro_distance_km: Optional[float] = Field(2.5, ge=0.0, le=40.0)
    school_distance_km: Optional[float] = Field(1.2, ge=0.0, le=20.0)
    tech_park_distance_km: Optional[float] = Field(4.0, ge=0.0, le=40.0)
    facing: Optional[str] = Field("East")
    rera_approved: Optional[str] = Field("Yes")

    @field_validator("locality")
    @classmethod
    def validate_locality(cls, v: str) -> str:
        trimmed = v.strip()
        matched = [loc for loc in VALID_LOCALITIES if loc.lower() == trimmed.lower()]
        if not matched:
            raise ValueError(
                f"Invalid locality '{v}'. Must be one of the 25 Bengaluru localities: {', '.join(VALID_LOCALITIES)}"
            )
        return matched[0]

    @field_validator("property_type")
    @classmethod
    def validate_property_type(cls, v: str) -> str:
        valid_types = ["Apartment", "Independent House", "Villa"]
        matched = [pt for pt in valid_types if pt.lower() == v.strip().lower()]
        if not matched:
            raise ValueError(f"Property type must be one of: {', '.join(valid_types)}")
        return matched[0]

    def to_df_dict(self) -> Dict[str, Any]:
        """Converts to a dictionary suitable for DataFrame input with inferred zone."""
        resolved_zone = self.zone or LOCALITY_ZONE_MAP.get(self.locality, "South")
        return {
            "locality": self.locality,
            "zone": resolved_zone,
            "property_type": self.property_type,
            "bhk": self.bhk,
            "area_sqft": self.area_sqft,
            "bathrooms": self.bathrooms,
            "balconies": self.balconies,
            "floor": self.floor,
            "total_floors": self.total_floors,
            "age_years": self.age_years,
            "furnishing": self.furnishing,
            "parking_slots": self.parking_slots,
            "lift_available": self.lift_available,
            "gated_community": self.gated_community,
            "amenities_score": self.amenities_score,
            "metro_distance_km": self.metro_distance_km,
            "school_distance_km": self.school_distance_km,
            "tech_park_distance_km": self.tech_park_distance_km,
            "facing": self.facing,
            "rera_approved": self.rera_approved
        }

class WhyThisPriceItem(BaseModel):
    factor: str
    impact: str  # e.g. "+Rs 18.4L", "-Rs 6.2L"
    direction: str  # "positive", "negative", "neutral"
    description: str

class PredictionResponse(BaseModel):
    predicted_price_lakhs: float
    confidence_range_lakhs: float
    price_range_low_lakhs: float
    price_range_high_lakhs: float
    price_per_sqft: float
    model_used: str
    why_this_price: List[WhyThisPriceItem]
    is_synthetic_data: bool = True

class WhatIfScenario(BaseModel):
    label: str
    changes: Dict[str, Any]

class WhatIfRequest(BaseModel):
    base_house: HouseInput
    scenarios: List[WhatIfScenario]

class WhatIfResult(BaseModel):
    label: str
    predicted_price_lakhs: float
    difference_lakhs: float
    percentage_change: float
    changes_applied: Dict[str, Any]

class WhatIfResponse(BaseModel):
    base_price_lakhs: float
    results: List[WhatIfResult]
    is_synthetic_data: bool = True

class CompareRequest(BaseModel):
    house_a: HouseInput
    house_b: HouseInput
    label_a: Optional[str] = "Property 1"
    label_b: Optional[str] = "Property 2"

class CompareResponse(BaseModel):
    property_a: Dict[str, Any]
    property_b: Dict[str, Any]
    price_difference_lakhs: float
    cheaper_option: str
    winner_banner: str
    category_winners: Dict[str, str]
    is_synthetic_data: bool = True

class DealCheckRequest(BaseModel):
    house: HouseInput
    asking_price_lakhs: float = Field(..., gt=0.1, lt=50000.0, description="Asking price in lakhs")

class DealCheckResponse(BaseModel):
    asking_price_lakhs: float
    predicted_price_lakhs: float
    difference_lakhs: float
    difference_percentage: float
    status: str  # "Bargain", "Fair", or "Overpriced"
    badge_color: str
    gauge_percentage: float  # 0 to 100 for gauge meter
    explanation: str
    is_synthetic_data: bool = True
