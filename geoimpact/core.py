"""Validated event contracts and transparent scenario calculations (SPEC-02/04)."""
import math
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field

LABELS = ['natural_hazard', 'conflict', 'public_health', 'infrastructure', 'trade_policy', 'other']

class Location(BaseModel):
    model_config = ConfigDict(extra='forbid')
    name: str = Field(min_length=1, max_length=150)
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)

class Event(BaseModel):
    model_config = ConfigDict(extra='forbid')
    title: str = Field(min_length=1, max_length=200)
    event_type: Literal['natural_hazard', 'conflict', 'public_health', 'infrastructure', 'trade_policy', 'other']
    location_name: str | None = None
    location_quote: str | None = None
    evidence_quote: str = Field(min_length=15, max_length=1500)
    reported_scope: str = Field(max_length=800)
    possible_impacts: list[str] = Field(max_length=5)

class Extraction(BaseModel):
    model_config = ConfigDict(extra='forbid')
    events: list[Event] = Field(max_length=8)


def validate_evidence(result: Extraction, source: str) -> Extraction:
    normalized = ' '.join(source.split())
    for event in result.events:
        for quote in [event.evidence_quote, event.location_quote]:
            if quote is not None and (not quote.strip() or ' '.join(quote.split()) not in normalized):
                raise ValueError('The model returned a quotation not present in the source. Review the document.')
        if event.location_name and not event.location_quote:
            raise ValueError('Event location requires a supporting quotation.')
    return result


def distance_km(a: Location, b: Location) -> float:
    lat1, lat2 = math.radians(a.latitude), math.radians(b.latitude)
    dlat, dlon = lat2-lat1, math.radians(b.longitude-a.longitude)
    h = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    return 6371.0088 * 2 * math.asin(math.sqrt(min(1., max(0., h))))


def scenario(event: Location, user: Location, radius: float, dependency: str) -> dict:
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError('Scenario radius must be positive and finite.')
    distance = distance_km(event, user)
    inside = distance <= radius
    return {
        'distance_km': round(distance, 1),
        'inside_assumed_radius': inside,
        'direct_exposure': 'Inside the assumed footprint; verify actual local conditions.' if inside else 'Outside the assumed footprint; this does not establish safety.',
        'indirect_exposure': f'Possible disruption through your declared {dependency} dependency; verify suppliers and routes.' if dependency != 'None / unknown' else 'Unknown: business dependency information is not available.',
        'basis': 'User-defined sensitivity scenario, not a probability or a validated hazard forecast.'
    }


def circle(location: Location, radius_km: float) -> tuple[list, list]:
    lat1, lon1, angular = math.radians(location.latitude), math.radians(location.longitude), radius_km/6371.0088
    lats, lons = [], []
    for bearing in range(0, 361, 5):
        b = math.radians(bearing)
        lat2 = math.asin(math.sin(lat1)*math.cos(angular)+math.cos(lat1)*math.sin(angular)*math.cos(b))
        lon2 = lon1 + math.atan2(math.sin(b)*math.sin(angular)*math.cos(lat1), math.cos(angular)-math.sin(lat1)*math.sin(lat2))
        lon = (math.degrees(lon2)+180)%360-180
        if lons and lons[-1] is not None and abs(lon-lons[-1])>180:
            lats.append(None); lons.append(None)
        lats.append(math.degrees(lat2)); lons.append(lon)
    return lats, lons
