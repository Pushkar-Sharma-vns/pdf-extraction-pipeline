from typing import List, Optional, Literal

from pydantic import BaseModel, Field


class LinkedAssetSchema(BaseModel):
    image_url: Optional[str] = Field(None, description="URL or filename string of the image if available.")
    asset_type: Literal["chart", "map", "table_image", "infographic"]
    caption: Optional[str] = Field(None, description="Original caption or title text of the visual element.")
    summary: str = Field(..., description="A clear summary of what data or trend this visual conveys.")


class ContextualRetrievalPipelineSchema(BaseModel):
    contextual_situation: str = Field(
        ...,
        description=(
            "A precise 2-sentence context anchoring this specific chunk text to the broader report. "
            "It must explicitly state the timeline, geographical scope, and primary topic covered "
            "by the parent report so this chunk makes perfect sense in complete isolation. "
            "CRITICAL: Rely ONLY on the provided text. Do not introduce outside knowledge or extrapolate."
        )
    )

    source_publisher: Literal[
        "JLL", "Knight_Frank", "Anarock", "Savills", "CBRE", "Colliers",
        "PropEquity", "ICRA", "CRISIL", "Cushman_Wakefield", "Liases_Foras", "Other"
    ] = Field(..., description="Who published the parent report?")

    report_title: str = Field(..., description="The complete, exact title of the source report.")

    report_type: Literal[
        "quarterly", "half_yearly", "annual", "thematic", "outlook", "sector_deepdive", "flash_note"
    ] = Field(..., description="Genre classification based on the parent report's scope.")

    publish_date_or_year: str = Field(
        ...,
        description="ISO 8601 date (YYYY-MM-DD) or fallback Year (YYYY) when the report was released."
    )

    time_horizon: Literal["historical", "current", "forecast"] = Field(
        ...,
        description="Temporal frame discussed inside this chunk. Note: 2025/2026 dates are treated as current."
    )

    forecast_years_upto: Optional[int] = Field(
        None,
        description="Target year of projection. Set strictly ONLY when time_horizon is 'forecast', otherwise return null."
    )

    zone: Literal["North", "South", "East", "West", "Central", "Peripheral", "PAN_City"] = Field(
        ..., description="Coarse city sub-zone covered by this text snippet."
    )

    city: Literal[
        "Bangalore", "Mumbai", "Delhi_NCR", "Pune", "Hyderabad", "Chennai",
        "Kolkata", "Ahmedabad", "Tier2", "PAN_India"
    ] = Field(..., description="Primary city scale. Maps Bengaluru/Bangalore consistently to 'Bangalore'.")

    micro_market: Optional[str] = Field(
        None,
        description="Lowest micro-market name mentioned (e.g., Whitefield, Hebbal, ORR_Sarjapur). Return null if not stated."
    )

    corridor_tags: List[Literal[
        "ORR", "Sarjapur_Belt", "Airport_Corridor", "Yellow_Line", "NICE_Road", "Hosur_Road", "Tumkur_Road"
    ]] = Field(default_factory=list, description="Infrastructure corridors explicitly referenced. Return empty list if none.")

    asset_class: Literal[
        "residential", "office", "retail", "industrial_warehousing", "hospitality", "data_center", "alt_assets"
    ] = Field(..., description="Top-level real estate industry segment.")

    sub_asset_class: Optional[Literal[
        "Resi: luxury", "Resi: premium", "Resi: mid", "Resi: affordable",
        "Office: GradeA_plus", "Office: GradeA", "Office: GradeB",
        "Retail: mall", "Retail: high_street", "Retail: standalone",
        "Warehousing: GradeA", "Warehousing: GradeB"
    ]] = Field(None, description="Detailed sub-tier breakdown. Match strictly to definitions or return null.")

    economic_lens: List[Literal[
        "supply", "absorption", "net_absorption", "launches", "unsold_inventory", "price_psf",
        "rental_psf", "yield", "vacancy", "cap_rate", "capital_value", "quarters_to_sell", "new_completions"
    ]] = Field(default_factory=list, description="All real estate KPIs actively discussed here. Return empty list if none.")

    content_intent: Literal[
        "informational", "analytical", "methodological", "evaluative", "predictive", "prescriptive", "regulatory"
    ] = Field(..., description="Core underlying purpose of this snippet's content.")

    content_certainty: Literal["asserted", "estimated", "probabilistic", "speculative"] = Field(
        ..., description="Reliability marker for the data discussed."
    )

    stakeholder_lens_discussed: List[Literal[
        "investor", "occupier_corp", "developer", "end_user_buyer", "policymaker", "broker"
    ]] = Field(default_factory=list, description="Target industry perspectives taken. Return empty list if none.")

    macro_event_anchors: List[Literal[
        "repo_rate", "GST", "RERA_amendment", "GCC_inflow", "IT_layoffs", "elections", "budget",
        "demonetisation_era", "COVID_era", "infra_announcement"
    ]] = Field(default_factory=list, description="External macro milestones mentioned. Return empty list if none.")

    methodology_basis: Literal[
        "primary_survey", "transaction_data", "listing_data", "RERA_filings", "proprietary_model", "secondary_research", "mixed"
    ] = Field(..., description="Underlying mechanism of how numbers were reached.")

    data_sources: List[Literal[
        "RBI", "RERA", "NHB", "MoSPI", "CREDAI", "Publisher_Survey", "Publisher_Model", "Broker_Network", "Government_Gazette", "Other"
    ]] = Field(default_factory=list, description="Feeder data entities mentioned. Return empty list if none.")

    comparison_axes: List[Literal["city", "segment", "time", "asset", "developer"]] = Field(
        default_factory=list, description="Dimensions across which any trends are evaluated. Return empty list if none."
    )

    linked_assets: List[LinkedAssetSchema] = Field(
        default_factory=list,
        description="Visual entities (charts, maps, data images) accompanying this text block. Return empty list if none."
    )
