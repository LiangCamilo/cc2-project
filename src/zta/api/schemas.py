"""Request/response contracts. The OpenAPI document served at /openapi.json is generated from these."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Sliders(BaseModel):
    """Position of each acoustic slider in [0, 1] (0 = dataset minimum, 1 = dataset maximum).

    A slider left out means "no preference" and takes the dataset average.
    """

    model_config = ConfigDict(extra="forbid")
    danceability: float | None = Field(None, ge=0, le=1)
    energy: float | None = Field(None, ge=0, le=1)
    valence: float | None = Field(None, ge=0, le=1)
    acousticness: float | None = Field(None, ge=0, le=1)
    instrumentalness: float | None = Field(None, ge=0, le=1)
    liveness: float | None = Field(None, ge=0, le=1)
    loudness: float | None = Field(None, ge=0, le=1)
    tempo: float | None = Field(None, ge=0, le=1)
    speechiness: float | None = Field(None, ge=0, le=1)

    def provided(self) -> dict[str, float]:
        return self.model_dump(exclude_none=True)


class ProfileInput(BaseModel):
    """Describe the user's taste with sliders OR with liked songs (exactly one of them)."""

    model_config = ConfigDict(extra="forbid")
    sliders: Sliders | None = None
    seed_ids: list[str] | None = Field(None, min_length=1, max_length=50, description="track_id of liked songs")

    @model_validator(mode="after")
    def _exactly_one(self):
        if (self.sliders is None) == (self.seed_ids is None):
            raise ValueError("Provide exactly one of 'sliders' or 'seed_ids'.")
        return self


class ProfileRequest(ProfileInput):
    top_genres: int = Field(5, ge=1, le=20)


class RecommendRequest(ProfileInput):
    method: Literal["zta", "knn"] = Field("zta", description="zta = tolerance zones; knn = nearest songs")
    mode: Literal["familiar", "discovery", "exploration"] = Field("discovery", description="Only used by zta")
    metric: Literal["euclidean", "mahalanobis"] = Field("mahalanobis", description="Only used by zta")
    k: int = Field(10, ge=1, le=50)
    genres: list[str] | None = Field(None, description="Keep only songs with at least one of these genres")
    popularity_lambda: float | None = Field(None, ge=0, description="Only used by zta; 0 = no popularity penalty")
    dedupe: bool = Field(True, description="Collapse repeated releases of the same title and artist")


class FeatureValue(BaseModel):
    feature: str
    slider: float
    real_value: float


class GenreScore(BaseModel):
    genre: str
    score: float


class ProfileResponse(BaseModel):
    profile: list[FeatureValue]
    expected_popularity: float
    genres: dict[str, list[GenreScore]] = Field(description="Top genres per model (rf, knn, svm, linear)")


class Song(BaseModel):
    rank: int
    track_id: str
    track_name: str
    artists: str
    genres: list[str]
    popularity: float
    distance: float


class RecommendResponse(BaseModel):
    method: str
    mode: str | None
    count: int
    songs: list[Song]


class FoundSong(BaseModel):
    track_id: str
    track_name: str
    artists: str
    genres: list[str]
    popularity: float


class SearchResponse(BaseModel):
    count: int
    songs: list[FoundSong]


class FeatureMeta(BaseModel):
    name: str
    min: float
    max: float
    default_slider: float


class MetaResponse(BaseModel):
    features: list[FeatureMeta]
    genres: list[str]
    modes: list[str]
    metrics: list[str]
    models: list[str]


class HealthResponse(BaseModel):
    status: str
    songs: int
    genres: int
