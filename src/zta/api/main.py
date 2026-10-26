"""FastAPI service.

  python -m zta.api                    # http://127.0.0.1:8000  (interactive docs at /docs)

The recommender (catalog, partitions, the eight .joblib models) is loaded once at startup.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
import os

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from zta.config import Settings
from zta.data import CatalogContractError, CsvCatalogSource, load_partitions
from zta.domain import ListeningMode
from zta.models.app import MusicRecommender
from zta.models.registry import MODEL_NAMES

from .schemas import (
    FeatureMeta, FeatureValue, FoundSong, GenreScore, HealthResponse, MetaResponse, ProfileInput,
    ProfileRequest, ProfileResponse, RecommendRequest, RecommendResponse, SearchResponse, Song,
)


def load_default_recommender() -> MusicRecommender:
    settings = Settings()
    catalog = CsvCatalogSource(settings=settings).load()
    return MusicRecommender(catalog, load_partitions(catalog, settings=settings), settings)


def _profile(rec: MusicRecommender, data: ProfileInput):
    if data.seed_ids is not None:
        return rec.profile_from_songs(data.seed_ids)
    return rec.profile_from_sliders(**data.sliders.provided())


def _songs(table) -> list[Song]:
    return [
        Song(rank=int(r.rank), track_id=str(r.track_id), track_name=str(r.track_name), artists=str(r.artists),
             genres=list(r.genres), popularity=float(r.popularity), distance=float(r.distance))
        for r in table.itertuples(index=False)
    ]


def create_app(recommender: MusicRecommender | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.recommender = recommender or load_default_recommender()
        yield

    app = FastAPI(
        title="ZTA music recommender", version="1.0.0", lifespan=lifespan,
        description="Profile from sliders or liked songs -> fitting genres (4 trained models) -> recommended songs (ZTA / KNN).",
    )
    origins = [o.strip() for o in os.environ.get("ZTA_CORS_ORIGINS", "").split(",") if o.strip()]
    if origins:
        app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["*"], allow_headers=["*"])

    @app.exception_handler(ValueError)
    async def _bad_value(_: Request, error: ValueError):
        return JSONResponse(status_code=422, content={"detail": str(error)})

    @app.exception_handler(CatalogContractError)
    async def _bad_contract(_: Request, error: CatalogContractError):
        return JSONResponse(status_code=422, content={"detail": str(error)})

    def rec_of(request: Request) -> MusicRecommender:
        return request.app.state.recommender

    @app.get("/health", response_model=HealthResponse, tags=["system"])
    def health(request: Request):
        rec = rec_of(request)
        return HealthResponse(status="ok", songs=len(rec.catalog), genres=len(rec.genres))

    @app.get("/meta", response_model=MetaResponse, tags=["system"], summary="Everything the front end needs to draw its controls")
    def meta(request: Request):
        rec = rec_of(request)
        features = [FeatureMeta(name=r.feature, min=r.min, max=r.max, default_slider=r.default_slider)
                    for r in rec.feature_info().itertuples(index=False)]
        return MetaResponse(features=features, genres=rec.genres, modes=[m.value for m in ListeningMode],
                            metrics=["euclidean", "mahalanobis"], models=list(MODEL_NAMES))

    @app.get("/songs/search", response_model=SearchResponse, tags=["songs"], summary="Find songs by title or artist")
    def search(request: Request, q: str = Query(..., min_length=2), limit: int = Query(10, ge=1, le=50)):
        found = rec_of(request).search_songs(q, limit)
        return SearchResponse(count=len(found), songs=[
            FoundSong(track_id=str(r.track_id), track_name=str(r.track_name), artists=str(r.artists),
                      genres=list(r.genres), popularity=float(r.popularity))
            for r in found.itertuples(index=False)
        ])

    @app.post("/profile", response_model=ProfileResponse, tags=["function 1: profile"],
              summary="Build a profile and get the genres that fit it")
    def profile(request: Request, body: ProfileRequest):
        rec = rec_of(request)
        prof = _profile(rec, body)
        described = rec.describe_profile(prof)
        genres = rec.suggest_genres(prof, top=body.top_genres)
        return ProfileResponse(
            profile=[FeatureValue(feature=name, slider=float(r.slider_0_1), real_value=float(r.real_value))
                     for name, r in described.iterrows()],
            expected_popularity=rec.expected_popularity(prof),
            genres={model: [GenreScore(genre=r.genre, score=float(r.score)) for r in rows.itertuples()]
                    for model, rows in genres.groupby("model", sort=False)},
        )

    @app.post("/recommend", response_model=RecommendResponse, tags=["function 2: recommend"],
              summary="Recommend songs for a profile")
    def recommend(request: Request, body: RecommendRequest):
        rec = rec_of(request)
        prof = _profile(rec, body)
        if body.method == "knn":
            table = rec.nearest_songs(prof, body.k, body.genres, body.dedupe)
            return RecommendResponse(method="knn", mode=None, count=len(table), songs=_songs(table))
        table = rec.recommend(prof, body.mode, body.k, body.metric, body.genres, body.popularity_lambda, body.dedupe)
        return RecommendResponse(method="zta", mode=body.mode, count=len(table), songs=_songs(table))

    return app


app = create_app()
