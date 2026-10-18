"""Main FastAPI application for Customer Analytics Platform."""

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import time

from src.api.routes import churn, segment, recommend, forecast
from src.api.schemas import HealthCheckResponse
from src.models.registry import ModelRegistry
from src.utils.logger import get_logger

logger = get_logger("api.main")

app = FastAPI(
    title="Customer Analytics Platform API 📊🚀",
    description=(
        "Enterprise-grade Machine Learning API providing real-time Churn Risk scoring, "
        "SHAP explainability, Prescriptive Retention Playbooks, Behavioral Segmentation, "
        "Hybrid Product Recommendations, and Forward Demand & Revenue Forecasting."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for web applications and Streamlit dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Middleware to measure and log API latency."""
    start_time = time.time()
    response = await call_next(request)
    process_time = (time.time() - start_time) * 1000
    response.headers["X-Process-Time-Ms"] = f"{process_time:.2f}"
    logger.info(f"{request.method} {request.url.path} completed in {process_time:.2f}ms with status {response.status_code}")
    return response


# Global Exception Handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception at {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An internal server error occurred while processing the request."}
    )


# Include Feature Routers
app.include_router(churn.router)
app.include_router(segment.router)
app.include_router(recommend.router)
app.include_router(forecast.router)


@app.get("/", summary="API Root Status")
def root_endpoint():
    """Root endpoint welcoming clients and linking to Swagger UI."""
    return {
        "service": "Customer Analytics Platform API",
        "version": "1.0.0",
        "status": "online",
        "documentation": "/docs",
        "endpoints": {
            "churn_prediction": "/churn/predict",
            "batch_churn": "/churn/batch",
            "segmentation": "/segment/predict",
            "recommendations": "/recommend/{customer_id}",
            "demand_forecast": "/forecast/revenue",
            "health_check": "/health"
        }
    }


@app.get("/health", response_model=HealthCheckResponse, summary="System Health Check")
def health_check():
    """Checks model loading status and platform health."""
    loaded_models = []
    try:
        ModelRegistry.load_churn_model()
        loaded_models.append("Churn Classifier (Calibrated)")
    except Exception:
        pass

    try:
        ModelRegistry.load_segmentation_model()
        loaded_models.append("K-Means Segmentation")
    except Exception:
        pass

    try:
        ModelRegistry.load_recommender()
        loaded_models.append("Hybrid Recommender")
    except Exception:
        pass

    try:
        ModelRegistry.load_forecast_models()
        loaded_models.append("Demand & Revenue Forecaster")
    except Exception:
        pass

    return HealthCheckResponse(
        status="healthy" if len(loaded_models) == 4 else "degraded",
        environment="production",
        version="1.0.0",
        models_loaded=loaded_models
    )
