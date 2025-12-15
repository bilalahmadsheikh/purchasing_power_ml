"""
FastAPI Production Application
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List
import pandas as pd
import logging
from datetime import datetime
import time

from .config import settings
from .schemas import (
    PredictionInput, PredictionOutput,
    BatchPredictionInput, BatchPredictionOutput,
    HealthCheck, ModelInfo, ErrorResponse, ComparisonRequest
)
from .predict import predict as predict_asset

# ============================================================================
# SETUP LOGGING
# ============================================================================

logging.basicConfig(
    level=settings.LOG_LEVEL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(settings.LOG_FILE),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# ============================================================================
# INITIALIZE APP
# ============================================================================

app = FastAPI(
    title=settings.API_TITLE,
    description=settings.API_DESCRIPTION,
    version=settings.API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Track uptime
START_TIME = time.time()

# ============================================================================
# EXCEPTION HANDLERS
# ============================================================================

@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=ErrorResponse(
            error="ValueError",
            detail=str(exc)
        ).dict()
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="InternalServerError",
            detail="An unexpected error occurred"
        ).dict()
    )

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/", response_model=HealthCheck)
def health_check():
    """Health check endpoint"""
    uptime = time.time() - START_TIME
    return HealthCheck(
        status="healthy",
        service=settings.API_TITLE,
        version=settings.API_VERSION,
        model=f"LightGBM (Macro-F1: {settings.MODEL_MACRO_F1:.4f})",
        uptime_seconds=round(uptime, 2)
    )

@app.post("/predict", response_model=PredictionOutput, status_code=status.HTTP_200_OK)
def predict_endpoint(input_data: PredictionInput):
    """
    Single asset prediction
    
    Returns classification, confidence, entry signal, and actionable insights
    """
    try:
        logger.info(f"Prediction request for {input_data.asset} ({input_data.horizon_years}Y horizon)")
        result = predict_asset(input_data.asset, input_data.horizon_years)
        logger.info(f"Prediction successful: {result.predicted_class}")
        return result
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/batch", response_model=BatchPredictionOutput)
def predict_batch_endpoint(input_data: BatchPredictionInput):
    """Batch prediction for multiple assets"""
    try:
        logger.info(f"Batch prediction request for {len(input_data.assets)} assets ({input_data.horizon_years}Y horizon)")
        
        predictions = []
        for asset in input_data.assets:
            try:
                result = predict_asset(asset, input_data.horizon_years)
                predictions.append(result)
            except Exception as e:
                logger.error(f"Error predicting {asset}: {e}")
                # Continue with other assets
                continue
        
        logger.info(f"Batch prediction complete: {len(predictions)} successful")
        
        return BatchPredictionOutput(
            predictions=predictions,
            count=len(predictions)
        )
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/upload", response_model=BatchPredictionOutput)
async def predict_upload_endpoint(file: UploadFile = File(...)):
    """Upload CSV file for batch prediction"""
    try:
        # Validate file type
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="File must be CSV format")
        
        # Read CSV
        df = pd.read_csv(file.file)
        
        if 'Asset' not in df.columns:
            raise HTTPException(status_code=400, detail="CSV must contain 'Asset' column")
        
        assets = df['Asset'].unique().tolist()
        
        logger.info(f"File upload prediction: {len(assets)} unique assets")
        
        # Use batch prediction
        return predict_batch_endpoint(BatchPredictionInput(assets=assets))
        
    except Exception as e:
        logger.error(f"File upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/assets")
def list_assets():
    """List all available assets"""
    return {
        "assets": settings.AVAILABLE_ASSETS,
        "count": len(settings.AVAILABLE_ASSETS),
        "categories": {
            "crypto": ["Bitcoin", "Ethereum", "Litecoin"],
            "metals": ["Gold", "Silver"],
            "indices": ["SP500", "NASDAQ", "DowJones"],
            "commodities": ["Oil"],
            "etfs": ["Gold_ETF", "TreasuryBond_ETF", "RealEstate_ETF"],
            "stocks": ["Apple", "Microsoft", "JPMorgan"]
        }
    }

@app.get("/model/info", response_model=ModelInfo)
def model_info_endpoint():
    """Get model information"""
    from .predict import model_manager
    
    encoder = model_manager.get_encoder()
    classes = encoder.classes_.tolist() if encoder else ["A_PRESERVER", "B_PARTIAL", "C_ERODER", "D_DESTROYER"]
    
    return ModelInfo(
        model_type="LightGBM Gradient Boosting",
        num_features=len(model_manager.get_features()),
        classes=classes,
        performance={
            "macro_f1": settings.MODEL_MACRO_F1,
            "accuracy": settings.MODEL_ACCURACY,
            "balanced_accuracy": 0.9480
        },
        training_date="2024-12-15",
        best_iteration=settings.MODEL_BEST_ITERATION
    )

@app.post("/compare")
def compare_assets(request: ComparisonRequest):
    """Compare multiple assets for the same investment horizon"""
    from .predict import predict
    
    if len(request.assets) < 2:
        raise HTTPException(
            status_code=400,
            detail="Comparison requires at least 2 assets"
        )
    if len(request.assets) > 10:
        raise HTTPException(
            status_code=400,
            detail="Maximum 10 assets per comparison"
        )
    
    try:
        comparisons = []
        for asset in request.assets:
            prediction = predict(asset, request.horizon_years)
            # Use final_composite_score if available, otherwise use confidence as fallback
            score = prediction.component_scores.final_composite_score if prediction.component_scores else prediction.confidence
            comparisons.append({
                "asset": asset,
                "classification": str(prediction.predicted_class.value),
                "confidence": prediction.confidence,
                "score": score,
                "component_scores": prediction.component_scores
            })
        
        # Rank by score
        ranked = sorted(comparisons, key=lambda x: x["score"], reverse=True)
        
        return {
            "horizon_years": request.horizon_years,
            "comparison_count": len(comparisons),
            "ranked_results": ranked,
            "best_asset": ranked[0]["asset"],
            "best_score": ranked[0]["score"]
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Asset error: {str(e)}")
    except Exception as e:
        logger.error(f"Comparison error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/asset/historical/{asset}")
def get_historical_data(
    asset: str,
    horizon_years: float = 1.0,
    limit: int = 100
):
    """Get historical performance data for an asset"""
    from .predict import model_manager
    import json
    from datetime import datetime
    
    if limit < 10 or limit > 1000:
        raise HTTPException(
            status_code=400,
            detail="Limit must be between 10 and 1000"
        )
    
    try:
        test_data = model_manager.get_data()
        
        # Handle missing test data (CI environment)
        if test_data is None:
            return {
                "asset": asset,
                "horizon_years": horizon_years,
                "records_count": 0,
                "date_range": {"start": "N/A", "end": "N/A"},
                "price_stats": {"min": 0, "max": 0, "mean": 0, "std": 0},
                "sample_records": [],
                "note": "Test data not available"
            }
        
        # Filter for asset and get last N records
        asset_data = test_data[test_data['Asset'] == asset].tail(limit)
        
        if asset_data.empty:
            raise ValueError(f"No historical data found for {asset}")
        
        # Get date range (handle missing dates)
        date_start = "unknown"
        date_end = "unknown"
        
        # Prepare response with key metrics
        response = {
            "asset": asset,
            "horizon_years": horizon_years,
            "records_count": len(asset_data),
            "date_range": {
                "start": date_start,
                "end": date_end
            },
            "price_stats": {
                "min": float(asset_data.get('Close', [0]).min() if 'Close' in asset_data else 0),
                "max": float(asset_data.get('Close', [0]).max() if 'Close' in asset_data else 0),
                "mean": float(asset_data.get('Close', [0]).mean() if 'Close' in asset_data else 0),
                "std": float(asset_data.get('Close', [0]).std() if 'Close' in asset_data else 0)
            },
            "sample_records": asset_data.head(10).to_dict('records')
        }
        
        return response
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Historical data error: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving historical data")

@app.get("/data/quality/{asset}")
def check_data_quality(asset: str):
    """Check data quality metrics for an asset"""
    from .predict import model_manager
    
    try:
        test_data = model_manager.get_data()
        
        # Handle missing test data (CI environment)
        if test_data is None:
            return {
                "asset": asset,
                "total_records": 0,
                "numeric_columns": 0,
                "quality_score": 100.0,
                "missing_values": {},
                "status": "UNAVAILABLE",
                "note": "Test data not available"
            }
        
        asset_data = test_data[test_data['Asset'] == asset]
        
        if asset_data.empty:
            raise ValueError(f"No data found for {asset}")
        
        # Calculate quality metrics
        total_records = len(asset_data)
        numeric_cols = asset_data.select_dtypes(include=['number']).columns
        
        missing_pct = (asset_data.isnull().sum() / len(asset_data) * 100).to_dict()
        
        return {
            "asset": asset,
            "total_records": total_records,
            "numeric_columns": len(numeric_cols),
            "quality_score": 100 - (asset_data.isnull().sum().sum() / (len(asset_data) * len(asset_data.columns)) * 100),
            "missing_values": {k: v for k, v in missing_pct.items() if v > 0},
            "status": "GOOD" if asset_data.isnull().sum().sum() < len(asset_data) * 0.05 else "ACCEPTABLE"
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Data quality check error: {str(e)}")
        raise HTTPException(status_code=500, detail="Error checking data quality")

# ============================================================================
# STARTUP/SHUTDOWN EVENTS
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Run on application startup"""
    logger.info("="*80)
    logger.info(f"Starting {settings.API_TITLE} v{settings.API_VERSION}")
    logger.info("="*80)
    logger.info("Loading models...")
    
    # Models are loaded via ModelManager singleton
    from .predict import model_manager
    logger.info(" All models loaded successfully")
    
    logger.info(f"API ready on http://{settings.HOST}:{settings.PORT}")
    logger.info("="*80)

@app.on_event("shutdown")
async def shutdown_event():
    """Run on application shutdown"""
    logger.info("Shutting down API...")
    logger.info("Shutdown complete")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.RELOAD,
        workers=settings.WORKERS
    )