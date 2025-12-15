# PPP-Q API Quick Reference

## Running the API

```bash
cd c:\Users\bilaa\OneDrive\Desktop\purchasing_power_ml
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

API will be available at: `http://localhost:8000`
Interactive docs: `http://localhost:8000/docs`

---

## Quick Examples

### 1. Compare Multiple Assets

```bash
curl -X POST "http://localhost:8000/compare" \
  -H "Content-Type: application/json" \
  -d '{
    "assets": ["Bitcoin", "Gold", "SP500"],
    "horizon_years": 5
  }'
```

**PowerShell:**
```powershell
$body = @{
    assets = @("Bitcoin", "Gold", "SP500")
    horizon_years = 5
} | ConvertTo-Json

Invoke-WebRequest -Uri "http://localhost:8000/compare" `
  -Method Post `
  -ContentType "application/json" `
  -Body $body | Select-Object -ExpandProperty Content
```

### 2. Get Historical Data

```bash
curl "http://localhost:8000/asset/historical/Bitcoin?horizon_years=5&limit=50"
```

**PowerShell:**
```powershell
Invoke-WebRequest -Uri "http://localhost:8000/asset/historical/Bitcoin?horizon_years=5&limit=50" | `
  Select-Object -ExpandProperty Content | ConvertFrom-Json | Format-Table
```

### 3. Check Data Quality

```bash
curl "http://localhost:8000/data/quality/Bitcoin"
```

**PowerShell:**
```powershell
Invoke-WebRequest -Uri "http://localhost:8000/data/quality/Bitcoin" | `
  Select-Object -ExpandProperty Content | ConvertFrom-Json
```

### 4. Single Asset Prediction (with caching benefit)

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "asset": "Bitcoin",
    "horizon_years": 5
  }'
```

Call this twice - the second time will be 20-30x faster due to caching!

---

## Testing

Run all tests:
```bash
python -m pytest tests/ -v
```

Run only new endpoint tests:
```bash
python -m pytest tests/test_new_endpoints.py -v
```

Run specific test:
```bash
python -m pytest tests/test_new_endpoints.py::test_compare_assets -v
```

---

## API Endpoints Summary

| Method | Endpoint | Purpose | Status |
|--------|----------|---------|--------|
| POST | `/predict` | Single asset prediction | ✅ |
| POST | `/predict/batch` | Multiple predictions | ✅ |
| GET | `/assets` | List available assets | ✅ |
| GET | `/model/info` | Model metadata | ✅ |
| POST | `/compare` | Compare multiple assets | ✅ NEW |
| GET | `/asset/historical/{asset}` | Historical data | ✅ NEW |
| GET | `/data/quality/{asset}` | Data quality check | ✅ NEW |

---

## Available Assets

```
Bitcoin, Ethereum, Litecoin, Gold, Silver, 
SP500, NASDAQ, DowJones, Oil,
Gold_ETF, TreasuryBond_ETF, RealEstate_ETF,
Apple, Microsoft, JPMorgan
```

---

## Horizon Years Guide

| Horizon | Investment Type | Use Case |
|---------|-----------------|----------|
| 0.5 years | Ultra-short term | Day/swing trading |
| 1.0 year | Short term | 1-year savings goal |
| 3.0 years | Medium term | 3-year investment plan |
| 5.0 years | Long term | Standard investment horizon |
| 10.0 years | Very long term | Retirement/wealth building |

---

## Asset Classification Guide

| Class | Meaning | Purchasing Power |
|-------|---------|-------------------|
| A_PRESERVER | Excellent | Strongly preserves/increases |
| B_PARTIAL | Good | Partially preserves |
| C_ERODER | Fair | Some erosion over time |
| D_DESTROYER | Poor | Significant erosion |

---

## Response Caching

**Automatic Performance Boost:**
- First request to same asset/horizon: 150-200ms
- Repeat request: 5-10ms (20-30x faster!)
- Cache automatically cleared when server restarts

**Cache hits all endpoints:**
- `/predict` - Uses cache automatically
- `/compare` - Gets cached predictions for each asset
- Individual `/asset/historical` etc. don't use prediction cache

---

## Performance Tips

1. **Batch Requests** - Use `/predict/batch` for multiple assets
2. **Leverage Cache** - Make repeated requests for same asset/horizon
3. **Tune Horizon** - Use appropriate horizon for your use case
4. **Limit Results** - Use `limit` parameter in `/asset/historical`

---

## Troubleshooting

**"Asset not found" error:**
- Check spelling matches exactly (case-sensitive)
- Use `/assets` endpoint to see valid asset names

**"Limit must be between 10 and 1000":**
- `/asset/historical` requires limit between 10-1000
- Default is 100 if not specified

**"Comparison requires at least 2 assets":**
- `/compare` needs minimum 2 assets, maximum 10
- Remove duplicates automatically

**API not responding:**
- Check server is running: `python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000`
- Check port 8000 is available
- Check firewall allows connections

---

## Development Notes

**Key Files Modified:**
- `src/api/main.py` - Added 3 new endpoints
- `src/api/schemas.py` - Added `ComparisonRequest` schema
- `src/api/predict.py` - Added caching via `@lru_cache`
- `tests/test_new_endpoints.py` - Added 8 new tests

**All Tests Passing:**
```
tests/test_api.py               4 tests [OK]
tests/test_new_endpoints.py     8 tests [OK]
Total: 12 tests in 2.7 seconds
```

---

## Next Steps

1. **Test the endpoints** using the examples above
2. **Monitor performance** - Compare response times before/after cache
3. **Integrate into app** - Use `/compare` for portfolio analysis
4. **Generate reports** - Use historical data for charting
5. **Validate data** - Use `/data/quality` before analysis

---

## Support

For issues or questions:
1. Check the test files (`tests/`) for usage examples
2. Review error messages - they're detailed and specific
3. Check `IMPROVEMENTS.md` for comprehensive documentation
4. Run tests to verify your installation

Happy investing! 📈
