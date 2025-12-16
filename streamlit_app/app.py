"""
PPP-Q Investment Intelligence Dashboard
======================================
A comprehensive Streamlit dashboard for the Purchasing Power Preservation Quotient model.
Connects to the FastAPI backend for predictions and analysis.
"""

import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from typing import Dict, List, Optional
import os

# =============================================================================
# Configuration
# =============================================================================

# API Configuration - Use environment variable or default
API_BASE_URL = os.getenv("API_BASE_URL", "https://purchasing-power-ml-api.onrender.com")

# Page Configuration
st.set_page_config(
    page_title="PPP-Q Investment Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .grade-A { background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); }
    .grade-B { background: linear-gradient(135deg, #56ab2f 0%, #a8e063 100%); }
    .grade-C { background: linear-gradient(135deg, #f7971e 0%, #ffd200 100%); }
    .grade-D { background: linear-gradient(135deg, #cb2d3e 0%, #ef473a 100%); }
    .info-box {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
    }
    .warning-box {
        background: #fff3cd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
    }
    .success-box {
        background: #d4edda;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# Asset Database
# =============================================================================

ASSETS = {
    # Cryptocurrencies
    "Bitcoin": {"category": "Cryptocurrency", "symbol": "BTC", "risk": "High", "liquidity": "High"},
    "Ethereum": {"category": "Cryptocurrency", "symbol": "ETH", "risk": "High", "liquidity": "High"},
    "Litecoin": {"category": "Cryptocurrency", "symbol": "LTC", "risk": "High", "liquidity": "Medium"},
    "Bitcoin_Cash": {"category": "Cryptocurrency", "symbol": "BCH", "risk": "High", "liquidity": "Medium"},
    "Cardano": {"category": "Cryptocurrency", "symbol": "ADA", "risk": "Very High", "liquidity": "Medium"},
    "Solana": {"category": "Cryptocurrency", "symbol": "SOL", "risk": "Very High", "liquidity": "Medium"},
    
    # Precious Metals
    "Gold": {"category": "Precious Metal", "symbol": "XAU", "risk": "Low", "liquidity": "High"},
    "Silver": {"category": "Precious Metal", "symbol": "XAG", "risk": "Medium", "liquidity": "High"},
    
    # Equities
    "SP500": {"category": "Index", "symbol": "SPX", "risk": "Medium", "liquidity": "Very High"},
    "NASDAQ": {"category": "Index", "symbol": "IXIC", "risk": "Medium-High", "liquidity": "Very High"},
    "DowJones": {"category": "Index", "symbol": "DJI", "risk": "Medium", "liquidity": "Very High"},
    "Apple": {"category": "Stock", "symbol": "AAPL", "risk": "Medium", "liquidity": "Very High"},
    "Microsoft": {"category": "Stock", "symbol": "MSFT", "risk": "Medium", "liquidity": "Very High"},
    "JPMorgan": {"category": "Stock", "symbol": "JPM", "risk": "Medium", "liquidity": "Very High"},
    
    # Commodities
    "Oil": {"category": "Commodity", "symbol": "CL", "risk": "High", "liquidity": "High"},
    "NaturalGas": {"category": "Commodity", "symbol": "NG", "risk": "Very High", "liquidity": "High"},
    
    # ETFs
    "Gold_ETF": {"category": "ETF", "symbol": "GLD", "risk": "Low", "liquidity": "Very High"},
    "TreasuryBond_ETF": {"category": "ETF", "symbol": "TLT", "risk": "Low", "liquidity": "Very High"},
    "RealEstate_ETF": {"category": "ETF", "symbol": "VNQ", "risk": "Medium", "liquidity": "High"},
}

MODEL_INFO = {
    "lgbm": {
        "name": "LightGBM",
        "accuracy": "90.28%",
        "description": "Gradient boosting framework using tree-based learning. Best for speed and efficiency.",
        "pros": [
            "⚡ Fastest training and inference",
            "📊 Handles large datasets efficiently",
            "🎯 Best Macro F1 score (90.28%)",
            "💾 Low memory usage",
            "🔧 Native categorical feature support"
        ],
        "cons": [
            "🔄 May overfit on small datasets",
            "📈 Sensitive to hyperparameters",
            "🌳 Less interpretable than simple models"
        ],
        "best_for": "Production deployments requiring fast inference"
    },
    "xgb": {
        "name": "XGBoost",
        "accuracy": "89.44%",
        "description": "Extreme Gradient Boosting with regularization. Known for competition-winning performance.",
        "pros": [
            "🏆 Industry standard for competitions",
            "🛡️ Built-in regularization prevents overfitting",
            "📊 Handles missing values well",
            "🔧 Extensive hyperparameter tuning options",
            "📈 Strong generalization"
        ],
        "cons": [
            "🐌 Slower than LightGBM",
            "💾 Higher memory consumption",
            "⚙️ More complex configuration"
        ],
        "best_for": "When model robustness is priority over speed"
    },
    "ensemble": {
        "name": "Ensemble (LightGBM + XGBoost)",
        "accuracy": "90.35%",
        "description": "Combines predictions from both models for improved reliability.",
        "pros": [
            "🎯 Highest combined accuracy (90.35%)",
            "🛡️ More robust predictions",
            "📊 Reduced variance in predictions",
            "⚖️ Balances strengths of both models",
            "🔄 Dynamic weighting by horizon"
        ],
        "cons": [
            "🐌 Slowest inference (2x computation)",
            "💾 Higher memory (loads 2 models)",
            "🔧 More complex maintenance"
        ],
        "best_for": "Critical decisions requiring highest confidence"
    }
}

COMPONENT_EXPLANATIONS = {
    "inflation_hedge": {
        "name": "Inflation Hedge Score",
        "description": "Measures how well the asset protects against inflation erosion.",
        "factors": ["Historical CPI correlation", "Real return vs inflation", "Purchasing power preservation"],
        "weight_short": 0.35,
        "weight_medium": 0.30,
        "weight_long": 0.25
    },
    "volatility": {
        "name": "Volatility Score",
        "description": "Assesses price stability and risk level of the asset.",
        "factors": ["Standard deviation of returns", "Maximum drawdown", "VIX correlation"],
        "weight_short": 0.30,
        "weight_medium": 0.25,
        "weight_long": 0.20
    },
    "liquidity": {
        "name": "Liquidity Score",
        "description": "Evaluates how easily the asset can be bought/sold without price impact.",
        "factors": ["Trading volume", "Bid-ask spread", "Market depth"],
        "weight_short": 0.20,
        "weight_medium": 0.20,
        "weight_long": 0.15
    },
    "growth": {
        "name": "Growth Potential Score",
        "description": "Measures expected appreciation potential over the investment horizon.",
        "factors": ["Historical CAGR", "Momentum indicators", "Fundamental growth drivers"],
        "weight_short": 0.15,
        "weight_medium": 0.25,
        "weight_long": 0.40
    }
}

# =============================================================================
# API Functions
# =============================================================================

@st.cache_data(ttl=300)
def get_prediction(asset: str, horizon_years: int, model_type: str) -> Optional[Dict]:
    """Get prediction from API."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict",
            json={
                "asset": asset,
                "horizon_years": horizon_years,
                "model_type": model_type
            },
            timeout=30
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Connection Error: {e}")
        return None

@st.cache_data(ttl=300)
def get_asset_list() -> List[str]:
    """Get list of available assets from API."""
    try:
        response = requests.get(f"{API_BASE_URL}/assets", timeout=10)
        if response.status_code == 200:
            return response.json().get("assets", list(ASSETS.keys()))
    except:
        pass
    return list(ASSETS.keys())

@st.cache_data(ttl=300)
def get_model_info() -> Optional[Dict]:
    """Get model information from API."""
    try:
        response = requests.get(f"{API_BASE_URL}/model/info", timeout=10)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None

@st.cache_data(ttl=600)
def compare_assets(assets: List[str], horizon_years: int, model_type: str) -> Optional[Dict]:
    """Compare multiple assets."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/compare",
            json={
                "assets": assets,
                "horizon_years": horizon_years,
                "model_type": model_type
            },
            timeout=60
        )
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None

# =============================================================================
# Visualization Functions
# =============================================================================

def create_gauge_chart(score: float, title: str) -> go.Figure:
    """Create a gauge chart for scores."""
    # Determine color based on score
    if score >= 65:
        color = "#28a745"  # Green
    elif score >= 55:
        color = "#17a2b8"  # Blue
    elif score >= 42:
        color = "#ffc107"  # Yellow
    else:
        color = "#dc3545"  # Red
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 16}},
        number={'suffix': "", 'font': {'size': 40}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 42], 'color': '#ffebee'},
                {'range': [42, 55], 'color': '#fff3e0'},
                {'range': [55, 65], 'color': '#e3f2fd'},
                {'range': [65, 100], 'color': '#e8f5e9'}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 2},
                'thickness': 0.75,
                'value': score
            }
        }
    ))
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
    return fig

def create_component_breakdown(components: Dict) -> go.Figure:
    """Create radar chart for component breakdown."""
    categories = list(components.keys())
    values = list(components.values())
    
    # Close the polygon
    categories = categories + [categories[0]]
    values = values + [values[0]]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Component Scores',
        fillcolor='rgba(31, 119, 180, 0.3)',
        line=dict(color='#1f77b4', width=2)
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )
        ),
        showlegend=False,
        height=400,
        margin=dict(l=60, r=60, t=40, b=40)
    )
    return fig

def create_grade_distribution(grade: str) -> go.Figure:
    """Create visual representation of grade thresholds."""
    grades = ['D', 'C', 'B', 'A']
    thresholds = [0, 42, 55, 65, 100]
    colors = ['#dc3545', '#ffc107', '#17a2b8', '#28a745']
    
    fig = go.Figure()
    
    for i, (g, color) in enumerate(zip(grades, colors)):
        opacity = 1.0 if g == grade else 0.3
        fig.add_trace(go.Bar(
            x=[thresholds[i+1] - thresholds[i]],
            y=[g],
            orientation='h',
            marker=dict(color=color, opacity=opacity),
            name=g,
            text=f"{thresholds[i]}-{thresholds[i+1]}",
            textposition='inside',
            hovertemplate=f"Grade {g}: {thresholds[i]}-{thresholds[i+1]}<extra></extra>"
        ))
    
    fig.update_layout(
        barmode='stack',
        showlegend=False,
        height=150,
        margin=dict(l=30, r=30, t=20, b=20),
        xaxis=dict(title="PPP-Q Score Range", range=[0, 100]),
        yaxis=dict(title="")
    )
    return fig

def create_correlation_heatmap() -> go.Figure:
    """Create correlation heatmap for asset categories."""
    # Sample correlation data (in production, fetch from API)
    categories = ['Crypto', 'Gold', 'Equities', 'Bonds', 'Commodities']
    
    # Simulated correlation matrix
    corr_matrix = np.array([
        [1.00, 0.15, 0.45, -0.20, 0.30],
        [0.15, 1.00, -0.10, 0.35, 0.40],
        [0.45, -0.10, 1.00, -0.30, 0.25],
        [-0.20, 0.35, -0.30, 1.00, -0.15],
        [0.30, 0.40, 0.25, -0.15, 1.00]
    ])
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix,
        x=categories,
        y=categories,
        colorscale='RdBu',
        zmid=0,
        text=np.round(corr_matrix, 2),
        texttemplate='%{text}',
        textfont={"size": 12},
        hovertemplate='%{x} vs %{y}<br>Correlation: %{z:.2f}<extra></extra>'
    ))
    
    fig.update_layout(
        title="Asset Category Correlations",
        height=400,
        margin=dict(l=60, r=60, t=60, b=60)
    )
    return fig

def create_horizon_comparison(asset: str, model_type: str) -> go.Figure:
    """Create comparison across different horizons."""
    horizons = [1, 2, 3, 5, 7, 10]
    scores = []
    grades = []
    
    for h in horizons:
        result = get_prediction(asset, h, model_type)
        if result:
            scores.append(result.get('pppq_score', 50))
            grades.append(result.get('pppq_grade', 'C'))
        else:
            scores.append(50)
            grades.append('C')
    
    colors = ['#28a745' if g == 'A' else '#17a2b8' if g == 'B' else '#ffc107' if g == 'C' else '#dc3545' for g in grades]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[f"{h}Y" for h in horizons],
        y=scores,
        marker_color=colors,
        text=[f"{s:.1f} ({g})" for s, g in zip(scores, grades)],
        textposition='outside'
    ))
    
    # Add threshold lines
    fig.add_hline(y=65, line_dash="dash", line_color="#28a745", annotation_text="Grade A (≥65)")
    fig.add_hline(y=55, line_dash="dash", line_color="#17a2b8", annotation_text="Grade B (≥55)")
    fig.add_hline(y=42, line_dash="dash", line_color="#ffc107", annotation_text="Grade C (≥42)")
    
    fig.update_layout(
        title=f"PPP-Q Score by Investment Horizon - {asset}",
        xaxis_title="Investment Horizon",
        yaxis_title="PPP-Q Score",
        yaxis=dict(range=[0, 100]),
        height=400
    )
    return fig

def create_multi_asset_comparison(comparison_data: Dict) -> go.Figure:
    """Create multi-asset comparison visualization."""
    if not comparison_data or 'comparisons' not in comparison_data:
        return None
    
    assets = []
    scores = []
    grades = []
    
    for comp in comparison_data['comparisons']:
        assets.append(comp['asset'])
        scores.append(comp['pppq_score'])
        grades.append(comp['pppq_grade'])
    
    colors = ['#28a745' if g == 'A' else '#17a2b8' if g == 'B' else '#ffc107' if g == 'C' else '#dc3545' for g in grades]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=assets,
        y=scores,
        marker_color=colors,
        text=[f"{s:.1f}<br>({g})" for s, g in zip(scores, grades)],
        textposition='outside'
    ))
    
    fig.update_layout(
        title="Asset Comparison",
        xaxis_title="Asset",
        yaxis_title="PPP-Q Score",
        yaxis=dict(range=[0, max(scores) + 15]),
        height=400
    )
    return fig

# =============================================================================
# Main App
# =============================================================================

def main():
    # Header
    st.markdown('<h1 class="main-header">📊 PPP-Q Investment Intelligence</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Purchasing Power Preservation Quotient - AI-Powered Asset Analysis</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://raw.githubusercontent.com/bilalahmadsheikh/purchasing_power_ml/main/docs/logo.png", width=150)
        st.markdown("---")
        
        st.header("⚙️ Configuration")
        
        # Model Selection
        st.subheader("🤖 Model Selection")
        model_type = st.selectbox(
            "Select Model",
            options=["ensemble", "lgbm", "xgb"],
            format_func=lambda x: MODEL_INFO[x]["name"],
            help="Choose the ML model for predictions"
        )
        
        # Display model info
        model = MODEL_INFO[model_type]
        st.info(f"**Accuracy:** {model['accuracy']}")
        
        with st.expander("Model Details"):
            st.write(model['description'])
            st.markdown("**✅ Pros:**")
            for pro in model['pros']:
                st.markdown(f"  {pro}")
            st.markdown("**⚠️ Cons:**")
            for con in model['cons']:
                st.markdown(f"  {con}")
            st.caption(f"**Best for:** {model['best_for']}")
        
        st.markdown("---")
        
        # Horizon Selection
        st.subheader("📅 Investment Horizon")
        horizon_mode = st.radio(
            "Selection Mode",
            ["Slider", "Manual Input"],
            horizontal=True
        )
        
        if horizon_mode == "Slider":
            horizon_years = st.slider(
                "Years",
                min_value=1,
                max_value=15,
                value=5,
                help="Investment time horizon in years"
            )
        else:
            horizon_years = st.number_input(
                "Years",
                min_value=1,
                max_value=30,
                value=5,
                help="Enter custom horizon (1-30 years)"
            )
        
        # Horizon context
        if horizon_years < 2:
            horizon_label = "Short-term"
            horizon_desc = "Focus on liquidity & volatility"
        elif horizon_years <= 5:
            horizon_label = "Medium-term"
            horizon_desc = "Balanced approach"
        else:
            horizon_label = "Long-term"
            horizon_desc = "Focus on growth potential"
        
        st.caption(f"**{horizon_label}:** {horizon_desc}")
        
        st.markdown("---")
        
        # API Status
        st.subheader("🔌 API Status")
        try:
            response = requests.get(f"{API_BASE_URL}/health", timeout=5)
            if response.status_code == 200:
                st.success("✅ API Connected")
            else:
                st.warning("⚠️ API Issues")
        except:
            st.error("❌ API Offline")
            st.caption("Using cached data")
    
    # Main Content Tabs
    tabs = st.tabs([
        "🎯 Single Asset Analysis",
        "📊 Compare Assets",
        "📈 Correlations & Insights",
        "📚 Documentation"
    ])
    
    # ==========================================================================
    # Tab 1: Single Asset Analysis
    # ==========================================================================
    with tabs[0]:
        st.header("🎯 Single Asset Analysis")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Asset Selection with search
            available_assets = get_asset_list()
            selected_asset = st.selectbox(
                "Select Asset",
                options=available_assets,
                index=0,
                help="Type to search or select from dropdown"
            )
        
        with col2:
            analyze_btn = st.button("🔍 Analyze Asset", type="primary", use_container_width=True)
        
        if selected_asset:
            # Display asset info
            asset_info = ASSETS.get(selected_asset, {})
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Category", asset_info.get("category", "Unknown"))
            with col2:
                st.metric("Symbol", asset_info.get("symbol", "N/A"))
            with col3:
                st.metric("Risk Level", asset_info.get("risk", "Unknown"))
            with col4:
                st.metric("Liquidity", asset_info.get("liquidity", "Unknown"))
        
        if analyze_btn or selected_asset:
            with st.spinner("Analyzing asset..."):
                result = get_prediction(selected_asset, horizon_years, model_type)
            
            if result:
                st.markdown("---")
                
                # Main Score Display
                col1, col2, col3 = st.columns([1, 2, 1])
                
                with col1:
                    st.plotly_chart(
                        create_gauge_chart(result['pppq_score'], "PPP-Q Score"),
                        use_container_width=True
                    )
                
                with col2:
                    # Grade Display
                    grade = result['pppq_grade']
                    grade_colors = {'A': 'grade-A', 'B': 'grade-B', 'C': 'grade-C', 'D': 'grade-D'}
                    
                    st.markdown(f"""
                    <div class="metric-card {grade_colors.get(grade, '')}">
                        <h1 style="font-size: 4rem; margin: 0;">Grade {grade}</h1>
                        <p style="font-size: 1.2rem; margin: 0.5rem 0;">{result.get('recommendation', 'Analysis Complete')}</p>
                        <p style="font-size: 0.9rem; opacity: 0.8;">{horizon_years}-Year Horizon | {MODEL_INFO[model_type]['name']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Grade thresholds visualization
                    st.plotly_chart(create_grade_distribution(grade), use_container_width=True)
                
                with col3:
                    st.metric("Confidence", f"{result.get('confidence', 0.85)*100:.1f}%")
                    st.metric("Model", MODEL_INFO[model_type]['name'])
                    st.metric("Horizon", f"{horizon_years} Years")
                
                st.markdown("---")
                
                # Component Breakdown
                st.subheader("📊 Component Score Breakdown")
                
                if 'component_scores' in result:
                    components = result['component_scores']
                    
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        st.plotly_chart(
                            create_component_breakdown(components),
                            use_container_width=True
                        )
                    
                    with col2:
                        for comp_key, comp_value in components.items():
                            comp_info = COMPONENT_EXPLANATIONS.get(comp_key, {})
                            
                            with st.expander(f"**{comp_info.get('name', comp_key)}**: {comp_value:.1f}/100"):
                                st.write(comp_info.get('description', ''))
                                st.markdown("**Key Factors:**")
                                for factor in comp_info.get('factors', []):
                                    st.markdown(f"- {factor}")
                                
                                # Weight by horizon
                                if horizon_years < 2:
                                    weight = comp_info.get('weight_short', 0.25)
                                elif horizon_years <= 5:
                                    weight = comp_info.get('weight_medium', 0.25)
                                else:
                                    weight = comp_info.get('weight_long', 0.25)
                                
                                st.progress(weight)
                                st.caption(f"Weight for {horizon_years}Y horizon: {weight*100:.0f}%")
                
                st.markdown("---")
                
                # Insights
                st.subheader("💡 AI-Generated Insights")
                
                if 'insights' in result:
                    insights = result['insights']
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("### ✅ Strengths")
                        for strength in insights.get('strengths', []):
                            st.markdown(f'<div class="success-box">{strength}</div>', unsafe_allow_html=True)
                            st.write("")
                    
                    with col2:
                        st.markdown("### ⚠️ Risks")
                        for risk in insights.get('risks', []):
                            st.markdown(f'<div class="warning-box">{risk}</div>', unsafe_allow_html=True)
                            st.write("")
                    
                    if 'recommendation' in insights:
                        st.markdown("### 📋 Recommendation")
                        st.markdown(f'<div class="info-box">{insights["recommendation"]}</div>', unsafe_allow_html=True)
                
                st.markdown("---")
                
                # Horizon Comparison
                st.subheader("📅 Score Across Different Horizons")
                st.plotly_chart(
                    create_horizon_comparison(selected_asset, model_type),
                    use_container_width=True
                )
    
    # ==========================================================================
    # Tab 2: Compare Assets
    # ==========================================================================
    with tabs[1]:
        st.header("📊 Compare Multiple Assets")
        
        available_assets = get_asset_list()
        
        # Multi-select for assets
        selected_assets = st.multiselect(
            "Select Assets to Compare (2-6 assets)",
            options=available_assets,
            default=["Bitcoin", "Gold", "SP500"],
            max_selections=6,
            help="Select 2-6 assets for comparison"
        )
        
        if len(selected_assets) >= 2:
            compare_btn = st.button("🔄 Compare Assets", type="primary")
            
            if compare_btn:
                with st.spinner("Comparing assets..."):
                    comparison = compare_assets(selected_assets, horizon_years, model_type)
                
                if comparison:
                    # Comparison Chart
                    fig = create_multi_asset_comparison(comparison)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Detailed Comparison Table
                    st.subheader("📋 Detailed Comparison")
                    
                    comparison_data = []
                    for comp in comparison.get('comparisons', []):
                        comparison_data.append({
                            "Asset": comp['asset'],
                            "PPP-Q Score": f"{comp['pppq_score']:.1f}",
                            "Grade": comp['pppq_grade'],
                            "Recommendation": comp.get('recommendation', 'N/A')
                        })
                    
                    df = pd.DataFrame(comparison_data)
                    st.dataframe(df, use_container_width=True, hide_index=True)
                    
                    # Best pick
                    if comparison.get('comparisons'):
                        best = max(comparison['comparisons'], key=lambda x: x['pppq_score'])
                        st.success(f"🏆 **Best Pick for {horizon_years}Y Horizon:** {best['asset']} (Score: {best['pppq_score']:.1f}, Grade: {best['pppq_grade']})")
                else:
                    # Fallback: individual predictions
                    st.info("Fetching individual predictions...")
                    results = []
                    for asset in selected_assets:
                        result = get_prediction(asset, horizon_years, model_type)
                        if result:
                            results.append({
                                "Asset": asset,
                                "PPP-Q Score": result['pppq_score'],
                                "Grade": result['pppq_grade']
                            })
                    
                    if results:
                        df = pd.DataFrame(results)
                        st.dataframe(df.sort_values('PPP-Q Score', ascending=False), use_container_width=True)
        else:
            st.warning("Please select at least 2 assets to compare")
    
    # ==========================================================================
    # Tab 3: Correlations & Insights
    # ==========================================================================
    with tabs[2]:
        st.header("📈 Correlations & Market Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Asset Category Correlations")
            st.plotly_chart(create_correlation_heatmap(), use_container_width=True)
            
            st.markdown("""
            **Interpretation:**
            - **Positive correlation (red):** Assets move together
            - **Negative correlation (blue):** Assets move opposite
            - **Near zero:** Independent movement
            
            **Diversification Tip:** Combine negatively correlated assets to reduce portfolio risk.
            """)
        
        with col2:
            st.subheader("Asset Categories Overview")
            
            categories = {}
            for asset, info in ASSETS.items():
                cat = info.get('category', 'Other')
                if cat not in categories:
                    categories[cat] = []
                categories[cat].append(asset)
            
            for cat, assets in categories.items():
                with st.expander(f"**{cat}** ({len(assets)} assets)"):
                    for asset in assets:
                        info = ASSETS[asset]
                        st.markdown(f"- **{asset}** ({info['symbol']}) - Risk: {info['risk']}")
        
        st.markdown("---")
        
        st.subheader("📊 Market Regime Indicators")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="info-box">
            <h4>🏦 Inflation Environment</h4>
            <p>Current CPI trends and monetary policy impact asset valuations differently.</p>
            <p><strong>High Inflation Favors:</strong> Gold, Commodities, TIPS</p>
            <p><strong>Low Inflation Favors:</strong> Growth Stocks, Long Bonds</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="info-box">
            <h4>📈 Growth vs Value</h4>
            <p>Economic cycle phase affects optimal asset allocation.</p>
            <p><strong>Expansion:</strong> Equities, Crypto outperform</p>
            <p><strong>Contraction:</strong> Bonds, Gold provide safety</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="info-box">
            <h4>⚡ Volatility Regime</h4>
            <p>VIX levels indicate market stress and risk appetite.</p>
            <p><strong>Low VIX (<15):</strong> Risk-on environment</p>
            <p><strong>High VIX (>25):</strong> Defensive positioning advised</p>
            </div>
            """, unsafe_allow_html=True)
    
    # ==========================================================================
    # Tab 4: Documentation
    # ==========================================================================
    with tabs[3]:
        st.header("📚 Documentation & Methodology")
        
        st.markdown("""
        ## PPP-Q Model Overview
        
        The **Purchasing Power Preservation Quotient (PPP-Q)** is a machine learning-based scoring system 
        that evaluates assets based on their ability to preserve and grow purchasing power over time.
        
        ### Grading System
        
        | Grade | Score Range | Interpretation |
        |-------|-------------|----------------|
        | **A** | ≥ 65 | Excellent - Strong purchasing power preservation |
        | **B** | 55 - 64 | Good - Above average protection |
        | **C** | 42 - 54 | Fair - Moderate protection with some risks |
        | **D** | < 42 | Poor - High risk to purchasing power |
        
        ### Component Scores
        
        The final PPP-Q score is calculated from four weighted components:
        """)
        
        for key, comp in COMPONENT_EXPLANATIONS.items():
            with st.expander(f"**{comp['name']}**"):
                st.write(comp['description'])
                st.markdown("**Key Factors:**")
                for factor in comp['factors']:
                    st.markdown(f"- {factor}")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Short-term Weight", f"{comp['weight_short']*100:.0f}%")
                with col2:
                    st.metric("Medium-term Weight", f"{comp['weight_medium']*100:.0f}%")
                with col3:
                    st.metric("Long-term Weight", f"{comp['weight_long']*100:.0f}%")
        
        st.markdown("""
        ### Model Architecture
        
        The system uses an **ensemble approach** combining:
        
        1. **LightGBM** - Fast gradient boosting for efficient inference
        2. **XGBoost** - Robust predictions with built-in regularization
        
        The ensemble averages predictions from both models, with dynamic weighting 
        based on the investment horizon for optimal accuracy.
        
        ### Data Sources
        
        - **Economic Data:** Federal Reserve (FRED) - CPI, Interest Rates, M2, GDP
        - **Asset Prices:** Yahoo Finance - Real-time and historical prices
        - **Crypto Data:** Yahoo Finance - Cryptocurrency prices and metrics
        
        ### API Reference
        
        The dashboard connects to a FastAPI backend. Key endpoints:
        
        | Endpoint | Method | Description |
        |----------|--------|-------------|
        | `/predict` | POST | Get PPP-Q prediction for an asset |
        | `/compare` | POST | Compare multiple assets |
        | `/assets` | GET | List available assets |
        | `/model/info` | GET | Get model metadata |
        
        ### Version Information
        
        - **Model Version:** v1.2.0
        - **Dashboard Version:** 1.0.0
        - **Last Updated:** December 2025
        """)

# =============================================================================
# Run App
# =============================================================================

if __name__ == "__main__":
    main()
