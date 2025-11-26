import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ic50_calculator import IC50Calculator
import io

# Page configuration
st.set_page_config(
    page_title="IC50 Calculator",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for cyber style
st.markdown("""
    <style>
    /* Dark cyber theme */
    .stApp {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 50%, #0f1419 100%);
    }
    
    /* Main header with neon glow */
    .main-header {
        font-size: 3rem;
        font-weight: 900;
        background: linear-gradient(90deg, #00ffff, #00ff88, #00ffff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 1rem;
        text-shadow: 0 0 20px rgba(0, 255, 255, 0.5);
        letter-spacing: 2px;
        font-family: 'Courier New', monospace;
    }
    
    .sub-header {
        color: #00ff88;
        font-size: 1.2rem;
        font-weight: 600;
        margin-top: 0.5rem;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: rgba(10, 14, 39, 0.8);
        border-right: 1px solid #00ffff33;
    }
    
    /* Metric cards with neon border */
    [data-testid="stMetricValue"] {
        color: #00ffff;
        font-size: 1.8rem;
        font-weight: bold;
    }
    
    [data-testid="stMetricLabel"] {
        color: #00ff88;
        font-size: 0.9rem;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #00ffff22, #00ff8822);
        color: #00ffff;
        border: 1px solid #00ffff;
        border-radius: 4px;
        transition: all 0.3s;
        font-weight: 600;
    }
    
    .stButton>button:hover {
        background: linear-gradient(90deg, #00ffff44, #00ff8844);
        box-shadow: 0 0 15px rgba(0, 255, 255, 0.5);
        transform: translateY(-2px);
    }
    
    /* Input fields */
    .stNumberInput>div>div>input, .stTextInput>div>div>input {
        background-color: rgba(0, 255, 255, 0.05);
        border: 1px solid #00ffff33;
        color: #00ffff;
    }
    
    .stSelectbox>div>div>select {
        background-color: rgba(0, 255, 255, 0.05);
        border: 1px solid #00ffff33;
        color: #00ffff;
    }
    
    /* Success/Info boxes */
    .stSuccess {
        background-color: rgba(0, 255, 136, 0.1);
        border-left: 3px solid #00ff88;
        color: #00ff88;
    }
    
    .stInfo {
        background-color: rgba(0, 255, 255, 0.1);
        border-left: 3px solid #00ffff;
        color: #00ffff;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: rgba(0, 255, 255, 0.05);
        border: 1px solid #00ffff33;
        color: #00ffff;
    }
    
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Text colors */
    h1, h2, h3 {
        color: #00ffff;
    }
    
    p, label {
        color: #b0b0b0;
    }
    
    /* Divider */
    hr {
        border-color: #00ffff33;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">⚡ IC50 CALCULATOR</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #00ff88; font-size: 0.9rem; letter-spacing: 3px; margin-top: -1rem;">━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━</p>', unsafe_allow_html=True)

# Sidebar for input options
st.sidebar.markdown('<p class="sub-header">⚡ INPUT</p>', unsafe_allow_html=True)
input_method = st.sidebar.radio(
    "",
    ["Manual Entry", "CSV Upload", "Example Data"],
    label_visibility="collapsed"
)

st.sidebar.markdown('<p class="sub-header">⚙️ CONFIG</p>', unsafe_allow_html=True)
model_type = st.sidebar.selectbox(
    "Model",
    ["4PL", "3PL", "5PL"],
    index=0,
    help="4PL: 4-Parameter Logistic (recommended)"
)

robust_method = st.sidebar.selectbox(
    "Robust Method",
    ["iterative", "huber"],
    index=0
)

detect_outliers = st.sidebar.checkbox("Detect Outliers", value=True)
outlier_method = st.sidebar.selectbox(
    "Outlier Method",
    ["cooks", "iqr", "zscore", "residual"],
    index=0,
    disabled=not detect_outliers
)

exclude_zero = st.sidebar.checkbox("Exclude Zero", value=True)

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown('<p class="sub-header">📋 DATA INPUT</p>', unsafe_allow_html=True)
    
    concentrations = None
    responses = None
    has_multiple_measurements = False
    
    if input_method == "Manual Entry":
        num_points = st.number_input("Data Points", min_value=3, max_value=20, value=6, step=1)
        
        # Concentration input
        st.markdown('<p style="color: #00ff88; font-size: 0.9rem; margin-top: 1rem;">CONCENTRATIONS</p>', unsafe_allow_html=True)
        conc_inputs = []
        for i in range(num_points):
            conc_val = st.number_input(f"C{i+1}", min_value=0.0, value=0.001 if i == 0 else 10.0**(i-3), 
                                      format="%.6f", key=f"conc_{i}", label_visibility="collapsed")
            conc_inputs.append(conc_val)
        
        # Response input
        st.markdown('<p style="color: #00ff88; font-size: 0.9rem; margin-top: 1rem;">RESPONSES (%)</p>', unsafe_allow_html=True)
        resp_inputs = []
        for i in range(num_points):
            resp_val = st.number_input(f"R{i+1}", min_value=-100.0, max_value=100.0, 
                                      value=float(i*15), key=f"resp_{i}", label_visibility="collapsed")
            resp_inputs.append(resp_val)
        
        concentrations = np.array(conc_inputs)
        responses = np.array(resp_inputs)
        
        # Multiple measurements option
        use_multiple = st.checkbox("Multiple Measurements", value=False)
        
        if use_multiple:
            num_measurements = st.number_input("Number of measurements", min_value=2, max_value=5, value=2, step=1)
            all_responses = [resp_inputs]  # First measurement
            
            for m in range(1, num_measurements):
                st.write(f"**Measurement {m+1}:**")
                resp_m = []
                for i in range(num_points):
                    resp_val = st.number_input(f"Response {i+1} (%)", min_value=-100.0, max_value=100.0, 
                                              value=float(i*15), key=f"resp_{m}_{i}")
                    resp_m.append(resp_val)
                all_responses.append(resp_m)
            
            responses = all_responses
            has_multiple_measurements = True
    
    elif input_method == "CSV Upload":
        uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.dataframe(df.head(), use_container_width=True)
                
                # Column selection
                conc_col = st.selectbox("Concentration Column", df.columns.tolist())
                resp_cols = st.multiselect("Response Column(s)", df.columns.tolist(), 
                                          default=[df.columns[1]] if len(df.columns) > 1 else None)
                
                if conc_col and resp_cols:
                    concentrations = df[conc_col].values
                    
                    if len(resp_cols) == 1:
                        responses = df[resp_cols[0]].values
                        has_multiple_measurements = False
                    else:
                        responses = [df[col].values for col in resp_cols]
                        has_multiple_measurements = True
                    
                    st.success(f"✓ Loaded {len(concentrations)} points")
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    else:  # Example Data
        st.markdown('<p style="color: #00ff88;">Example Dataset</p>', unsafe_allow_html=True)
        
        concentrations = np.array([0.001, 0.01, 0.1, 1, 5, 10])
        inhibition_1 = np.array([-10, 5, 10, 40, 75, 85])
        inhibition_2 = np.array([-11, 1, 10, 50, 74, 85])
        responses = [inhibition_1, inhibition_2]
        has_multiple_measurements = True
        
        # Display example data
        example_df = pd.DataFrame({
            'Concentration': concentrations,
            'Measurement 1': inhibition_1,
            'Measurement 2': inhibition_2
        })
        st.dataframe(example_df)

with col2:
    st.markdown('<p class="sub-header">📊 RESULTS</p>', unsafe_allow_html=True)
    
    if concentrations is not None and responses is not None:
        try:
            # Initialize calculator
            calculator = IC50Calculator(model=model_type)
            
            # Perform fitting
            with st.spinner("⚡ Processing..."):
                results = calculator.robust_fit(
                    concentrations,
                    responses,
                    detect_outliers=detect_outliers,
                    outlier_method=outlier_method,
                    exclude_zero=exclude_zero,
                    robust_method=robust_method
                )
            
            # Display results
            st.success("✓ Calculation Complete")
            
            # Key results in boxes
            col_a, col_b, col_c = st.columns(3)
            
            with col_a:
                st.metric("IC50", f"{results['IC50']:.4f}")
            
            with col_b:
                st.metric("R²", f"{results['R_squared']:.4f}")
            
            with col_c:
                if results['IC50_CI']:
                    ci_range = results['IC50_CI'][1] - results['IC50_CI'][0]
                    st.metric("CI Range", f"{ci_range:.4f}")
            
            # Detailed results
            with st.expander("⚡ DETAILS", expanded=False):
                st.write(f"**Model:** {results['model']}")
                st.write(f"**IC50:** {results['IC50']:.6f}")
                
                if results['IC50_CI']:
                    st.write(f"**IC50 95% CI:** [{results['IC50_CI'][0]:.6f}, {results['IC50_CI'][1]:.6f}]")
                
                st.write(f"**R²:** {results['R_squared']:.4f}")
                
                if model_type == '4PL':
                    st.write(f"**Bottom:** {results['bottom']:.4f}")
                    st.write(f"**Top:** {results['top']:.4f}")
                    st.write(f"**Hill Slope:** {results['hill_slope']:.4f}")
                elif model_type == '3PL':
                    st.write(f"**Bottom:** {results['bottom']:.4f}")
                    st.write(f"**Top:** {results['top']:.4f}")
                elif model_type == '5PL':
                    st.write(f"**Bottom:** {results['bottom']:.4f}")
                    st.write(f"**Top:** {results['top']:.4f}")
                    st.write(f"**Hill Slope:** {results['hill_slope']:.4f}")
                    st.write(f"**Asymmetry:** {results['asymmetry']:.4f}")
                
                if calculator.outliers is not None:
                    num_outliers = np.sum(calculator.outliers)
                    if num_outliers > 0:
                        st.warning(f"⚠ {num_outliers} outlier(s) detected")
                        outlier_indices = np.where(calculator.outliers)[0]
                        st.write(f"Indices: {outlier_indices.tolist()}")
                    else:
                        st.success("✓ No outliers")
            
            # Generate plot
            st.markdown('<p class="sub-header" style="margin-top: 2rem;">📈 VISUALIZATION</p>', unsafe_allow_html=True)
            
            # Cyber-style plot with dark background
            fig, ax = plt.subplots(figsize=(10, 6), facecolor='#0a0e27')
            ax.set_facecolor('#0a0e27')
            
            # Prepare data for plotting
            conc_plot = concentrations[concentrations != 0] if exclude_zero else concentrations
            
            if calculator.has_multiple_measurements:
                inhib_plot = calculator.response_mean[concentrations != 0] if exclude_zero else calculator.response_mean
                inhib_std = calculator.response_std[concentrations != 0] if exclude_zero else calculator.response_std
            else:
                resp_array = np.array(responses)
                inhib_plot = resp_array[concentrations != 0] if exclude_zero else resp_array
                inhib_std = None
            
            # Generate smooth curve
            x_fit = np.logspace(np.log10(min(conc_plot)), np.log10(max(conc_plot) * 10), 100)
            
            if model_type == '4PL':
                y_fit = calculator.four_pl(x_fit, *calculator.params)
            elif model_type == '3PL':
                y_fit = calculator.three_pl(x_fit, *calculator.params)
            else:  # 5PL
                y_fit = calculator.five_pl(x_fit, *calculator.params)
            
            # Plot data points with neon colors
            ax.scatter(conc_plot, inhib_plot, s=120, alpha=0.8, label='Data (mean)' if calculator.has_multiple_measurements else 'Data', 
                      color='#00ffff', edgecolors='#00ff88', linewidths=1.5, zorder=3)
            
            # Add error bars if multiple measurements
            if inhib_std is not None:
                ax.errorbar(conc_plot, inhib_plot, yerr=inhib_std, fmt='none', 
                           color='#00ff88', alpha=0.6, capsize=4, label='±SD', zorder=2, linewidth=1.5)
            
            # Mark outliers
            if calculator.outliers is not None:
                outlier_mask = calculator.outliers[concentrations != 0] if exclude_zero else calculator.outliers
                if np.any(outlier_mask):
                    ax.scatter(conc_plot[outlier_mask], inhib_plot[outlier_mask], 
                              s=200, marker='x', color='#ff0088', linewidths=4, label='Outlier', zorder=4)
            
            # Plot fitted curve with neon cyan
            ax.plot(x_fit, y_fit, color='#00ffff', linewidth=2.5, label='Fitted Curve', zorder=1, alpha=0.9)
            
            # Add IC50 line with neon green
            ax.axvline(results['IC50'], color='#00ff88', linestyle='--', alpha=0.8, linewidth=2.5, label='IC50')
            
            # Add confidence interval lines
            if results['IC50_CI']:
                ax.axvline(results['IC50_CI'][0], color='#00ff88', linestyle=':', alpha=0.5, linewidth=2)
                ax.axvline(results['IC50_CI'][1], color='#00ff88', linestyle=':', alpha=0.5, linewidth=2)
                ylim = ax.get_ylim()
                ax.text(results['IC50_CI'][0], ylim[1]*0.95, f"CI: {results['IC50_CI'][0]:.3f}", 
                       rotation=90, va='top', ha='right', color='#00ff88', fontsize=9, fontweight='bold')
                ax.text(results['IC50_CI'][1], ylim[1]*0.95, f"CI: {results['IC50_CI'][1]:.3f}", 
                       rotation=90, va='top', ha='left', color='#00ff88', fontsize=9, fontweight='bold')
            
            # Formatting with cyber colors
            ax.set_xscale('log')
            ax.set_xlabel('Concentration', fontsize=12, color='#00ffff', fontweight='bold')
            ax.set_ylabel('Response (%)', fontsize=12, color='#00ffff', fontweight='bold')
            ax.set_title(f'{model_type} Model | IC50 = {results["IC50"]:.4f} | R² = {results["R_squared"]:.4f}', 
                        fontsize=13, fontweight='bold', color='#00ff88', pad=15)
            ax.legend(loc='best', fontsize=9, facecolor='#0a0e27', edgecolor='#00ffff', labelcolor='#00ffff')
            ax.grid(True, alpha=0.2, color='#00ffff', linestyle='--')
            ax.tick_params(colors='#00ffff', which='both')
            for spine in ax.spines.values():
                spine.set_edgecolor('#00ffff')
                spine.set_linewidth(1.5)
            
            plt.tight_layout()
            st.pyplot(fig, facecolor='#0a0e27')
            
            # Download options
            st.markdown('<p class="sub-header" style="margin-top: 2rem;">💾 EXPORT</p>', unsafe_allow_html=True)
            
            col_d1, col_d2 = st.columns(2)
            
            with col_d1:
                # Download plot
                buf = io.BytesIO()
                fig.savefig(buf, format='png', dpi=300, bbox_inches='tight', facecolor='#0a0e27')
                buf.seek(0)
                st.download_button(
                    label="📥 Plot (PNG)",
                    data=buf,
                    file_name="ic50_plot.png",
                    mime="image/png"
                )
            
            with col_d2:
                # Download results as CSV
                results_df = pd.DataFrame({
                    'Parameter': ['IC50', 'IC50_CI_Lower', 'IC50_CI_Upper', 'R_squared', 'Model'],
                    'Value': [
                        results['IC50'],
                        results['IC50_CI'][0] if results['IC50_CI'] else 'N/A',
                        results['IC50_CI'][1] if results['IC50_CI'] else 'N/A',
                        results['R_squared'],
                        results['model']
                    ]
                })
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Results (CSV)",
                    data=csv,
                    file_name="ic50_results.csv",
                    mime="text/csv"
                )
        
        except Exception as e:
            st.error(f"✗ Error: {str(e)}")
            st.info("Check input data and try again.")
    else:
        st.info("← Select input method in sidebar")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #00ffff33; font-size: 0.8rem; letter-spacing: 2px; margin-top: 2rem;'>
        <p>IC50 CALCULATOR | STREAMLIT</p>
    </div>
    """,
    unsafe_allow_html=True
)


