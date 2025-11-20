import streamlit as st
import pandas as pd
from src.matcher import SupplierMatcher
from src.learner import MatchLearner
import plotly.express as px
import plotly.graph_objects as go

# Page config
st.set_page_config(
    page_title="Supplier Matching Engine",
    page_icon="🔍",
    layout="wide"
)

# Initialize session state
if 'matches' not in st.session_state:
    st.session_state.matches = None
if 'accepted_matches' not in st.session_state:
    st.session_state.accepted_matches = []
if 'rejected_matches' not in st.session_state:
    st.session_state.rejected_matches = []
if 'df' not in st.session_state:
    st.session_state.df = None
if 'learner' not in st.session_state:
    st.session_state.learner = MatchLearner()
if 'current_match_scores' not in st.session_state:
    st.session_state.current_match_scores = None

# Title
st.title("🔍 Intelligent Supplier Matching Engine")
st.markdown("""
**The Problem:** Companies waste 100+ hours/week manually matching duplicate suppliers.

**The Solution:** AI-powered matching that learns from your corrections and gets smarter over time.

**Built with:** 6 algorithms (Levenshtein, Fuzzy Matching, Token Sort, Jaro-Winkler) + Machine Learning
""")

# Sidebar
st.sidebar.header("⚙️ Configuration")

use_ml = st.sidebar.checkbox(
    "🧠 Use ML Predictions",
    value=st.session_state.learner.is_trained,
    help="Enable ML to learn from your corrections"
)

threshold = st.sidebar.slider(
    "Match Threshold (%)",
    min_value=50,
    max_value=100,
    value=80,
    help="Minimum similarity score to consider a match"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Statistics")

# Get training stats
training_stats = st.session_state.learner.get_training_stats()
st.sidebar.metric("Training Samples", training_stats['total_samples'])
st.sidebar.metric("Accepted", training_stats['accepted'])
st.sidebar.metric("Rejected", training_stats['rejected'])

if training_stats['model_trained']:
    st.sidebar.success("✓ ML Model Trained")
else:
    st.sidebar.info("ℹ️ Need 10+ samples to train")

# Train button
if st.sidebar.button("🎓 Train ML Model", use_container_width=True):
    with st.spinner("Training model..."):
        success = st.session_state.learner.train(min_samples=10)
        if success:
            st.sidebar.success("✓ Model trained!")
            st.rerun()
        else:
            st.sidebar.error("Need at least 10 samples with both accept/reject")

# Main tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📤 Upload Data",
    "🔍 Find Matches",
    "✅ Review Matches",
    "📊 Analytics",
    "🧠 ML Insights"
])

# ==================== TAB 1: UPLOAD ====================
with tab1:
    st.header("Upload Supplier Data")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Option 1: Upload Your CSV")
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv', 'xlsx'],
            help="File should contain supplier names"
        )
        
        if uploaded_file is not None:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.session_state.df = df
            st.success(f"✓ Loaded {len(df)} records")
            
            name_column = st.selectbox(
                "Which column contains supplier names?",
                options=df.columns.tolist()
            )
            st.session_state.name_column = name_column
            
            st.markdown("#### Data Preview")
            st.dataframe(df.head(10), use_container_width=True)
            
    with col2:
        st.markdown("### Option 2: Use Sample Data")
        if st.button("Load Sample Data", type="primary"):
            try:
                df = pd.read_csv('data/raw/sample_suppliers.csv')
                st.session_state.df = df
                st.session_state.name_column = 'vendor_name'
                st.success(f"✓ Loaded {len(df)} records")
                st.rerun()
            except:
                st.error("Run `python data/generate_sample_data.py` first")

# ==================== TAB 2: FIND MATCHES ====================
with tab2:
    st.header("Find Duplicate Suppliers")
    
    if st.session_state.df is None:
        st.warning("⚠️ Upload data first")
    else:
        df = st.session_state.df
        name_column = st.session_state.name_column
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Suppliers", len(df))
        with col2:
            st.metric("Unique Names", df[name_column].nunique())
        with col3:
            st.metric("Potential Duplicates", len(df) - df[name_column].nunique())
        
        st.markdown("---")
        
        if st.button("🔍 Run Matching Algorithm", type="primary", use_container_width=True):
            with st.spinner("Finding matches..."):
                matcher = SupplierMatcher(threshold=threshold, use_ml=use_ml)
                
                if use_ml and st.session_state.learner.is_trained:
                    matches = matcher.find_matches_with_ml(df, name_column=name_column)
                else:
                    matches = matcher.find_matches(df, name_column=name_column)
                
                st.session_state.matches = matches
                
                if len(matches) > 0:
                    st.success(f"✓ Found {len(matches)} potential matches!")
                else:
                    st.info("No matches found. Try lowering threshold.")
        
        if st.session_state.matches is not None and len(st.session_state.matches) > 0:
            matches = st.session_state.matches
            
            st.markdown("### 📋 Potential Matches")
            
            # Filters
            col1, col2 = st.columns(2)
            with col1:
                min_sim = st.slider(
                    "Min Similarity",
                    float(matches['similarity'].min()),
                    float(matches['similarity'].max()),
                    float(matches['similarity'].min())
                )
            with col2:
                search = st.text_input("Search", "")
            
            filtered = matches[matches['similarity'] >= min_sim]
            if search:
                filtered = filtered[
                    filtered['name_1'].str.contains(search, case=False) |
                    filtered['name_2'].str.contains(search, case=False)
                ]
            
            st.dataframe(
                filtered.style.background_gradient(subset=['similarity'], cmap='RdYlGn'),
                use_container_width=True,
                height=400
            )
            
            csv = filtered.to_csv(index=False)
            st.download_button(
                "📥 Download Matches",
                csv,
                "supplier_matches.csv",
                "text/csv"
            )

# ==================== TAB 3: REVIEW ====================
with tab3:
    st.header("Review and Confirm Matches")
    
    if st.session_state.matches is None or len(st.session_state.matches) == 0:
        st.warning("⚠️ Find matches first")
    else:
        matches = st.session_state.matches
        matcher = SupplierMatcher(threshold=threshold)
        
        total = len(matches)
        reviewed = len(st.session_state.accepted_matches) + len(st.session_state.rejected_matches)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total", total)
        with col2:
            st.metric("Reviewed", reviewed)
        with col3:
            progress_pct = (reviewed / total * 100) if total > 0 else 0
            st.metric("Progress", f"{progress_pct:.1f}%")
        
        st.progress(reviewed / total if total > 0 else 0)
        st.markdown("---")
        
        reviewed_pairs = set(st.session_state.accepted_matches + st.session_state.rejected_matches)
        unreviewed = matches[~matches.apply(
            lambda row: (row['name_1'], row['name_2']) in reviewed_pairs, axis=1
        )]
        
        if len(unreviewed) > 0:
            st.markdown("### 🔍 Review Next Match")
            current = unreviewed.iloc[0]
            
            # Store scores for ML training
            scores = matcher.calculate_similarity(current['name_1'], current['name_2'])
            st.session_state.current_match_scores = scores
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### Supplier 1")
                st.info(f"**{current['name_1']}**")
            with col2:
                st.markdown("#### Supplier 2")
                st.info(f"**{current['name_2']}**")
            
            st.markdown("#### 📊 Similarity Scores")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Overall", f"{current['similarity']:.1f}%")
            with col2:
                st.metric("Levenshtein", f"{current['levenshtein']:.1f}%")
            with col3:
                st.metric("Token Sort", f"{current['token_sort']:.1f}%")
            with col4:
                st.metric("Jaro-Winkler", f"{current['jaro_winkler']:.1f}%")
            
            # ML prediction if available
            if use_ml and st.session_state.learner.is_trained:
                ml_pred, ml_conf = st.session_state.learner.predict(scores)
                st.info(f"🧠 ML Prediction: **{'Match' if ml_pred == 1 else 'No Match'}** (Confidence: {ml_conf*100:.1f}%)")
            
            st.markdown("---")
            col1, col2, col3 = st.columns([1, 1, 2])
            
            with col1:
                if st.button("✅ Accept", type="primary", use_container_width=True):
                    st.session_state.accepted_matches.append((current['name_1'], current['name_2']))
                    # Add to training data
                    st.session_state.learner.add_training_example(
                        st.session_state.current_match_scores,
                        is_match=True
                    )
                    st.success("Match accepted & added to training!")
                    st.rerun()
            
            with col2:
                if st.button("❌ Reject", type="secondary", use_container_width=True):
                    st.session_state.rejected_matches.append((current['name_1'], current['name_2']))
                    # Add to training data
                    st.session_state.learner.add_training_example(
                        st.session_state.current_match_scores,
                        is_match=False
                    )
                    st.warning("Match rejected & added to training!")
                    st.rerun()
            
            with col3:
                st.markdown(f"**{len(unreviewed) - 1}** remaining")
        
        else:
            st.success("🎉 All matches reviewed!")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Accepted", len(st.session_state.accepted_matches))
            with col2:
                st.metric("Rejected", len(st.session_state.rejected_matches))
            
            if len(st.session_state.accepted_matches) > 0:
                accepted_df = pd.DataFrame(
                    st.session_state.accepted_matches,
                    columns=['Supplier_1', 'Supplier_2']
                )
                csv = accepted_df.to_csv(index=False)
                st.download_button(
                    "📥 Download Accepted Matches",
                    csv,
                    "accepted_matches.csv",
                    "text/csv",
                    use_container_width=True
                )

# ==================== TAB 4: ANALYTICS ====================
with tab4:
    st.header("Analytics & Insights")
    
    if st.session_state.matches is None:
        st.warning("⚠️ No matches to analyze")
    else:
        matches = st.session_state.matches
        
        st.markdown("### 📊 Similarity Distribution")
        fig = px.histogram(
            matches,
            x='similarity',
            nbins=20,
            title="Match Score Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("### 🔬 Algorithm Comparison")
        algo_avg = matches[['levenshtein', 'token_sort', 'jaro_winkler']].mean()
        
        fig = go.Figure(data=[
            go.Bar(
                x=['Levenshtein', 'Token Sort', 'Jaro-Winkler'],
                y=algo_avg.values,
                text=[f"{v:.1f}%" for v in algo_avg.values],
                textposition='auto',
                marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1']
            )
        ])
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Avg Similarity", f"{matches['similarity'].mean():.1f}%")
        with col2:
            st.metric("Median", f"{matches['similarity'].median():.1f}%")
        with col3:
            high = len(matches[matches['similarity'] >= 90])
            st.metric("High Confidence (>90%)", high)

# ==================== TAB 5: ML INSIGHTS ====================
with tab5:
    st.header("🧠 Machine Learning Insights")
    
    if not st.session_state.learner.is_trained:
        st.info("ℹ️ ML model not trained yet. Review at least 10 matches to enable ML.")
    else:
        st.success("✓ ML model is trained and active!")
        
        # Feature importance
        st.markdown("### 📊 Feature Importance")
        st.markdown("Which similarity scores matter most for predictions?")
        
        importance_df = st.session_state.learner.get_feature_importance()
        
        if importance_df is not None:
            fig = px.bar(
                importance_df,
                x='importance',
                y='feature',
                orientation='h',
                title="Feature Importance Scores",
                labels={'importance': 'Importance', 'feature': 'Algorithm'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.dataframe(importance_df, use_container_width=True)
        
        # Training data stats
        st.markdown("### 📈 Training Data")
        stats = training_stats
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Samples", stats['total_samples'])
        with col2:
            st.metric("Accepted", stats['accepted'])
        with col3:
            st.metric("Rejected", stats['rejected'])
        
        # Download training data
        try:
            training_df = pd.read_csv('data/training_data.csv')
            csv = training_df.to_csv(index=False)
            st.download_button(
                "📥 Download Training Data",
                csv,
                "training_data.csv",
                "text/csv"
            )
        except:
            pass

# Update sidebar stats
if st.session_state.df is not None:
    st.sidebar.metric("Records", len(st.session_state.df))
if st.session_state.matches is not None:
    st.sidebar.metric("Matches Found", len(st.session_state.matches))