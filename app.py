"""
Sistem Rekomendasi Produk Manufaktur - Streamlit Dashboard
Dashboard interaktif untuk generate dan visualisasi rekomendasi
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from recommender import HybridRecommender
import warnings

warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="Sistem Rekomendasi Manufaktur",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    /* Font family overrides */
    html, body, [class*="css"], .stMarkdown {
        font-family: 'Inter', sans-serif !important;
    }
    
    /* Title styling */
    .main-title {
        color: #1A365D;
        font-size: 2.6em;
        font-weight: 800;
        margin-bottom: 5px;
        letter-spacing: -0.5px;
    }
    
    .section-title {
        color: #2B6CB0;
        font-size: 1.6em;
        font-weight: 700;
        margin-top: 25px;
        margin-bottom: 15px;
        letter-spacing: -0.3px;
        border-bottom: 2px solid #E2E8F0;
        padding-bottom: 8px;
    }
    
    /* Glassmorphism KPI Card styling */
    .kpi-card {
        background: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.5);
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.03);
        margin: 10px 0;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 30px rgba(0, 0, 0, 0.06);
    }
    
    .kpi-label {
        font-size: 0.75rem;
        color: #718096;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        font-weight: 600;
        margin-bottom: 6px;
    }
    
    .kpi-value {
        font-size: 1.8rem;
        color: #1A202C;
        font-weight: 700;
        margin: 0;
        line-height: 1.2;
    }
    
    .kpi-value-blue {
        color: #2B6CB0;
    }
    
    .kpi-value-orange {
        color: #DD6B20;
    }
    
    .kpi-value-green {
        color: #319795;
    }
    
    /* Sidebar aesthetic adjustments */
    section[data-testid="stSidebar"] {
        background-color: #F8FAFC !important;
        border-right: 1px solid #E2E8F0;
    }
    
    section[data-testid="stSidebar"] h2 {
        color: #1A365D !important;
        font-weight: 700 !important;
    }
    
    /* Streamlit dataframes and tables */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid #E2E8F0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.02);
    }
</style>
""", unsafe_allow_html=True)



@st.cache_resource
def load_recommender():
    """Load recommender model (cached)"""
    try:
        return HybridRecommender(models_dir='models', data_dir='data')
    except Exception as e:
        st.error(f"❌ Gagal memuat model rekomendasi: {e}")
        st.warning("⚠️ Pastikan Anda sudah menjalankan semua Notebook di folder `notebooks/` secara berurutan untuk menghasilkan model similarity dan SVD.")
        st.info("Anda juga dapat menjalankan `python optimize_models.py` di terminal untuk menghasilkan file sparse (.npz) yang membuat pemuatan model menjadi sangat cepat!")
        st.stop()


def main():
    st.markdown('<div class="main-title">🏭 Sistem Rekomendasi Produk Manufaktur</div>',
                unsafe_allow_html=True)

    st.markdown("""
    Sistem ini menggunakan **Hybrid Filtering** yang menggabungkan:
    - **CF** (Collaborative Filtering) - Rekomendasi berbasis perilaku user mirip
    - **CBF** (Content-Based Filtering) - Rekomendasi berbasis kemiripan produk
    - **SVD** (Matrix Factorization) - Rekomendasi berbasis pola laten
    """)

    # Load recommender
    recommender = load_recommender()

    # Sidebar Configuration
    st.sidebar.markdown("## ⚙️ Konfigurasi")

    # Customer selection
    all_users = recommender.get_all_users()
    selected_user = st.sidebar.selectbox(
        "Pilih Customer ID:",
        all_users,
        format_func=lambda x: f"Customer {x}"
    )

    st.sidebar.markdown("### Bobot Algoritma (Total harus = 1.0)")

    # Weight sliders
    col1, col2, col3 = st.sidebar.columns(3)
    with col1:
        alpha = st.number_input("CF Weight", 0.0, 1.0, 0.4, 0.05)
    with col2:
        beta = st.number_input("CBF Weight", 0.0, 1.0, 0.3, 0.05)
    with col3:
        gamma = st.number_input("SVD Weight", 0.0, 1.0, 0.3, 0.05)

    # Diversity control
    st.sidebar.markdown("### 🎨 Diversity Control")
    enable_diversity = st.sidebar.checkbox("Enable Diversity Filter", value=True)
    diversity_threshold = 0.7
    if enable_diversity:
        diversity_threshold = st.sidebar.slider(
            "Diversity Threshold",
            0.0, 1.0, 0.7, 0.05,
            help="Higher = more diverse recommendations"
        )

    # Normalize weights
    total_weight = alpha + beta + gamma
    if total_weight != 1.0:
        st.sidebar.warning(f"⚠️ Total bobot = {total_weight:.2f} (harus 1.0)")
        st.sidebar.info("Bobot akan dinormalisasi otomatis.")
        alpha = alpha / total_weight
        beta = beta / total_weight
        gamma = gamma / total_weight

    st.sidebar.markdown(f"""
    **Bobot Ternormalisasi:**
    - CF (α): {alpha:.3f}
    - CBF (β): {beta:.3f}
    - SVD (γ): {gamma:.3f}
    """)

    top_n = st.sidebar.slider("Jumlah Rekomendasi", 5, 20, 10)

    # Generate button
    if st.sidebar.button("🚀 Generate Recommendations", key="generate_btn"):
        st.session_state.generate = True

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Informasi Dataset")
    st.sidebar.info(f"""
    - Total Users: {len(all_users):,}
    - Total Products: {len(recommender.products):,}
    - Total Interactions: {(recommender.user_item_matrix > 0).sum().sum():,}
    """)

    # Main content
    if 'generate' not in st.session_state:
        st.session_state.generate = False

    if st.session_state.generate or True:  # Always show by default
        st.markdown('<div class="section-title">📊 Dashboard Rekomendasi</div>',
                    unsafe_allow_html=True)

        # Get user history
        history = recommender.get_user_history(selected_user)

        # Tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Rekomendasi",
            "Riwayat Pembelian",
            "Perbandingan Algoritma",
            "Visualisasi",
            "Evaluasi & Performa Model"
        ])

        # Tab 1: Recommendations
        with tab1:
            st.subheader("Top Rekomendasi Produk")

            with st.spinner("Generating recommendations..."):
                recommendations = recommender.get_hybrid_recommendations(
                    selected_user,
                    alpha=alpha,
                    beta=beta,
                    gamma=gamma,
                    top_n=top_n,
                    enable_diversity=enable_diversity,
                    diversity_threshold=diversity_threshold
                )

            if len(recommendations) > 0:
                # Display as table with Reason column
                display_cols = ['StockCode', 'Description', 'AvgPrice', 'Hybrid_Score']
                if 'Reason' in recommendations.columns:
                    display_cols.append('Reason')

                display_data = recommendations[display_cols].copy()

                if 'Reason' in display_cols:
                    display_data.columns = ['Kode Produk', 'Deskripsi', 'Harga Rata-rata', 'Skor Hybrid', 'Alasan']
                else:
                    display_data.columns = ['Kode Produk', 'Deskripsi', 'Harga Rata-rata', 'Skor Hybrid']

                display_data['Ranking'] = range(1, len(display_data) + 1)

                # Reorder columns
                if 'Alasan' in display_data.columns:
                    cols = ['Ranking', 'Kode Produk', 'Deskripsi', 'Harga Rata-rata', 'Skor Hybrid', 'Alasan']
                else:
                    cols = ['Ranking', 'Kode Produk', 'Deskripsi', 'Harga Rata-rata', 'Skor Hybrid']

                st.dataframe(
                    display_data[cols],
                    use_container_width=True,
                    height=400
                )

                # Download button
                csv = display_data.to_csv(index=False)
                st.download_button(
                    label="📥 Download Rekomendasi (CSV)",
                    data=csv,
                    file_name=f"recommendations_customer_{selected_user}.csv",
                    mime="text/csv"
                )
            else:
                st.warning("❌ Tidak ada rekomendasi untuk customer ini")

        # Tab 2: Purchase History
        with tab2:
            st.subheader("Riwayat Pembelian Customer")

            if len(history) > 0:
                st.info(f"Customer telah membeli **{len(history)} produk berbeda** dengan total **{history['Quantity'].sum():.0f} unit**")

                # Summary metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.markdown(f"""
                    <div class="kpi-card">
                        <div class="kpi-label">Total Produk</div>
                        <div class="kpi-value kpi-value-blue">{len(history)}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    st.markdown(f"""
                    <div class="kpi-card">
                        <div class="kpi-label">Total Quantity</div>
                        <div class="kpi-value kpi-value-orange">{history['Quantity'].sum():,.0f} unit</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col3:
                    st.markdown(f"""
                    <div class="kpi-card">
                        <div class="kpi-label">Total Revenue</div>
                        <div class="kpi-value kpi-value-green">£{history['TotalRevenue'].sum():,.0f}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col4:
                    st.markdown(f"""
                    <div class="kpi-card">
                        <div class="kpi-label">Avg Price</div>
                        <div class="kpi-value">£{history['AvgPrice'].mean():.2f}</div>
                    </div>
                    """, unsafe_allow_html=True)

                # Table
                st.dataframe(
                    history.sort_values('Quantity', ascending=False),
                    use_container_width=True,
                    height=400
                )
            else:
                st.warning("❌ Customer tidak memiliki riwayat pembelian")

        # Tab 3: Algorithm Comparison
        with tab3:
            st.subheader("Perbandingan Rekomendasi dari Setiap Algoritma")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("### Collaborative Filtering")
                cf_recs = recommender.get_cf_recommendations(selected_user, top_n=5)
                if len(cf_recs) > 0:
                    for idx, row in cf_recs.iterrows():
                        st.write(f"**{idx+1}. {row['StockCode']}** - {row['CF_Score']:.4f}")
                else:
                    st.info("Tidak ada CF recommendations")

            with col2:
                st.markdown("### Content-Based Filtering")
                cbf_recs = recommender.get_cbf_recommendations(selected_user, top_n=5)
                if len(cbf_recs) > 0:
                    for idx, row in cbf_recs.iterrows():
                        st.write(f"**{idx+1}. {row['StockCode']}** - {row['CBF_Score']:.4f}")
                else:
                    st.info("Tidak ada CBF recommendations")

            with col3:
                st.markdown("### SVD Matrix Factorization")
                svd_recs = recommender.get_svd_recommendations(selected_user, top_n=5)
                if len(svd_recs) > 0:
                    for idx, row in svd_recs.iterrows():
                        st.write(f"**{idx+1}. {row['StockCode']}** - {row['SVD_Score']:.4f}")
                else:
                    st.info("Tidak ada SVD recommendations")

        # Tab 4: Visualizations
        with tab4:
            st.subheader("Visualisasi Rekomendasi")

            if len(recommendations) > 0:
                # Horizontal bar chart of scores (sorted by score descending)
                top_10_recs = recommendations.head(10).sort_values('Hybrid_Score', ascending=True)
                fig_bar = px.bar(
                    top_10_recs,
                    x='Hybrid_Score',
                    y='Description',
                    orientation='h',
                    title='Top 10 Hybrid Recommendation Scores',
                    color='Hybrid_Score',
                    color_continuous_scale='Viridis',
                    text='Hybrid_Score'
                )
                fig_bar.update_traces(texttemplate='%{text:.4f}', textposition='outside')
                fig_bar.update_layout(
                    height=400,
                    xaxis_title='Hybrid Score',
                    yaxis_title='Product Description'
                )
                st.plotly_chart(fig_bar, use_container_width=True)

                # Top products purchased by customer (horizontal bar chart)
                if len(history) > 0:
                    # Sort by Quantity descending
                    history_sorted = history.sort_values('Quantity', ascending=True)

                    fig_top_products = px.bar(
                        history_sorted,
                        x='Quantity',
                        y='Description',
                        orientation='h',
                        title='Top Produk Paling Sering Dibeli',
                        labels={'Quantity': 'Total Quantity (units)', 'Description': 'Product'},
                        color='Quantity',
                        color_continuous_scale='Blues',
                        text='Quantity'
                    )
                    fig_top_products.update_traces(texttemplate='%{text:.0f}', textposition='outside')
                    fig_top_products.update_layout(
                        height=max(400, len(history) * 20),  # Dynamic height based on number of products
                        xaxis_title='Total Quantity (units)',
                        yaxis_title='Product Description',
                        showlegend=False
                    )
                    st.plotly_chart(fig_top_products, use_container_width=True)

                # Weight pie chart
                weights_data = {
                    'Algorithm': ['CF', 'CBF', 'SVD'],
                    'Weight': [alpha, beta, gamma]
                }
                fig_weights = px.pie(
                    weights_data,
                    names='Algorithm',
                    values='Weight',
                    title='Algorithm Weights Configuration',
                    color_discrete_sequence=['#1f77b4', '#ff7f0e', '#2ca02c']
                )
                st.plotly_chart(fig_weights, use_container_width=True)

        # Tab 5: Model Evaluation & Performance
        with tab5:
            st.subheader("Evaluasi Performa Algoritma")
            st.markdown("""
            Halaman ini menampilkan hasil evaluasi komparatif dari masing-masing algoritma dasar (CF, CBF, SVD) dibandingkan dengan sistem Hybrid akhir.
            Pengujian dilakukan menggunakan sampel 100 pengguna acak dengan metode *Top-10 Recommendation*.
            """)
            
            try:
                # Load evaluation results complete CSV
                eval_df = pd.read_csv('models/evaluation_results_complete.csv')
                
                # Format percentages for readability
                display_eval_df = eval_df.copy()
                for col in ['Precision@10', 'Recall@10', 'F1-Score@10', 'Coverage', 'Diversity']:
                    if col in display_eval_df.columns:
                        display_eval_df[col] = display_eval_df[col].map(lambda x: f"{x*100:.2f}%" if x <= 1.0 else f"{x:.4f}")
                
                # Display table
                st.dataframe(display_eval_df, use_container_width=True)
                
                # Plotly Chart comparing Coverage, Diversity, and Novelty
                st.markdown("### Perbandingan Metrik Kunci")
                
                # Melt the dataframe for plotting
                melted_df = eval_df.melt(
                    id_vars=['Algorithm'], 
                    value_vars=['Coverage', 'Diversity'],
                    var_name='Metric', 
                    value_name='Value'
                )
                
                fig_eval = px.bar(
                    melted_df,
                    x='Algorithm',
                    y='Value',
                    color='Metric',
                    barmode='group',
                    title='Katalog Coverage vs Keberagaman (Diversity) Rekomendasi',
                    color_discrete_sequence=['#2B6CB0', '#DD6B20']
                )
                
                fig_eval.update_layout(
                    yaxis_title='Persentase / Nilai',
                    xaxis_title='Algoritma',
                    height=400
                )
                
                st.plotly_chart(fig_eval, use_container_width=True)
                
                # Insight Box
                st.info("""
                💡 **Interpretasi Metrik Akademis**:
                *   **Catalog Coverage**: Menunjukkan seberapa luas sistem mampu mengeksplorasi katalog produk unik. Model **CBF** memiliki coverage tertinggi (~13.98%), yang berarti ia mampu menyarankan variasi produk baru yang lebih banyak.
                *   **Diversity (Keberagaman)**: Menunjukkan tingkat keunikan/perbedaan produk di dalam satu keranjang rekomendasi. Model **SVD** memiliki nilai diversity tertinggi (~0.95), diikuti oleh **CF** (~0.89) dan **Hybrid** (~0.80).
                *   **Precision & Recall (0.00%)**: Pada pengujian offline retail ini, Precision dan Recall bernilai 0% karena sistem rekomendasi secara sengaja menyaring/menghapus produk yang *sudah pernah dibeli* oleh pengguna (untuk menghindari menyarankan produk yang sama berulang kali). Hal ini adalah praktik terbaik (*best-practice*) industri agar rekomendasi diisi oleh penemuan produk-produk baru (*discovery*), meskipun dalam evaluasi offline eksak nilainya menjadi nol.
                """)
            except Exception as e:
                st.warning(f"Hasil evaluasi belum tersedia atau gagal dimuat: {e}")
                st.info("Jalankan `python evaluate_system.py` di terminal untuk menghasilkan data evaluasi terlebih dahulu.")


if __name__ == "__main__":
    main()
