import streamlit as st
import pandas as pd
import os
import json
from PIL import Image
import random
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import NearestNeighbors

# --- AYARLAR ---
st.set_page_config(page_title="AI Fashion Assistant", page_icon="🛍️", layout="wide")

# --- 1. API KEY & GEMINI KURULUMU ---
try:
    import google.generativeai as genai
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False

# API Key'i önce ortam değişkeninden, yoksa config.py'den al
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    try:
        import config
        API_KEY = config.MY_KEY
    except ImportError:
        pass

if API_KEY and HAS_GENAI:
    genai.configure(api_key=API_KEY)

# --- 2. VERİ YÖNETİMİ VE CACHE ---
@st.cache_data # Bu dekoratör veriyi hafızada tutar, her defasında tekrar yüklemez
def load_data():
    DATA_FILE = 'fashion_products_mock_data.csv'
    
    if os.path.exists(DATA_FILE):
        return pd.read_csv(DATA_FILE)
    else:
        # Dosya yoksa burada (Notebook'taki gibi) mock data üretilebilir
        # Şimdilik notebook'ta ürettiğin dosyayı kullanacağımız için hata döndürüyoruz
        return pd.DataFrame() 

df = load_data()

# --- 3. MODEL SINIFI ---
class FashionRecommender:
    def __init__(self, data, weights=None):
        self.df = data.copy()
        
        # Varsayılan veya özel ağırlıklar
        self.weights = weights if weights else {
            'Product_Type': 10.0,
            'Gender': 8.0,
            'Style': 6.0,
            'Color': 5.0,
            'Season': 3.0,
            'Price': 1.5,
            'Brand': 0.5
        }
        
        self.complementary_map = {
            'T-Shirt': ['Jeans', 'Shorts', 'Skirt', 'Sweatpants'],
            'Blouse': ['Skirt', 'Trousers', 'Shorts'],
            'Shirt': ['Trousers', 'Jeans', 'Blazer'],
            'Hoodie': ['Sweatpants', 'Jeans', 'Sneakers'],
            'Jeans': ['T-Shirt', 'Shirt', 'Sweater', 'Sneakers', 'Belt'],
            'Dress': ['Heels', 'Sandals', 'Jewelry', 'Bag'],
            'Sneakers': ['Socks', 'Cap', 'Backpack', 'T-Shirt'],
            'Trousers': ['Shirt', 'T-Shirt', 'Sweater', 'Belt', 'Sneakers'],
        }
        
        self._build_model()

    def _build_model(self):
        categorical_features = ['Product_Type', 'Color', 'Material', 'Pattern', 
                                'Style', 'Season', 'Gender', 'Occasion', 'Fit']
        numerical_features = ['Price', 'Review_Rating'] 

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', MinMaxScaler(), numerical_features),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
            ],
            verbose_feature_names_out=True
        )

        transformed_data = self.preprocessor.fit_transform(self.df)
        feature_names = self.preprocessor.get_feature_names_out()
        
        weighted_data = transformed_data.copy()
        for feature_name, weight in self.weights.items():
            indices = [i for i, name in enumerate(feature_names) if feature_name in name]
            if indices:
                weighted_data[:, indices] = weighted_data[:, indices] * weight
        
        self.features_matrix = weighted_data
        self.model = NearestNeighbors(n_neighbors=2000, metric='cosine', algorithm='brute')
        self.model.fit(self.features_matrix)

    def get_similar_products(self, idx, n_recommendations=15):
        query_vector = self.features_matrix[idx].reshape(1, -1)
        distances, indices = self.model.kneighbors(query_vector)
        similar_indices = indices[0][1:n_recommendations+1]
        return self.df.iloc[similar_indices]

    def get_outfit_recommendation(self, idx, n_recommendations=15):
        source_product = self.df.iloc[idx]
        source_type = source_product['Product_Type']
        target_types = self.complementary_map.get(source_type, ['Jeans', 'Sneakers'])
        
        items_per_category = max(1, n_recommendations // len(target_types)) + 1
        recommended_indices = []
        
        for target_type in target_types:
            target_indices = self.df.index[self.df['Product_Type'] == target_type].tolist()
            if not target_indices: continue
                
            source_vector = self.features_matrix[idx].reshape(1, -1)
            target_vectors = self.features_matrix[target_indices]
            
            k = min(len(target_indices), items_per_category)
            nbrs = NearestNeighbors(n_neighbors=k, metric='cosine').fit(target_vectors)
            dist, relative_idx = nbrs.kneighbors(source_vector)
            
            for i in range(k):
                best_match_idx = target_indices[relative_idx[0][i]]
                recommended_indices.append(best_match_idx)
                
        return self.df.iloc[recommended_indices[:n_recommendations]]

# Modeli Önbelleğe Al (Hızlandırmak için)
@st.cache_resource
def init_recommender(data):
    return FashionRecommender(data)

if not df.empty:
    recommender = init_recommender(df)
else:
    st.error("Veri dosyası bulunamadı! Lütfen önce Notebook'u çalıştırıp veriyi üretin.")
    st.stop()

# --- 4. ARAYÜZ (FRONTEND) ---
st.title("🛍️ AI Fashion Assistant")
st.markdown("Kıyafet fotoğrafını yükle, yapay zeka senin için **benzerlerini** ve **kombin önerilerini** bulsun.")

with st.sidebar:
    st.header("⚙️ Ayarlar")
    strict_mode = st.checkbox("Sıkı Kategori Filtresi", value=True, help="Sadece aynı kategorideki ürünleri getir.")
    st.info("Bu proje Eyyüp Sabri Çelik tarafından geliştirilmiştir.")

uploaded_file = st.file_uploader("Bir fotoğraf seç...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Resmi Göster
    col1, col2 = st.columns([1, 2])
    with col1:
        image = Image.open(uploaded_file)
        st.image(image, caption='Yüklenen Görsel', use_container_width=True)
    
    with col2:
        st.write("🔍 **Analiz Ediliyor...**")
        
        # GEMINI ANALİZİ
        extracted_features = None
        if API_KEY and HAS_GENAI:
            try:
                model = genai.GenerativeModel('gemini-2.0-flash')
                prompt = """
                Analyze the clothing item. Return ONLY JSON.
                Strictly use ONLY the values from lists:
                - Product_Type: ['T-Shirt', 'Blouse', 'Shirt', 'Tank Top', 'Hoodie', 'Sweater', 'Cardigan', 'Jacket', 'Coat', 'Blazer', 'Jeans', 'Trousers', 'Shorts', 'Skirt', 'Dress', 'Sneakers', 'Boots', 'Heels']
                - Color: ['Black', 'White', 'Red', 'Blue', 'Green', 'Yellow', 'Purple', 'Pink', 'Grey', 'Beige', 'Navy', 'Orange', 'Brown', 'Multi']
                - Style: ['Casual', 'Formal', 'Sporty', 'Bohemian', 'Vintage', 'Chic', 'Streetwear', 'Business']
                """
                response = model.generate_content([prompt, image])
                json_str = response.text.strip().replace('```json', '').replace('```', '')
                extracted_features = json.loads(json_str)
                
                # Liste düzeltmesi
                for key, val in extracted_features.items():
                    if isinstance(val, list):
                        extracted_features[key] = val[0] if val else "Unknown"
                        
                st.success(f"Tespit Edildi: **{extracted_features.get('Color')} {extracted_features.get('Product_Type')}** ({extracted_features.get('Style')})")
                
            except Exception as e:
                st.error(f"API Hatası: {e}")
        else:
            st.warning("API Key bulunamadı, Mock Data kullanılıyor.")
            # Mock Data fallback (Basitçe dosya ismine göre)
            extracted_features = {"Product_Type": "T-Shirt", "Color": "White", "Style": "Casual"}

        # ÖNERİ MOTORU ÇALIŞTIRMA
        if extracted_features:
            # Girdiyi vektöre çevir
            input_df = pd.DataFrame([extracted_features])
            for col in recommender.df.columns:
                if col not in input_df.columns:
                    if recommender.df[col].dtype == 'object':
                        mode_val = recommender.df[col].mode()
                        input_df[col] = mode_val[0] if not mode_val.empty else "None"
                    else:
                        input_df[col] = 0
            
            input_df = input_df[recommender.df.columns]
            query_vector = recommender.preprocessor.transform(input_df)
            
            distances, indices = recommender.model.kneighbors(query_vector, n_neighbors=2000)
            all_candidates = indices[0]
            
            # Filtreleme
            if strict_mode:
                candidates = [i for i in all_candidates if recommender.df.iloc[i]['Product_Type'] == extracted_features.get('Product_Type')]
                if not candidates: candidates = all_candidates.tolist()
            else:
                candidates = all_candidates.tolist()
            
            # Renk Önceliği
            detected_color = extracted_features.get('Color')
            exact = [i for i in candidates if recommender.df.iloc[i]['Color'] == detected_color]
            others = [i for i in candidates if recommender.df.iloc[i]['Color'] != detected_color]
            final_indices = (exact + others)[:15]
            
            # Sonuçları Getir
            # 1. Benzer Ürünler
            similar_df = recommender.df.iloc[final_indices][['Product_ID', 'Brand', 'Price', 'Product_Type', 'Color', 'Style']]
            
            # 2. Kombin
            best_match_idx = final_indices[0]
            outfit_df = recommender.get_outfit_recommendation(best_match_idx)[['Product_ID', 'Brand', 'Price', 'Product_Type', 'Color', 'Style']]
            
            st.subheader("✅ Benzer Ürünler")
            st.dataframe(similar_df, use_container_width=True)
            
            st.subheader("✨ Kombin Önerileri")
            st.dataframe(outfit_df, use_container_width=True)