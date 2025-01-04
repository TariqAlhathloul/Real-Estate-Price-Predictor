import streamlit as st
import joblib
import pandas as pd
import os
import csv


st.set_page_config(
    page_title="Real Estate Price Predictor",
    page_icon="🏠",
    layout="centered",
    initial_sidebar_state="expanded",
)

@st.cache_resource
def load_models(model_path):
    """
    loading the models
    caching ensures models are only loaded  once.
    """
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        st.error(f"Model file not found: {model_path}")
        return None

# model paths
MODEL_PATHS = {
    'شقة للبيع': os.path.join(os.path.dirname(__file__), "..", "models","gbr_apartment_sale_model(1).joblib"),
    'شقة للإيجار': os.path.join(os.path.dirname(__file__), "..","models","gbr_apartment_rent_model(1).joblib"),
    'فيلا للبيع':  os.path.join(os.path.dirname(__file__), "..","models","rfr_villa_sale_model.pkl"),
    'فيلا للإيجار': os.path.join(os.path.dirname(__file__), "..","models","rfr_villa_rent_model.pkl")
}

property_types = {
    'شقة للبيع': 'apartment_sale',
    'شقة للإيجار': 'apartment_rent',
    'فيلا للبيع': 'villa_sale',
    'فيلا للإيجار': 'villa_rent'
}

# sidebar
st.sidebar.header("أدخل تفاصيل العقار")

property_type = st.sidebar.selectbox(
    'نوع العقار',
    list(MODEL_PATHS.keys())
)

# load the selected model
model = load_model(MODEL_PATHS[property_type])

loaded_models = {}

DISTRICTS_PATH = os.path.join(os.path.dirname(__file__), "..", "data","district_mapping.csv")
def get_district_id(district_name: str) -> int:
    distric_mapping = {}
    
    with open(DISTRICTS_PATH, mode='r') as file:
        reader = csv.reader(file)
        for row in reader:
            distric_mapping[row[1]] = row[0]

    district_id = distric_mapping.get(district_name)
    return int(district_id)

def districts() -> int:
    district_mapping = {}
   
    with open(DISTRICTS_PATH, mode='r') as file:
        reader = csv.reader(file)
        for row in reader:
            district_mapping[row[1]] = row[0]

    
    return district_mapping.keys()

# get the property details
def user_input_features():
    district_name = st.sidebar.selectbox('الحي', districts())
    livings = st.sidebar.number_input('الصالات', min_value=0, max_value=10, value=2, step=1)
    has_kitchen = st.sidebar.selectbox('يحتوي مطبخ', ('نعم', 'لا'))
    has_acs = st.sidebar.selectbox('يحتوي تكييف', ('نعم', 'لا'))
    floor = st.sidebar.number_input('عدد الطوابق، أو رقم الطابق', min_value=0, max_value=100, value=2, step=1)
    is_furnished = st.sidebar.selectbox('مؤثثة', ('نعم', 'لا'))
    area = st.sidebar.number_input('المساحة بالمتر المربع', min_value=10.0, max_value=1000.0, value=150.0, step=0.1)
    rooms = st.sidebar.number_input('الغرف', min_value=0, max_value=10, value=2, step=1)
    wc = st.sidebar.number_input('دورات المياه', min_value=0, max_value=10, value=3, step=1)
    longitude = st.sidebar.number_input('خطوط الطول', format="%.6f", value=46.656104)
    latitude = st.sidebar.number_input('خطوط العرض', format="%.6f", value=24.859876)
    
    # encoding categorical var
    has_kitchen = 1 if has_kitchen == 'نعم' else 0
    has_acs = 1 if has_acs == 'نعم' else 0
    is_furnished = 1 if is_furnished == 'نعم' else 0
    
    data = {
        'district_id': get_district_id(district_name),
        'livings': livings,
        'has_kitchen': has_kitchen,
        'has_acs': has_acs,
        'floor': floor,
        'is_furnished': is_furnished,
        'area': area,
        'rooms': rooms,
        'wc': wc,
        'longitude': longitude,
        'latitude': latitude
    }
    return data

input_data = user_input_features()

# main panel
st.title("🏠 التنبؤ بسعر العقار")
st.markdown("""
اهلاً بك في تطبيق التنبؤ بسعر العقار في مدينة الرياض. يمكنك استخدام هذا التطبيق لتوقع سعر العقار.
""")

st.subheader("التفاصيل المُدخلة")
col1, col2 = st.columns(2)

with col1:
    st.markdown(f"**نوع العقار:** {property_type}")
    st.markdown(f"**الحي:** {input_data['district_id']}")
    st.markdown(f"**الصالات:** {input_data['livings']}")
    st.markdown(f"**مطبخ:** {'نعم' if input_data['has_kitchen'] else 'لا'}")
    st.markdown(f"**تكييف:** {'نعم' if input_data['has_acs'] else 'لا'}")
    st.markdown(f"**الطوابق:** {input_data['floor']}")
    

with col2:
    st.markdown(f"**مؤثثة:** {'نعم' if input_data['is_furnished'] else 'لا'}")
    st.markdown(f"**المساحة بالمتر المربع:** {input_data['area']}")
    st.markdown(f"**الغرف:** {input_data['rooms']}")
    st.markdown(f"**دورات المياه:** {input_data['wc']}")
    st.markdown(f"**خطوط الطول:** {input_data['longitude']}")
    st.markdown(f"**خطوط العرض:** {input_data['latitude']}")

# function to make prediction
def predict_price(input_data, model):
    data = pd.DataFrame([input_data])
    
    # if model:
    #     prediction = model.predict(input_df)
    #     return prediction[0]
    # else:
    #     return None
    type = property_types[property_type]
    if type == 'apartment_sale':
        rent_pred = model["apartment_rent"].predict(data)
        sale_pred = model["apartment_sale"].predict(data)
        rent_yield = (rent_pred[0] / sale_pred[0]) * 100
        return {f"السعر المتوقع للبيع {sale_pred[0]:,.2f}", 
                f"السعر المتوقع للإيجار{rent_pred[0]:,.2f}",
                f"العائد السنوي المتوقع للإيجار السنوي {rent_yield:.2f}%"
            }
    
    elif type == 'apartment_rent':
        rent_pred = model["apartment_rent"].predict(data)
        sale_pred = model["apartment_sale"].predict(data)
        return {"Predicted Sale Price": f"{sale_pred[0]:,.2f}", 
                "Predicted Rental Price": f"{rent_pred[0]:,.2f}"
            }
    
    elif type == 'villa_rent':
        rent_pred = model["villa_rent"].predict(data)
        return {"Predicted Rental Price": f"{rent_pred[0]:,.2f}"}
    
    elif type == 'villa_sale':
        sale_pred = model["villa_sale"].predict(data)
        return {"Predicted Sale Price": f"{sale_pred[0]:,.2f}"}
    
    else:
        return {"message": "Invalid Property Type"}
    

# predict btton
if st.button("🔮 توقع السعر"):
    if model:
        with st.spinner("جاري التنبؤ..."):
            price = predict_price(input_data, model)
        if price is not None:
            st.success(f" **السعر المتوقع للعقار {price:,.2f}** ريال سعودي")
        else:
            st.error("لم يتم التنبؤ. يرجى التحقق من النموذج.")
    else:
        st.error("لم يتم تحميل النموذج. يرجى التحقق من مسار النموذج.")


st.markdown("---")
st.markdown("""
*هذا التطبيق يستخدم تعلم الآلة للتنبؤ بأسعار العقار في مدينة الرياض خصوصاً، دقة الأسعار المعطاة تعتمد على جودة البيانات المُدخلة.*
""")
