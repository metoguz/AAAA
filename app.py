


import streamlit as st
import joblib
import pandas as pd

@st.cache_data
def get_data():
    df = pd.read_csv('data/kalp_hasta.csv')
    return df

tab_home, tab_modelinfo,  tab_model = st.tabs(["Ana Sayfa", "Model Hakkında Bilgi", "Model"])
#anasayfa
column_hastalik, column_dataset = tab_home.columns(2)
column_dataset.subheader("Kalp Hastalıkları Veri Seti")
df = get_data()
column_dataset.dataframe(df)

column_hastalik.subheader("Kalp Hastalığı Nedir?")
column_hastalik.write("Kalp hastalığı, kalpte meydana gelen ve kalbi etkileyen herhangi bir bozukluğu kapsayan bir terimdir. Kalp hastalığı başlığı altında koroner arter hastalığı gibi kan damar hastalıkları, kalp ritmi problemleri (aritmiler) ve bu hastalıkların yanında doğuştan gelen kalp kusurları yer alır.")
column_hastalik.write("Erkeklerde en sık 50-60 yaşları arasında görülürken,kadınlarda koruyucu östrojen hormonunun azaldığı dönem olan (menopoz) 60-70 yaşları arasında daha sık görülür.")
column_hastalik.subheader("İçerik")
column_hastalik.write("Kaggledan temin ettiğim hazır veri setini modelime göre işleyip görselleştirerek gerçekçi tahminler üretebilecek bir makine öğrenmesi modeli tasarladım.")
column_hastalik.subheader("Veri Setimdeki Özellikler ve Anlamları")
column_hastalik.write("Veri setim 13 öznitelik ve bu 13 özniteliğe bağlı değişkenlik gösteren amaç değişkeninden oluşmaktadır.")
column_hastalik.subheader("•	Yaş:")
column_hastalik.write("Kişinin Yaşı")
column_hastalik.subheader("•	Cinsiyet:")
column_hastalik.write("Kişinin Cinsiyeti")
column_hastalik.subheader("•	Göğüs Ağrı Tipi")
column_hastalik.markdown("0: Tipik Göğüs Ağrısı   "
                       "1: Tipik Olmayan Göğüs Ağrısı   "
                       "2: Göğüs Ağrısı Olmayan   " 
                       "3: Belirtisiz")
column_hastalik.subheader("•	Dinlenme Kan Basıncı:")
column_hastalik.write("Kişinin İstirahat Tansiyonu")
column_hastalik.subheader("•	Kolesterol:")
column_hastalik.write("mg /dl cinsinden Serum / Kolesterol")
column_hastalik.subheader("•	Açlık Kan Şekeri:")
column_hastalik.write("0:120 mg/ml'den az "
                    "1:12 mg/ml'den fazla")
column_hastalik.subheader("•	Dinlenme Elektrokardiyografik Ölçümü:")
column_hastalik.write("0: Normal "
                    "1: ST-T Dalgası Anormalliği "
                    "2: Sol Ventriküller Hipertrofi")
column_hastalik.subheader("•	Ulaşılan Maksimum Kalp Hızı:")
column_hastalik.write("Ulaşılan Maksimum Kalp Hızı")
column_hastalik.subheader("•	Egzersize Bağlı Göğüs Ağrısı:")
column_hastalik.write("1: Evet "
                    "0: Hayır")
column_hastalik.subheader("•	ST Depresyonu: ")
column_hastalik.write("Dinlenmeye bağlı egzersizden kaynaklanan ST depresyonu")
column_hastalik.subheader("Eğim")
column_hastalik.write("0: Yukarı Eğimli "
                    "1: Düz " 
                    "2: Aşağı Eğimli")
column_hastalik.subheader("•	CA:")
column_hastalik.write("Floroskopi ile renklendirilmiş ana damarların tal sayısı 5 farklı değer  veri alır.")
column_hastalik.subheader("•	Talasemi: ")
column_hastalik.write("0: Normal " 
                    "1: Sabit Kusur "
                    "2: Tersinir Kusur "
                    "3: Tanımlanamayan")
#model hakkında
tab_modelinfo.title("Modele Yönelik Çalışmalar ")
tab_modelinfo.subheader("Yöntem")
tab_modelinfo.write("Kalp hastalığı olan bireyleri tespit edebilmek için KNN makine öğrenmesi algoritmasını kullandım.Bu algoritma sınıflandırıcı ve değişkenler üzerindeki ilişkiyi ortaya koyan bir analiz yöntemidir.KNN algoritması denetimli bir makine öğrenmesi algoritmasıdır.Bu algoritma genellikle sınıflandırma problemlerinin çözümünde kullanılan bir yöntemdir.")
tab_modelinfo.header("Veri Analizi")
tab_modelinfo.write("1. Tek Değişkenli Analiz: Burada, verisetinin dağılımını ve kapsamını anlamak için her seferinde tek bir özelliğe odaklandım." 
                    "\n 2. İki Değişkenli Analiz: Bu adımda her özellik ile amaç (çıktı) arasındaki ilişkiye bakacağız. Bu, her özelliğin amaç (çıktı) üzerindeki önemini ve etkisini anlamamıza yardımcı olur.")
tab_modelinfo.subheader("Çıkarım")
tab_modelinfo.image("galeri/Figure_4.png", use_column_width=True)
tab_modelinfo.write("Sayısal özellikler için:  "
                    " \nKalp hastalığı olan kişiler, stres testleri sırasında olmayanlara göre daha yüksek bir kalp atış hızına ulaştıkları gözlemleniyor\n"
                    " \nST depresyonu kalp hastalığı olan kişlerde belirgin şekilde daha düşüktür.")
tab_modelinfo.image("galeri/Figure_7.png",use_column_width=True)
tab_modelinfo.write("Kategorik özellikler için:  "
                    "\nÖzetle görsel temsile dayanarak\n "
                    "\nAmaç Değişkeni Üzerinde Daha Yüksek Etki: ca, göğüs ağrı tipi,egzersize bağlı göğüs ağrısı, cinsiyet, eğim ve talasemi\n"
                    "\nAmaç Değişkeni Üzerinde Orta Derecede Etki: dinlenme elektrokardiyografik ölçüm\n "
                    "\nAmaç Değişkeni Üzerindeki Düşük Etki: açlık kan şekeri.")
tab_modelinfo.header("Veri Ön İşleme")
tab_modelinfo.write("Veri ön işleme kısmında; Kayıp değer kontrol, Aykırı değer kontrol, Kategorik Özelliklerin Kodlanması, Çarpık Unsurları Dönüştürme ve Veri Sızıntısı başlıklarını ele aldım.")
tab_modelinfo.write("Gerekli kontrolleri yaptığımda herhangi bir kayıp değere rastlamadım.IQR yöntemini kullanarak aykırı değer tesbiti yaptığımda ise şu sonuçlara vardım:\n"
                    " \nYaş: 0 aykırı değer\n"
                    " \nDinlenme Kan Basıncı:9 aykırı değer\n"
                    "\nKolesterol: 5 aykırı değer\n"
                    "\nUlaşılan Maksimum kalp Hızı:1 aykırı değer\n"
                    "\nST Deprestonu: 5 aykırı değer")
tab_modelinfo.image('galeri/Figure_78.jpeg', use_column_width=True)
tab_modelinfo.write("Kullandığım algoritmanın doğası (KNN) ve veri setimin küçük boyutunu göz önüne alınca,aykırı değerleri doğrudan kaldırmanın işime yaramayacağını düşündüm."
                    "Bunun yerine aykırı değerlerin etkisini azaltmak ve verileri modelleme için uygun hale getirmek amacıyla box-cox dönüşümünü uyguladım aynı zamanda veriyi daha normal dağılıma benzeyen hale"
                    " getirmeye yarayan bir yöntemdir.Yukarıdaki şekilde bunu gözlemleyebiliyoruz.")
tab_modelinfo.write("Kategorik özelliklerin kodlanması:\n"
                    "\nKategorik özelliklerimizi nominal ve ordinal değişken olarak sınıflayıp nominal olan değişkenlerime one-hot kodlama yapıp sayısal hale getirdim")
tab_modelinfo.header("Model Oluşturma")
tab_modelinfo.write("Öncelikle kNN modelini tanımladım ve ölçelendirme ile birleştirdiğim bir pipeline kurdum\n"
                    "\n Daha sonra Hyperparameters grid (hiperparametreler ızgarası) kurdum ve KNN iş akışı için en uygun"
                    " hiperparametreleri belirleyeceğim bir fonksiyon oluşturdum.Ve bunu kullanarak modelimi oluşturdum.")
tab_modelinfo.header("Modelin Performansını Değerlendirme")
tab_modelinfo.image("galeri/Figusre_7.png", use_column_width=True)
# Eğitilmiş modeli yükle
model = joblib.load('egitilmis_model.pkl')
# uygulama
tab_model.title('Kalp Hastalığı Tahmin Uygulaması')
tab_model.markdown('Bu uygulama, kullanıcının sağlık verilerini kullanarak kalp hastalığı riskini tahmin eder.')
# Kullanıcıdan giriş verilerini al
yas=st.sidebar.slider("Yaşınız",min_value=0,max_value=100,value=30)
cinsiyet = st.sidebar.radio('Cinsiyet(0E,1K)', [0, 1])
dinlenme_kan_basıncı = st.sidebar.slider('Dinlenme Kan Basıncı', min_value=80, max_value=200, value=120)
Kolesterol = st.sidebar.slider('Kolesterol', min_value=100, max_value=400, value=200)
Aclık_Kan_Sekeri = st.sidebar.radio('Açlık Kan Şekeri(0 Normal,1 Yüksek)', [0, 1])
Ulasılan_maks_kalp_hızı = st.sidebar.slider('Ulaşılan Maksimum Kalp Hızı', min_value=80, max_value=200, value=150)
egzersize_baglı_durumu = st.sidebar.radio('Egzersize Bağlı Göğüs Ağrısı(0 Yok 1 Var)', [0, 1])
depresyon_ST = st.sidebar.slider('ST Depresyonu', min_value=0.0, max_value=6.2)
egim = st.sidebar.slider('Eğim', min_value=0, max_value=2, value=1)
ca = st.sidebar.slider('Floroskopi ile Renkledirilmiş Ana Damar Sayısı', min_value=0, max_value=4)
st.sidebar.subheader("Göğüs Ağrı Tipi")
gogus_agrı_tipi_0 = st.sidebar.radio('Göğüs Ağrı Tip0', [0, 1])
gogus_agrı_tipi_1 = st.sidebar.radio('Göğüs Ağrı Tip1', [0, 1])
gogus_agrı_tipi_2 = st.sidebar.radio('Göğüs Ağrı Tipi2', [0, 1])
gogus_agrı_tipi_3 = st.sidebar.radio('Göğüs Ağrı Tipi3', [0, 1])
st.sidebar.subheader("EKG Ölçümü")
Elektrokardiyografik_Ölcümü_0 = st.sidebar.radio('normal', [0, 1])
Elektrokardiyografik_Ölcümü_1 = st.sidebar.radio('ST T Dalgası Anormalliği', [0, 1])
Elektrokardiyografik_Ölcümü_2 = st.sidebar.radio('Sol Ventriküller Hipertrofi', [0, 1])
st.sidebar.subheader("Talasemi")
talasemi_0 = st.sidebar.radio('Normal', [0, 1])
talasemi_1 = st.sidebar.radio('Sabit Kusur', [0, 1])
talasemi_2 = st.sidebar.radio('Tersinir Kusur', [0, 1])
talasemi_3 = st.sidebar.radio('Tanımlanamayan', [0, 1])



# Tahmin yap
if st.sidebar.button('Tahmin Yap'):
    input_data = {
        'yas': yas,
        'cinsiyet': cinsiyet,
        'dinlenme_kan_basıncı': dinlenme_kan_basıncı,
        'Kolesterol': Kolesterol,
        'Aclık_Kan_Sekeri': Aclık_Kan_Sekeri,
        'Ulasılan_maks_kalp_hızı': Ulasılan_maks_kalp_hızı,
        'egzersize_baglı_durumu': egzersize_baglı_durumu,
        'depresyon_ST': depresyon_ST,
        'egim': egim,
        'ca': ca,
        'gogus_agrı_tipi_1': gogus_agrı_tipi_1,
        'gogus_agrı_tipi_2': gogus_agrı_tipi_2,
        'gogus_agrı_tipi_3': gogus_agrı_tipi_3,
        'Elektrokardiyografik_Ölcümü_1': Elektrokardiyografik_Ölcümü_1,
        'Elektrokardiyografik_Ölcümü_2': Elektrokardiyografik_Ölcümü_2,
        'talasemi_1': talasemi_1,
        'talasemi_2': talasemi_2,
        'talasemi_3': talasemi_3

    }
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)  # Giriş verilerini modele uygula

    # Sonucu göster
    tab_model.subheader('Sonuç:')
    if prediction[0] == 1:
        tab_model.error('Kalp Hastalığı Riski Yüksek! En Yakın Sağlık Kuruluşuna Başvurunuz')
    else:
        tab_model.success('Kalp Hastalığı Riski Düşük.')

# Model hakkında bilgi
st.sidebar.header('Model Hakkında')
st.sidebar.info('Gireceğiniz veriler hakkında eksik bilgiler için "ANASAYFA" sekmesinde bulunan "verisetimdeki özellikler ve anlamları" kısmına göz atabilirsiniz...')

# İstediğiniz kadar giriş ekleyebilir ve modelinize uygun hale getirebilirsiniz.










