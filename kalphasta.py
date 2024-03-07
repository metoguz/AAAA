import warnings

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import confusion_matrix

warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from scipy.stats import boxcox
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import confusion_matrix

import joblib

warnings.simplefilter(action="ignore")

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 200)

df = pd.read_csv('data/kalp_hasta.csv')


def check_df(dataframe, head=5):
    print("##### Shape #####")  # kac kolon kac satir
    print(dataframe.shape)
    print("##### Types #####")  # kolonların, yani özelliklerin tipleri (int,float,str..)
    print(dataframe.dtypes)
    print("##### Tail #####")  # Veri setinin son 5 degerini inceliyoruz.
    print(dataframe.tail(head))
    print("##### Head #####")  # Veri setinin ilk 5 degerini inceliyoruz.
    print(dataframe.head(head))



check_df(df)


number_of_disease = len(df[df.amac == 1])
number_of_healthy = len(df[df.amac == 0])
percentage_of_disease = (number_of_disease / (len(df.amac)) * 100)
percentage_of_healthy = (number_of_healthy / (len(df.amac)) * 100)
print(f"Kalp Hastaliği olan hastaların oranı : ",percentage_of_disease)
print(f"Sağlıklı olan hastaların oranı : ",percentage_of_healthy)

#continuous (sürekli) değişkenleri tanımlıyoruz
continuous_features = ['yas', 'dinlenme_kan_basıncı', 'Kolesterol', 'Ulasılan_maks_kalp_hızı', 'depresyon_ST']
#dönüştürülecek değişkenleri belirliyoruz
features_to_convert = [feature for feature in df.columns if feature not in continuous_features]
#dönüştürme işlemini yapıyoruz
df[features_to_convert] = df[features_to_convert].astype('object')
print(df.dtypes)

print(df.describe().T)

print(df.describe(include='object'))

# Tek değişkenli analiz için continuous (sürekli) özellikleri filtreleyin
df_continuous = df[continuous_features]

# subplotu ayarla (arkaplan)
fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))

#Sürekli özellliklerin herbirine histogram çizebilmek için döngü
for i, col in enumerate(df_continuous.columns):
    x = i // 3
    y = i % 3
    values, bin_edges = np.histogram(df_continuous[col],
                                     range=(np.floor(df_continuous[col].min()), np.ceil(df_continuous[col].max())))

    graph = sns.histplot(data=df_continuous, x=col, bins=bin_edges, kde=True, ax=ax[x, y],
                         edgecolor='none', color='blue', alpha=0.6, line_kws={'lw': 3})
    ax[x, y].set_xlabel(col, fontsize=15)
    ax[x, y].set_ylabel('Count', fontsize=12)
    ax[x, y].set_xticks(np.round(bin_edges, 1))
    ax[x, y].set_xticklabels(ax[x, y].get_xticks(), rotation=45)
    ax[x, y].grid(color='lightgrey')

    for j, p in enumerate(graph.patches):
        ax[x, y].annotate('{}'.format(p.get_height()), (p.get_x() + p.get_width() / 2, p.get_height() + 1),
                          ha='center', fontsize=10, fontweight="bold")

    textstr = '\n'.join((
        r'$\mu=%.2f$' % df_continuous[col].mean(),
        r'$\sigma=%.2f$' % df_continuous[col].std()
    ))
    ax[x, y].text(0.75, 0.9, textstr, transform=ax[x, y].transAxes, fontsize=12, verticalalignment='top',
                  color='white', bbox=dict(boxstyle='round', facecolor='#ff926e', edgecolor='white', pad=0.5))

ax[1, 2].axis('off')
plt.suptitle('Distribution of Continuous Variables', fontsize=20)
plt.tight_layout()
plt.subplots_adjust(top=0.92)
plt.show()


#Kategorik değişkenleri tanımlama
categorical_features = df.columns.difference(continuous_features)
df_categorical = df[categorical_features]
# 4*2 subplot (arkaplan) ayarla
fig, ax = plt.subplots(nrows=5, ncols=2, figsize=(8,10))

# 4x2 düzeninde her kategorik özellik için çubuk grafikleri çizmek için döngü
for i, col in enumerate(categorical_features):
    row = i // 2
    col_idx = i % 2

    # frekans yüzdeleri hesabı
    value_counts = df[col].value_counts(normalize=True).mul(100).sort_values()

    # çubuk grafiği
    value_counts.plot(kind='barh', ax=ax[row, col_idx], width=0.3, color='blue')

    # çubuklara freakans yüzdeleri ekleme
    for index, value in enumerate(value_counts):
        ax[row, col_idx].text(value, index, str(round(value, 1)) + '%', fontsize=6, weight='bold', va='center')

    ax[row, col_idx].set_xlim([0, 95])
    ax[row, col_idx].set_xlabel('Frekans yüzdesi', fontsize=5)
    ax[row, col_idx].set_title(f'{col}', fontsize=7)

ax[4, 1].axis('off')
plt.suptitle('Kategorik Değişkenlerin Dağılımı', fontsize=10)
plt.tight_layout()
plt.subplots_adjust(top=0.95)
plt.show()


sns.set_palette(['#ff826e', 'blue'])
fig, ax = plt.subplots(len(continuous_features), 2, figsize=(10, 10), gridspec_kw={'width_ratios': [1, 2]})
#grafikler için döngü
for i, col in enumerate(continuous_features):
# her amaç kategorisi için özelliklerin ortalma değeri
    graph = sns.barplot(data=df, x="amac", y=col, ax=ax[i, 0])
  # her amaç kategorisi için dağılımı gösteren kde grafiği
    sns.kdeplot(data=df[df["amac"] == 0], x=col, fill=True, linewidth=2, ax=ax[i, 1], label='0')
    sns.kdeplot(data=df[df["amac"] == 1], x=col, fill=True, linewidth=2, ax=ax[i, 1], label='1')
    ax[i, 1].set_yticks([])
    ax[i, 1].legend(title='Kalp Hastalıkları', loc='upper right')
  # Bar grafiğine ortalama değerler ekleme
    for cont in graph.containers:
        graph.bar_label(cont, fmt='         %.3g')
# grafik genel başlık
plt.suptitle('Sürekli Özellikler ve Amaç Değişkeni', fontsize=15)
plt.tight_layout()
plt.show()

#kategorik özelliklerin amaça göre analizi
#Hedefi kategorik özelliklerden kaldırıyorum
categorical_features = [feature for feature in categorical_features if feature != 'amac']
fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(15, 10))

for i, col in enumerate(categorical_features):


    cross_tab = pd.crosstab(index=df[col], columns=df['amac'])

    #normalize = True argümanını kullanmak bize verilerin indeks bazında oranını verir
    cross_tab_prop = pd.crosstab(index=df[col], columns=df['amac'], normalize='index')

    # renk haritası
    cmp = ListedColormap(['#6f8ede', 'blue'])

    #grafik çizimi
    x, y = i // 4, i % 4
    cross_tab_prop.plot(kind='bar', ax=ax[x, y], stacked=True, width=0.8, colormap=cmp,
                        legend=False, ylabel='Proportion', sharey=True)

    # çubukların oranlarını ve sayılarını grafiğe ekleme
    for idx, val in enumerate([*cross_tab.index.values]):
        for (proportion, count, y_location) in zip(cross_tab_prop.loc[val], cross_tab.loc[val],
                                                   cross_tab_prop.loc[val].cumsum()):
            ax[x, y].text(x=idx - 0.3, y=(y_location - proportion) + (proportion / 2) - 0.03,
                          s=f'    {count}\n({np.round(proportion * 100, 1)}%)',
                          color="black", fontsize=9, fontweight="bold")

    # başlıkları ekleme
    ax[x, y].legend(title='amaç', loc=(0.7, 0.9), fontsize=8, ncol=2)
    # y limitlerini ayarlama
    ax[x, y].set_ylim([0, 1.12])
    # x işaret döndürme
    ax[x, y].set_xticklabels(ax[x, y].get_xticklabels(), rotation=0)

plt.suptitle('Kategorik Özellikler ve Amaç Değişkeni', fontsize=22)
plt.tight_layout()
plt.show()

#veri ön işlemeye geçiş
#kayıp veri varmı kontrol
df.isnull().sum().sum()

#sayısal (sürekli verilerimiz)
continuous_features
#aykırı değer kontrol
Q1 = df[continuous_features].quantile(0.25)
Q3 = df[continuous_features].quantile(0.75)
IQR = Q3 - Q1
outliers_count_specified = ((df[continuous_features] < (Q1 - 1.5 * IQR)) | (df[continuous_features] > (Q3 + 1.5 * IQR))).sum()

print(outliers_count_specified)


#onehot kodlama
df_encoded = pd.get_dummies(df, columns=['gogus_agrı_tipi', 'Elektrokardiyografik_Ölcümü', 'talasemi'], drop_first=True)

# onehot kodlamaya ihtiyaç olmayan diğer verileri int e dönüştürme
features_to_convert = ['cinsiyet', 'Aclık_Kan_Sekeri', 'egzersize_baglı_durumu', 'egim', 'ca', 'amac']
for feature in features_to_convert:
    df_encoded[feature] = df_encoded[feature].astype(int)

print(df_encoded.dtypes)

# çarpık unsurları dönüştürme box-cox
#nitelikleri x çıktıyı y tanımladım
X = df_encoded.drop('amac', axis=1)
y = df_encoded['amac']
# datamı test eğitim böldüm
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
# st depresyonunu sabit ekleme
X_train['depresyon_ST'] = X_train['depresyon_ST'] + 0.001
X_test['depresyon_ST'] = X_test['depresyon_ST'] + 0.001

# sürekli niteliklerin dağılım kontrol
fig, ax = plt.subplots(2, 5, figsize=(15, 10))

# orjinal dağılım
for i, col in enumerate(continuous_features):
    sns.histplot(X_train[col], kde=True, ax=ax[0, i], color='#ff826e').set_title(f'Orjinal {col}')

#boxcox uygulama
# her özellik için lambda değerlerini saklayacak sözlük
lambdas = {}

for i, col in enumerate(continuous_features):
    if X_train[col].min() > 0:
        X_train[col], lambdas[col] = boxcox(X_train[col])
        # Aynı lambdayı test verilerine uygulama
        X_test[col] = boxcox(X_test[col], lmbda=lambdas[col])
        sns.histplot(X_train[col], kde=True, ax=ax[1, i], color='red').set_title(f'Dönüştürülmüş {col}')
    else:
        sns.histplot(X_train[col], kde=True, ax=ax[1, i], color='green').set_title(f'{col} (Not Transformed)')

fig.tight_layout()
plt.show()

X_train.head()


def hparam_clf(clf, param_grid, X_train, y_train, scoring='recall', n_splits=3):


    # çapraz doğrulama nesnesi
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)

    #gridsearch oluşturma
    clf_grid = GridSearchCV(clf, param_grid, cv=cv, scoring=scoring, n_jobs=-1)

    # gridsearch eğitim setine sıkıştırma
    clf_grid.fit(X_train, y_train)

    #optimal parametre
    best_hyperparameters = clf_grid.best_params_

    return clf_grid.best_estimator_, best_hyperparameters

# KNN tanımlama ve iş akışını (pipeline) oluşturma

knn_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier())
])
# ...

# KNN için hiperparmetre ızgarası
knn_param_grid = {
    'knn__n_neighbors': list(range(1, 12)),
    'knn__weights': ['uniform', 'distance'],
    'knn__p': [1, 2]  # 1: Manhattan mesafesi, 2: Öklid mesafesi
}
best_knn, best_knn_hyperparams = hparam_clf(knn_pipeline, knn_param_grid, X_train, y_train)
print('KNN Optimal Hyperparameters: \n', best_knn_hyperparams)

print(classification_report(y_train, best_knn.predict(X_train)))

print(classification_report(y_test, best_knn.predict(X_test)))

best_knn.fit(X_train,y_train)
prediction=best_knn.predict(X_test)


# Confusion matrix oluştur
conf_matrix = confusion_matrix(y_test, prediction)

# Confusion matrixi görselleştir
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Tahmin Edilen')
plt.ylabel('Gerçek')
plt.title('Confusion Matrix - kNN Modeli')
plt.show()

#df.iloc[253]
print(best_knn.named_steps['scaler'].get_feature_names_out(input_features=X_train.columns))
joblib.dump(best_knn, 'egitilmis_model.pkl')

