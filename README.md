# Hava Durumu Tahmin Modeli


## İçindekiler
- [Genel Bakış](#genel-bakış)
- [Veri Seti Detayları](#veri-seti-detayları)
- [Adımlar ve İş Akışı](#adımlar-ve-i̇ş-akışı)
  - [1. Veri Ön İşleme](#1-veri-ön-i̇şleme)
  - [2. Özellik Mühendisliği](#2-özellik-mühendisliği)
  - [3. Veri Normalizasyonu](#3-veri-normalizasyonu)
  - [4. Veri Setinin Bölünmesi](#4-veri-setinin-bölünmesi)
- [Uygulanan Modeller](#uygulanan-modeller)
  - [1. Random Forest Regressor](#1-random-forest-regressor)
  - [2. LightGBM Regressor](#2-lightgbm-regressor)
  - [3. Support Vector Regressor (SVR)](#3-support-vector-regressor-svr)
- [MultiOutput Regressor](#multioutput-regressor)
- [Hiperparametre Ayarı](#hiperparametre-ayarı)
- [Örnek Model](#örnek-model)
- [Sonuçlar ve Karşılaştırma](#sonuçlar-ve-karşılaştırma)
- [Örnek Kullanım](#örnek-kullanım)

## Genel Bakış

Bu proje, bir günün her saati için günlük hava durumu parametrelerini (Sıcaklık, Nem, Bulut Kapalılığı, Güneşlenme Süresi ve Basınç) tahmin etmek üzere bir makine öğrenimi modeli oluşturmayı amaçlamaktadır. Ana hedef, geçmiş veriler ve döngüsel mevsimsel desenlere dayanarak doğru tahminler yapabilen sağlam bir sistem geliştirmektir.



## Veri Seti Detayları

Veri seti, [meteoblue](https://www.meteoblue.com/tr/hava/archive/export?daterange=2023-01-01%20-%202024-11-25&locations%5B%5D=basel_%25c4%25b0svi%25c3%25a7re_2661604&domain=ERA5T&params%5B%5D=&params%5B%5D=&params%5B%5D=&params%5B%5D=&params%5B%5D=&params%5B%5D=&params%5B%5D=&params%5B%5D=&utc_offset=%2B00%3A00&timeResolution=daily&temperatureunit=CELSIUS&velocityunit=KILOMETER_PER_HOUR&energyunit=watts&lengthunit=metric&degree_day_type=10%3B30&gddBase=10&gddLimit=30) sayfasının ücretsiz olarak Geçmiş Hava Durumu Verilerini İndirme sayfasından alınmıştır. Aşağıdaki sütunları içeren saatlik hava durumu verilerini içermektedir:

- **Time: Zaman bilgisi (Y/A/G)
- **Wind_Speed: Rüzgar hızı (km/h)
- **Wind_Direction: Rüzgar yönü (°)
- Temperature: Saatlik sıcaklık (°C).
- Humidity: Göreceli nem oranı (%).
- Cloud_Cover: Bulut kapalılığı oranı (%).
- Sunshine_Duration: Güneşlenme süresi (saat).
- Pressure: Atmosfer basıncı (hPa).
- *Year, Month, Day, Hour: Tarih bilgisinden türetilen zaman bileşenleri.
- *Season_Spring, Season_Summer, Season_Winter: One-hot kodlanmış mevsimsel göstergeler.
- *Hour_Sin, Hour_Cos: Saatlerin döngüsel kodlaması.
- *Month_Sin, Month_Cos: Ayların döngüsel kodlaması.
- *Lag Özellikleri: Geçmiş sıcaklık, nem, bulut kapalılığı, güneşlenme süresi ve basınç değerleri.
- *Last24_Mean, Last24_Max, Last24_Min: Son 24 saatin ortalama, maksimum ve minimum değerleri.

>  "*" ile işaretli olan satırlar veri işleme sonrası dahil edilmiş sütunları ifade eder. 
>
> "**" ile işaretli olan satırlar verisetinin orojinalinde olan fakat daha sonra çıkarılan sütunları ifade eder. 




## Adımlar ve İş Akışı

1. **Veri Ön İşleme**

    1. Gereksiz Satırları Silme:

        - Çoğaltılmış veya ilgisiz bilgiler içeren satırlar kaldırıldı.
  
            ```
            data1_cleaned = data1.iloc[9:].reset_index(drop=True)
            ```

    2. Sütun İsimlerini Yeniden Adlandırma:

       - Tutarlılık ve netlik sağlamak için sütun adları standardize edildi.
         ```
         data1_cleaned.rename(columns={
            'location': 'Time',
            'Basel': 'Temperature',
            'Basel.1': 'Humidity',
            'Basel.2': 'Wind_Speed',
            'Basel.3': 'Wind_Direction',
            'Basel.4': 'Cloud_Cover',
            'Basel.5': 'Sunshine_Duration',
            'Basel.6': 'Pressure'
         }, inplace=True)
         ```

    3. Eksik Değerlerin İşlenmesi:

        - Eksik değerler `SimpleImputer` ile ilgili sütunların ortalaması ile dolduruldu.
          ```
          print(data1_cleaned.isnull().sum())

          from sklearn.impute import SimpleImputer

          imputer = SimpleImputer(strategy='mean')
            
          data1_cleaned[['Temperature', 'Humidity', 'Wind_Speed', 'Wind_Direction', 'Cloud_Cover', 'Sunshine_Duration', 'Pressure']] = imputer.fit_transform(data1_cleaned[['Temperature', 'Humidity', 'Wind_Speed', 'Wind_Direction', 'Cloud_Cover', 'Sunshine_Duration', 'Pressure']])

          ```

    4. Veri Setlerini Birleştirme:

        - Birden fazla veri kaynağı kullanıldığı için `pd.concat(,axis=0)` ile veri setleri birleştirildi.
          ```
          dataset = pd.concat([data1_cleaned, dataset2_cleaned], axis=0)
          ```

    5. Korelasyon Analizi:

        - Korelasyon matrisi görselleştirilerek değişkenler arası ilişkiler incelendi. (Time eğişkeni sayısal bir sütun olmadığı için çıkartılarak kolerasyon matrisi oluşturuldu.)

        ```
        # Sadece sayısal sütunları seçmek için tarih sütununu çıkarıyoruz
        numeric_columns = dataset.drop(columns=dataset.columns[0])

        corr_matrix = numeric_columns.corr()

        plt.figure(figsize=(10, 6))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
        plt.title("Korelasyon Matrisi")
        plt.show()
        ```
        ![image](https://github.com/user-attachments/assets/6a8875d6-efa3-42c0-9fb2-ec7bc5bd905f)


        - Hedef değişkenle düşük korelasyona sahip sütunlar (Wind_Speed, Wind_Direction) kaldırıldı.

    6. Zaman Bileşenlerini Çıkartma:

        - Time sütunu parçalanarak Year, Month, Day ve Hour sütunları oluşturuldu.
        ```
        df['Year'] = df['Time'].str[:4].astype(int)
        df['Month'] = df['Time'].str[4:6].astype(int)
        df['Day'] = df['Time'].str[6:8].astype(int)
        df['Hour'] = df['Time'].str[9:11].astype(int)
        ```

        - Orijinal Time sütunu kaldırıldı.
        ```
        df.drop('Time', axis=1, inplace=True)
        ```


    7. Mevsimsel Özelliklerin Eklenmesi:
   
        - Month sütunu bilgilerine dayanarak Season sütunu oluşturuldu ve "Winter, Spring, Summer, Autumn" bilgileri ile dolduruldu.

        ```
        df['Season'] = df['Month'].apply(lambda x: 'Winter' if x in [12, 1 , 2] else
                                                   'Spring' if x in [3, 4, 5] else
                                                   'Summer' if x in [6, 7, 8] else 'Autumn')
        ```

        - Season sütunu one-hot encoding (`pd.get_dummies(,,drop_first=True)`) yöntemiyle kodlandı. 
          


    8. Histogram ve Boxplot Grafikleri: 
        - Tüm sayısal sütunlar için histogram grafikleri oluşturuldu. Amaç: Değişkenlerin dağılımlarını ve olası simetrik olmayan desenleri görmek.
          ```
          for column in ['Temperature',	'Humidity',	'Cloud_Cover',	'Sunshine_Duration',	'Pressure']:
                plt.figure(figsize=(8, 4))
                sns.histplot(df[column], kde=True)
                plt.title(f"{column} Dağılımı")
                plt.show()
          ```
            ![image](https://github.com/user-attachments/assets/f54b443f-7891-425e-bba6-fd863e970415)
            ![image](https://github.com/user-attachments/assets/69cfd4e8-643f-447d-990a-7118742e6ea9)
            ![image](https://github.com/user-attachments/assets/c847c442-cff3-4acb-81b4-64e6a45c666f)
            ![image](https://github.com/user-attachments/assets/ab3c67b9-f7fb-4ab6-be71-d6bd886b9589)
            ![image](https://github.com/user-attachments/assets/0ad1fc39-42a9-4520-b979-a8e8b26c4e1d)


        - Boxplot grafikleri kullanılarak aykırı değerler tespit edildi Özellikle: Temperature, Humidity, Cloud_Cover, Sunshine_Duration ve Pressure sütunlarında belirgin aykırı değerler bulundu. 
          
          ```
          for column in ['Temperature',	'Humidity',	'Cloud_Cover',	'Sunshine_Duration',	'Pressure']:
                plt.figure(figsize=(8, 4))
                sns.boxplot(df[column])
                plt.title(f"{column} için Boxplot")
                plt.show()
          ```
            ![image](https://github.com/user-attachments/assets/0063a315-b0f3-40f5-893b-3d837832aca3)
            ![image](https://github.com/user-attachments/assets/78905658-66af-4d43-b22c-e40bb1c56ca4)
            ![image](https://github.com/user-attachments/assets/bdb1fc6b-1fd0-460a-bdc0-35942bbceb86)
            ![image](https://github.com/user-attachments/assets/88456032-f5c5-404d-a76e-8a22883577bf)
            ![image](https://github.com/user-attachments/assets/1b0ed5bf-e4a6-4c23-afef-12e5f79dce9d)


            - Aynı zamanda veriler mevsimsel olarak normali değiştiği için mevsimsel boxplot grafiklerine de bakıldı. Örneğin;
             ```
             df["Season_Autumn"] = 1 - (df["Season_Spring"] + df["Season_Summer"] + df["Season_Winter"])

             df["Season"] = df[["Season_Spring", "Season_Summer", "Season_Winter", "Season_Autumn"]].idxmax(axis=1)

             # İşlenecek sütunlar
             columns_to_plot = ["Temperature", "Humidity", "Cloud_Cover", "Sunshine_Duration", "Pressure"]

             # Her sütun için boxplot oluştur
             for column in columns_to_plot:
                plt.figure(figsize=(8, 6))
                sns.boxplot(x="Season", y=column, data=df)
                plt.title(f"Mevsimlere Göre {column} Dağılımı")
                plt.xlabel("Mevsim")
                plt.ylabel(column)
                plt.show()
             ```
             ![image](https://github.com/user-attachments/assets/df3a3114-8b2d-4b0b-89af-5b50ca60ad76)


2. **Özellik Mühendisliği**

    1. Döngüsel Özellikler:

        - Döngüsel özellik gösteren ``Hour``, `Day` ve ``Month`` için sinüs ve kosinüs dönüşümleri eklendi.
          ```
          df['Hour_Sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
          df['Hour_Cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
          df['Month_Sin'] = np.sin(2 * np.pi * df['Month'] / 12)
          df['Month_Cos'] = np.cos(2 * np.pi * df['Month'] / 12)
          df['Day_Sin'] = np.sin(2 * np.pi * df['Day'] / 7)
          df['Day_Cos'] = np.cos(2 * np.pi * df['Day'] / 7)
          ```
         
    2. Lag Özellikleri:

        - ``Temperature``, ``Humidity``, ``Cloud_Cover``, ``Sunshine_Duration`` ve ``Pressure`` için geçmiş (1 saatlik, 2 saatlik, 3 saatlik) lag özellikleri tanımlandı.
          ```
          df['Temperature_Lag1'] = df['Temperature'].shift(1)
          df['Temperature_Lag2'] = df['Temperature'].shift(2)
          df['Temperature_Lag3'] = df['Temperature'].shift(3)

          # İlk birkaç satır NaN oldu
          df.dropna(inplace=True)
          ```
    3. İstatistiksel Özetler:

        - ``Temperature``, ``Humidity``, ``Cloud_Cover``, ``Sunshine_Duration`` ve ``Pressure`` için son 24 saate ait ortalama, minimum ve maksimum değerler hesaplandı.
          ```
          df['Last24_Mean'] = df['Temperature'].rolling(window=24).mean()
          df['Last24_Max'] = df['Temperature'].rolling(window=24).max()
          df['Last24_Min'] = df['Temperature'].rolling(window=24).min()

          df.fillna(method='bfill', inplace=True)
          ```

    4. Aykırı Değer Tespiti:

        - Aykırı değerler, mevsimsel ve saatlik dağılımlara göre `IQR` yöntemi ile analiz edildi.
        - Uç değerler yerine mevsimsel ortalama değerler atanarak dengeli bir veri seti oluşturuldu.
            - Örneğin, kışın ölçülen aşırı yüksek sıcaklıklar yerine o mevsim için makul bir değer atanmıştır.
                ```
                def detect_outliers(series):
                    Q1 = series.quantile(0.25)
                    Q3 = series.quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    return (series < lower_bound) | (series > upper_bound)

                
                seasons = ['Season_Spring', 'Season_Summer', 'Season_Winter', 'Season_Autumn']

                columns_to_process = ['Temperature', 'Humidity', 'Cloud_Cover', 'Sunshine_Duration', 'Pressure']

                max_iterations = 5  # Maksimum tekrar sayısı
                for column in columns_to_process:
                    for season in seasons:
                        for iteration in range(max_iterations):
                            season_mask = df[season] == 1

                            season_data = df.loc[season_mask, column]

                            outliers = detect_outliers(season_data)

                            # Uç değer yoksa dur
                            if outliers.sum() == 0:
                                break

                            season_mean = season_data[~outliers].mean()

                            df.loc[season_mask & outliers, column] = season_mean
                ```

3. **Veri Normalizasyonu** 

    1. Standardizasyon:

        - Sayısal özellikler `StandardScaler` ile ölçeklenerek model performansı artırıldı.
            ```
            from sklearn.preprocessing import StandardScaler

            scaler = StandardScaler()

            # Her mevsim için standartlaştırma işlemi
            for season in seasons:
                season_data = df[df[season] == 1]
                season_columns = season_data[columns_to_process]
                standardized_data = scaler.fit_transform(season_columns)

                df.loc[df[season] == 1, columns_to_process] = standardized_data
            ```
4. **Veri Setinin Bölünmesi**

    1. Random Bölme:

        - Veri seti %80 eğitim ve %20 test verisi olacak şekilde rastgele bölündü.
            ```
            from sklearn.model_selection import train_test_split

            # Özellik ve hedef değişkenleri ayırma
            X = df[['Day_Sin', 'Day_Cos', 'Season_Spring', 'Season_Summer', 'Season_Winter', 'Hour_Sin', 'Hour_Cos', 'Month_Sin', 'Month_Cos',
                    'Temperature_Lag1', 'Temperature_Lag2', 'Temperature_Lag3', 'Last24_Mean', 'Last24_Max', 'Last24_Min']]
            y = df[['Temperature', 'Humidity', 'Cloud_Cover', 'Sunshine_Duration', 'Pressure']]

            # Veri setini eğitim ve test olarak ayırma
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            ```
    2. Zaman Tabanlı Bölme:

        - Zaman serisi analizine uygun bir şekilde sıralı olarak geçmiş verilere dayalı bölme yapıldı.
            ```
            df = df.sort_values(by='Year')

            train_size = int(len(df) * 0.8)

            # Eğitim ve test setlerini ayırın
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]
            ```
    3. Zaman Serisi Çapraz Doğrulama:

        - Daha sağlam bir değerlendirme için kademeli zaman serisi çapraz doğrulama kullanıldı.
            ```
            from sklearn.model_selection import TimeSeriesSplit
            # 5 katlı çapraz doğrulama
            tscv = TimeSeriesSplit(n_splits=5)  
            for train_index, test_index in tscv.split(X):
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            ```
## Uygulanan Modeller

   1. Random Forest Regressor

        - Birden fazla karar ağacı kullanan sağlam bir topluluk öğrenme yöntemi.

        - Ayarlanan hiperparametreler:

            - n_estimators: 100

            - max_depth: 20

            - min_samples_split: 5

            - min_samples_leaf: 2

   2. LightGBM Regressor

        - Hız ve verimlilik için optimize edilmiş bir gradyan artırma çerçevesi.

        - Ayarlanan hiperparametreler:

            - num_leaves: 40

            - learning_rate: 0.05

            - max_depth: 7

            - n_estimators: 200
            - min_child_samples: 30

   3. Support Vector Regressor (SVR)

        - Doğrusal olmayan ilişkileri modellemek için destek vektör makinelerini kullanan bir yöntem.

        - Ayarlanan hiperparametreler: 

            - kernel: linear

            - C: 0.1

            - epsilon: 0.3



## MultiOutput Regressor

``MultiOutputRegressor`` ile birden fazla hedef değişken aynı anda tahmin edilmiştir:

- Temperature

- Humidity

- Cloud_Cover

- Sunshine_Duration

- Pressure

## Hiperparametre Ayarı

Grid Search ile Çapraz Doğrulama: Her model için optimum hiperparametreler GridSearchCV kullanılarak belirlendi.

- Ölçüt: Negatif Ortalama Kare Hatası (``neg_mean_squared_error``).

- Yapılandırma: Çapraz doğrulama katman sayısı (cv): 3

<br/><br/>
---

## Örnek Model;

```
param_grid = {
    'base_estimator__learning_rate': [0.01, 0.05, 0.1],  # Öğrenme hızı
    'base_estimator__n_estimators': [100, 200, 500],     # Ağaç sayısı
    'base_estimator__max_depth': [3, 5, 7],              # Maksimum derinlik
    'base_estimator__num_leaves': [20, 31, 40],          # Yaprak sayısı
    'base_estimator__min_child_samples': [10, 20, 30]    # Minimum yaprak örnek sayısı
}

base_lgbm = LGBMRegressor(random_state=42)

regressor_chain = RegressorChain(base_estimator=base_lgbm, order='random', random_state=42)

grid_search = GridSearchCV(estimator=regressor_chain, param_grid=param_grid, cv=3, scoring='r2', n_jobs=-1)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
best_params = grid_search.best_params_
print("En İyi Parametreler:", best_params)

y_pred_lgb = best_model.predict(X_test)

# genel
mse = mean_squared_error(y_test, y_pred_lgb)
r2 = r2_score(y_test, y_pred_lgb)

print(f"Genel MSE: {mse:.4f}")
print(f"Genel R2 Score: {r2:.4f}")

for i, column in enumerate(y.columns):
    mse_i = mean_squared_error(y_test.iloc[:, i], y_pred_lgb[:, i])
    r2_i = r2_score(y_test.iloc[:, i], y_pred_lgb[:, i])
    print(f"\nHedef Değişken: {column}")
    print(f"MSE: {mse_i:.4f}")
    print(f"R2 Score: {r2_i:.4f}")

```

---
<br/><br/>

## Sonuçlar ve Karşılaştırma

Modeller bir çok hiperparametre ile birçok kez eğitildi ve aşağıda gösterilen kriterlere göre değerlendirildi:

- Ortalama Kare Hatası (MSE)

- Ortalama Mutlak Hata (MAE)

- R-kare (R²)

Random Forest, verilerdeki doğrusal olmayan ilişkileri yakalamada en iyi performansı gösterdi.

LightGBM, daha hızlı eğitim süresiyle rekabetçi sonuçlar sağladı.

SVR, büyük özellik seti nedeniyle daha yavaş ve daha az hassas oldu.


Genel;

| - | Random Forest Regressor  | LightGBM Regressor | Support Vector Regressor (SVR) |
| -- | -- | -- | -- |
| R² | 12 | 0.9 | 12 |
| MSE | 12 | 0.1 | 12 |
| MAE | 12 | 0.2 | 12 |


Hedef Değişkenlere Göre;



## Örnek Kullanım

Girdi: Kullanıcı bir tarih girer (örneğin, 2024-03-28).

Çıktı: Sistem, o günün tüm saatleri için hava durumu parametrelerini tahmin eder.



