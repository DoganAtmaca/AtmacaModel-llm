# AtmacaModel-llm
PyTorch ile sıfırdan inşa edilmiş Türkçe Hukuk odaklı Mikro-LLM mimarisi.
# 🦅 Atmaca LLM — Sıfırdan (From-Scratch) Türkçe Yapay Zeka Mimarisi

Bu proje, modern Büyük Dil Modellerinin (Llama-3 ve Qwen gibi sistemlerin) arka planda nasıl çalıştığını göstermek amacıyla, hiçbir hazır model şablonu kullanılmadan **saf PyTorch** ile sıfırdan inşa edilmiş bir Türkçe dil modeli çekirdeğidir (`atmaca_model.py`).

Mimari, tek bir tüketici ekran kartı (**NVIDIA RTX 4070 Ti 12GB**) üzerinde en yüksek hızda çalışacak şekilde optimize edilmiş ve Türk Hukuku metinleri üzerinde eğitilerek test edilmiştir.

---

## 🛠️ Kullanılan Teknolojiler Nelerdir?

* **Saf PyTorch (`torch` & `torch.nn`):** Modelin içindeki tüm yapay sinir ağı katmanları, matris çarpımları ve dikkat (attention) mekanizmaları dışarıdan hazır bir kütüphane (`transformers` vb.) çağrılmadan sıfırdan kodlanmıştır.
* **Donanım Dostu Matris Tasarımı (FlashAttention & Tensor Core Uyumu):** Modelin içindeki tüm sayı boyutları (`512`, `64`, `1408`, `16384`), ekran kartının (GPU) işlemleri tek seferde ve en hızlı şekilde yapabilmesi için `64` ve `128`'in tam katları olarak tasarlanmıştır.
* **KV-Cache (Hafıza Önbellekleme):** Model yeni bir kelime üretirken, cümlenin başındaki kelimeleri her seferinde tekrar tekrar hesaplamak yerine hafızasında (`past_key_value`) tutar; böylece çok daha hızlı metin üretir.
* **Esnek Ölçeklenebilirlik (`Preset` Sistemi):** Tek bir kod üzerinden donanım gücüne göre **25 Milyon**, **35 Milyon**, **40 Milyon** veya **60 Milyon** parametrelik farklı büyüklüklerde modeller kurulabilir.

---

## 🧩 Kod Mimarisi: Hangi Sınıf Ne İş Yapar? (`atmaca_model.py`)

Modelin kodu, bir cümlenin okunup sıradaki kelimenin tahmin edilmesine kadar geçen süreci **8 ana parçada** yönetir:

1. **`AtmacaConfig` (Modelin Kimlik Kartı ve Ayar Merkezi):** Modelin kaç kelime bildiğini (`16.384`), tek seferde ne kadar uzunlukta bir metni okuyabildiğini (`512` token) ve beynin kaç katmandan oluşacağını belirleyen kontrol merkezidir.
2. **`AtmacaRMSNorm` (Sayı Dengeleyici / Voltaj Regülatörü):** Kelimeler katmanlardan geçip birbiriyle çarpıldıkça içlerindeki sayılar aşırı büyür; bu sınıf sayıların oranını bozmadan hepsini güvenli bir seviyeye dengeler.
3. **`AtmacaRotaryEmbedding` ve `apply_rotary_pos_emb` (Kelime Sırası ve Konum Saati - RoPE):** Her kelimeyi cümledeki sırasına göre belirli bir açıyla döndürür; böylece model kelimelerin sırasını ve birbirine olan mesafesini anlar.
4. **`AtmacaAttention` (Kelimelerin Konuştuğu Toplantı Odası):** Bir kelimenin cümledeki önceki kelimelere bakarak hangisinden ne kadar anlam alacağına karar verdiği ve gelecekteki kelimelerin maskeleme ile gizlendiği katmandır.
5. **`belge_izolasyon_maskesi` (Metinler Arası Güvenlik Duvarı):** Eğitim sırasında 512 tokenlık pencerelerde `<|endoftext|>` sınırından önceki belgenin sonraki belgeye attention sızdırmasını engelleyen `[B, 1, T, T]` boyutlu özel koruma maskesidir.
6. **`AtmacaMLP` (Bilgiyi Sindirme ve Düşünme Odası - SwiGLU):** Toplantı odasında diğer kelimelerden bilgi toplayan her kelimenin, önemli bilgileri geçirip gereksiz bilgileri filtreleyen bir kapı (`SiLU`) mekanizmasıyla anlamını netleştirdiği katmandır.
7. **`AtmacaDecoderLayer` ve `AtmacaModel` (Çok Katlı Ana Beyin):** `AtmacaDecoderLayer` tek bir beyin katmanını, `AtmacaModel` ise üst üste dizilen tüm katmanları (`ModuleList`) ve kelime tablosunu (`embed_tokens`) barındıran ana gövdeyi oluşturur.
8. **`AtmacaForCausalLM` (Sıradaki Kelimeyi Tahmin Eden Dış Kabuk):** En üst kattan çıkan anlamı alır, `16.384` kelimenin tamamına olasılık puanı (`logits`) verir ve giriş kelime tablosu ile çıkış tahmin tablosunu ortak kullanır (`Weight Tying`).

---

## 📊 Model Hiperparametreleri ve Ölçek Tablosu

* **Sözlük Boyutu (`vocab_size`):** `16,384` (Özel Türkçe Byte-Level BPE)
* **Maksimum Bağlam Uzunluğu (`context_length`):** `512` Token
* **Pozisyon Kodlama (`rope_theta`):** RoPE (`10000.0` tabanlı dinamik önbellekli)
* **Normalizasyon (`rms_norm_eps`):** Pre-Norm `AtmacaRMSNorm` (`1e-6`)
* **Aktivasyon ve MLP:** `SwiGLU` (`SiLU` kapı filtreli)
* **Ağırlık Paylaşımı (`Weight Tying`):** Aktif (`embed_tokens` ve `lm_head` ortak matris kullanır)

| Ölçek (Preset) | Toplam Parametre | Katman (`layers`) | Gizli Boyut (`hidden_size`) | Kafa (`heads`) | Kafa Boyutu (`head_dim`) | MLP Ara Boyut (`intermediate`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`25m`** | ~25.2M | 8 | 384 | 6 | 64 | 1536 |
| **`35m`** *(Önerilen)* | ~34.1M | 8 | 512 | 8 | 64 | 1408 |
| **`40m`** *(Varsayılan)* | ~40.5M | 10 | 512 | 8 | 64 | 1408 |
| **`60m`** | ~57.5M | 12 | 576 | 9 | 64 | 1536 |

*(Tablo değerleri `AtmacaConfig` sınıfındaki tanımlamalara dayanmaktadır.*

---

## 🧪 Gerçek Model Çıktıları ve Şeffaf Hata Analizi (`sicaklik=0.0`)

Aşağıdaki çıktılar, ~35M-40M parametre ölçeğinde eğitilen modelin hiçbir seçme (cherry-picking) yapılmadan ürettiği ham yanıtlarıdır:

### 1. Tam İsabetli Hukuki Tanımlar

* **Soru:** `iddianame nedir`
  * **Atmaca:** *"İddianame, ceza muhakemesinde Cumhuriyet savcısının soruşturma evresi sonunda topladığı delillerin suçun işlendiği hususunda yeterli şüphe oluşturması halinde kamu davası açmak amacıyla düzenleyip görevli ve yetkili mahkemeye sunduğu resmi belgedir (CMK m. 170). İddianamede şüphelinin kimliği, yüklenen suç, uygulanması istenen kanun maddeleri ve mevcut deliller gösterilir. Mahkemenin iddianameyi kabul etmesiyle soruşturma evresi biter, kovuşturma (mahkeme) evresi başlar ve şüpheli sanık sıfatını alır."*
  * **🔍 Analiz:** ✅ **Tam Doğru.** Kanun maddesi atfı (CMK m. 170), evre geçişi ve şüpheli/sanık sıfat değişimi doğru ifade edilmiştir.

* **Soru:** `tck`
  * **Atmaca:** *"TCK, 5237 sayılı Türk Ceza Kanunu'nun kısaltmasıdır. Hangi fiillerin suç teşkil ettiğini, ceza sorumluluğunun genel esaslarını (kast, taksir, teşebbüs, iştirak, meşru savunma) ve suçlara verilecek hapis veya adli para cezalarını düzenleyen temel maddi ceza kanunudur. Kişi hak ve özgürlüklerini, kamu düzenini ve toplum barışını korumayı amaçlar."*
  * **🔍 Analiz:** ✅ **Tam Doğru.** Tek kelimelik küçük harfli girdi doğru algılanmış ve kanunun kapsamı tanımlanmıştır.

* **Soru:** `HAGB nedir?`
  * **Atmaca:** *"Hükmün açıklanmasının geri bırakılması (HAGB), sanığa verilen 2 yıl veya daha az süreli hapis ya da adli para cezasının belirli şartlarla (kasıtlı suçtan sabıkasızlık, zararın giderilmesi ve olumlu kanaat) hukuki sonuç doğurmamasını sağlayan kurumdur (CMK m. 231). Sanık 5 yıllık denetim süresinde kasıtlı suç işlemezse dava düşer ve sabıka kaydına işlemez. HAGB kararına karşı iki hafta içinde istinaf yoluna başvurulabilir ve karar verilmesi için sanığın kabulü şartı aranmaz."*
  * **🔍 Analiz:** ✅ **Tam Doğru.** CMK m. 231 kapsamındaki istinaf yolu ve sanığın kabul şartının aranmaması detayı doğru üretilmiştir.

* **Soru:** `Tereke ne demek?`
  * **Atmaca:** *"Tereke, vefat eden kişinin (mirasbırakanın / murisin) ölümüyle geride bıraktığı ve mirasçılarına intikal eden para, taşınır mal, taşınmaz mal, alacak ve borçların tamamından oluşan malvarlığı bütünüdür (TMK m. 599). Kişiye sıkı sıkıya bağlı haklar terekeye dahil olmaz."*
  * **🔍 Analiz:** ✅ **Tam Doğru.** TMK m. 599 atfı ve istisna kuralı net biçimde üretilmiştir.

### 2. Kısmi Doğrular ve Kavram Kaymaları (Kapasite Sınırları)

* **Soru:** `TCK ile CMK arasındaki fark nedir?`
  * **Atmaca:** *"Türk Ceza Kanunu (TCK) hangi eylemlerin suç olduğunu ve bu suçlara ne kadar ceza verileceğini belirleyen maddi ceza hukuku kanunudur. Ceza Muhakemesi Kanunu (CMK) ise bir suç işlendiği iddia edildiğinde soruşturmanın ve mahkeme yargılamasının nasıl yapılacağını, delillerin nasıl toplanacağını, sanık ile mağdurun usuli haklarını, delillerle ispatlanabileceklerini, suçun sübutuna, kamu görevlisine karşı, hak aramayı, o anki haklarına uygun ve açık bir şekilde savunması alındıktan sonra yargılamanın adil ve şeffaf bir şekilde yürütülmesini sağlar."*
  * **🔍 Analiz:** ⚠️ **Kısmi Başarı.** Maddi ceza hukuku ile usul hukuku ayrımı başta doğru yapılmış, ancak uzun cümle sonunda bağlam dağılmıştır.

* **Soru:** `Rüşvet suçu (TCK m. 252) hangi unsurlarla oluşur?`
  * **Atmaca:** *"TCK m. 252'ye göre rüşvet, bir kamu görevlisinin görevinin ifasıyla ilgili bir işi yapması veya yapmaması için doğrudan ya da aracılar vasıtasıyla kendisi ya da göstereceği bir başka kişi ile birlikte hareket etmesi halinde cezalandırılır. Rüşvette kamu görevlisi, görevi nedeniyle zilyetliği kendisine devredilen bir malı, kendi veya başkasının yararına sahipliğine geçirir."*
  * **🔍 Analiz:** ⚠️ **Kavram Birleşmesi.** İlk cümlede rüşvet suçu tanımlanırken, ikinci cümlede zimmet suçunun tanımına kayma gerçekleşmiştir

* **Soru:** `Savcılığın verdiği takipsizlik (KYOK) kararına nereye ve ne kadar sürede itiraz edilir?`
  * **Atmaca:** *"Savcılığın verdiği takipsizlik kararının kesinleşmesinden itibaren iki hafta içinde, Cumhuriyet Savcısının itirazına rağmen, itirazı kabul etmemesine rağmen, Cumhuriyet Savcısı tarafından verilen kovuşturmaya yer olmadığına dair karara karşı, kararın kendisine tebliğinden itibaren iki hafta içinde Sulh Ceza Hakimliği'ne itiraz edilebilir. İtirazın iptali davası ise, Cumhuriyet Savcısının talebi üzerine Yargıtay'a yapılır."*
  * **🔍 Analiz:** ⚠️ **Kısmi Doğru + Halüsinasyon.** Tebliğden itibaren iki hafta içinde Sulh Ceza Hakimliği'ne itiraz bilgisi doğru verilmiş, ancak cümle sonunda icra hukukundaki "itirazın iptali" kavramı karıştırılmıştır.

* **Soru:** `Doğrudan kast ile olası kast arasındaki fark nedir?`
  * **Atmaca:** *"Doğrudan kast ile olası kast arasındaki temel fark, failin suçun kanuni tanımındaki unsurları bilerek ve neticeyi isteyerek gerçekleştirmesidir. Olası kastta ise fail neticenin gerçekleşebileceğini öngörür fakatistemez ve neticenin gerçekleşmesini istemez. Bilinçli taksirde ise fail neticenin gerçekleşebileceğini öngörüp kabullenir; ancak bu durumda fail neticenin gerçekleşebileceğini öngörür."*
  * **🔍 Analiz:** ⚠️ **Tanım Yer Değiştirmesi.** Doğrudan kast doğru tanımlanmış, ancak olası kast ile bilinçli taksir tanımları yer değiştirmiştir.

* **Soru:** `İlamsız icra takibinde ödeme emrine itiraz süresi kaç gündür?`
  * **Atmaca:** *"İlamsız icra takibinde ödeme emrine karşı yapılan itiraz üzerine takip kendiliğinden durur. İtirazın kaldırılması veya iptali davası, alacaklının itirazın kaldırılması veya iptali davaları için açılır."*
  * **🔍 Analiz:** ⚠️ **Eksik Bilgi.** İtirazın takibi durdurma etkisi belirtilmiş ancak sorulan süre yanıt içinde atlanmıştır.

### 3. Muhakeme Hataları ve Aşırı Hukuk Şartlanması

* **Soru:** `Merhaba, sen kimsin ve bana hangi konularda yardımcı olabilirsin?`
  * **Atmaca:** *"Merhaba, ben Atmaca. Türk hukuku, yürürlükteki temel kanunlar ve yargı süreçleri konularında bilgi vermek üzere eğitilmiş Atmaca. Türk hukuku, yürürlükteki temel kanunlar ve yargı süreçleri konusunda bilgi vermek üzere eğitilmiş Atmaca."*
  * **🔍 Analiz:** 🔄 **Döngüsel Tekrar.** Kimlik bilgisi doğru öğrenilmiş ancak kısa tanım cümlesi iki kez tekrarlanmıştır.

* **Soru:** `Kira sözleşmesiyle aynı gün imzalanan boş tahliye taahhütnamesi geçerli midir?`
  * **Atmaca:** *"Kira sözleşmesiyle aynı gün imzalatılan tahliye taahhütnamesine dayanılarak yenilenen kira dönemlerinde uygulanacak kira sözleşmelerinde, her iki durumda da hüküm ifade eder. Kiracının bu taahhütnamede belirtilen süre içinde hiçbir gerekçe göstermeksizin ve cezai şart ödemeden sözleşmeyi feshetme hakkı bulunmaktadır."*
  * **🔍 Analiz:** ❌ **Hatalı Muhakeme.** Küçük model burada kira fesih kurallarıyla tahliye taahhütnamesi şartlarını birbirine karıştırmıştır.

* **Soru:** `İş yerinde 3 yıldır çalışıyorum, kendi isteğimle istifa edersem kıdem tazminatı alabilir miyim?`
  * **Atmaca:** *"İş Kanunu kapsamında kıdem tazminatına hak kazanabilmek için işçinin en az 6 ay önce bildirim yapılması zorunludur. İşçinin haklı nedene dayanmayan feshi, emeklilik, muvazzaf askerlik, kadın işçinin evliliği veya işverenin haklı nedenle feshi gibi özel sebepler ile kıdem tazminatı talep etme hakkı bulunmaktadır. Bu şartların yerine getirilmemesi durumunda kıdem tazminatı hakkı doğmaz."*
  * **🔍 Analiz:** ❌ **Sayısal Halüsinasyon.** İstisnalar sıralanmış ancak "en az 6 ay önce bildirim" şeklinde hatalı bir şart üretilmiştir.
