# Deep Learning in the cloud


## Microsoft Azure

$200 einmaliges kostenloses Kontingent

[Deep Learning Virtual Machine](https://azuremarketplace.microsoft.com/en-us/marketplace/apps/microsoft-ads.dsvm-deep-learning)

based on [Data Science Virtual Machine](https://azuremarketplace.microsoft.com/de-DE/marketplace/apps/microsoft-ads.linux-data-science-vm-ubuntu)

__Cost__: 

|Instanz|Kategorie|Kerne|RAM|Festplattenspeicher|Laufwerktyp|Infrastrukturkosten|Softwarekosten|Stündlich|Monatlich|
|--- |--- |--- |--- |--- |--- |--- |--- |--- |--- |
|DS2v2*|Allgemein|2|7GB|14GB|SSD|0,146 $|Kostenlos|0,146 $|108,624 $|
|DS3v2*|Allgemein|4|14GB|28GB|SSD|0,293 $|Kostenlos|0,293 $|217,992 $|
|DS4v2*|Allgemein|8|28GB|56GB|SSD|0,585 $|Kostenlos|0,585 $|435,24 $|
|NC6|GPU|6|56GB|380GB|HDD|0,90 $|Kostenlos|0,90 $|669,60 $|
|NC12|GPU|12|112GB|680GB|HDD|1,80 $|Kostenlos|1,80 $|1.339,20 $|
|NC24|GPU|24|224GB|1440GB|HDD|3,60 $|Kostenlos|3,60 $|2.678,40 $|



## Amazon AWS

[Deep Learning AMI](https://aws.amazon.com/de/machine-learning/amis/)


__Cost__: 

|Name|GPUs|vCPUs|RAM (GiB)|Netzwerk Bandbreite|Preis/Stunde*|RI-Preis/Stunde**|
|--- |--- |--- |--- |--- |--- |--- |
|p2.xlarge|1|4|61|Hoch|0,900 USD|0,425 USD|
|p2.8xlarge|8|32|488|10 GBit/s|7,200 USD|3,400 USD|
|p2.16xlarge|16|64|732|20 GBit/s|14,400 USD|6,800 USD|


## Google Cloud

__Cost__

[Google MAchine Learning Engine](https://cloud.google.com/ml-engine/) is a not a image that you install, but rather a higher level service to train models.

_Scaling_

|||
|--- |--- |
|BASIC|0,2774 $ (0,5661)|
|STANDARD_1|2,9025 $ (5,9234)|
|PREMIUM_1|24,1683 $ (49,3230)|
|BASIC_GPU|1,2118 $ (2,4731)|
|BASIC_TPU (Beta)|9,7674 $ (19,9335)|
|CUSTOM|Wenn Sie CUSTOM als Skalierungsstufe auswählen, können Sie die Anzahl und den Typ der virtuellen Maschinen festlegen, die für den Trainingsjob verwendet werden. Mögliche Maschinentypen finden Sie in der Tabelle.|


_Machine Type_

|||
|--- |--- |
|standard|0,2774 $ (0,5661)|
|large_model|0,6915 $ (1,4111)|
|complex_model_s|0,4141 $ (0,8450)|
|complex_model_m|0,8281 $ (1,6900)|
|complex_model_l|1,6562 $ (3,3800)|
|standard_gpu|1,2118 $ (2,4731)|
|complex_model_m_gpu|3,7376 $ (7,6278)|
|complex_model_l_gpu|7,4752 $ (15,2555)|
|standard_p100 (Beta)|2,6864 $ (5,4824)|
|complex_model_m_p100 (Beta)|9,636 $ (19,6653)|
|cloud_tpu (Beta)|9,4900 $ (19,3673)|

<!--Awu30tz27k.b(h3o9n-->

Low level Cloud computer instances with deep learning pre configurations.


## Conclusion


   - Pricing is very intransparent on all platforms
   - GCloud has custom image for Deep Learning/DataScience
   - AWS seems promising, but I could not get a Data Science AMI up due quota restictions
   - Could get an Azure Deep Learning VM up and running. Everything nicely preconfigured
     => Using a NC6 instance with one K80 instance