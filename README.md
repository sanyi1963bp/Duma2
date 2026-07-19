# Voicetex v3 – magyar diktáló és feliratkészítő

Asztali alkalmazás Windowsra, amely magyar nyelvű diktálást tesz lehetővé
bármelyik programba (Word, böngésző, e-mail stb.), és a hangfelismerő modellt
a saját hangodhoz tudja igazítani LoRA finomhangolással.

## Fő funkciók

- **Diktálás bárhová** – a felismert szöveg automatikusan beillesztődik oda,
  ahol a kurzor áll (a vágólap eredeti tartalma megőrződik).
- **Két felvételi mód** – PTT (CTRL+WIN nyomva tartás) vagy VAD mód, amely a
  Silero neurális hálóval automatikusan érzékeli, mikor beszélsz.
- **Hangparancsok** – „pont", „vessző", „pontosvessző", „kettőspont",
  „felkiáltójel", „kérdőjel", „új sor", „új bekezdés", valamint szmájlik
  („szmájli" → 🙂, „nevető szmájli" → 😂, „szív szmájli" → ❤️ stb.).
- **Öntanulás** – a felismert hang + javított szöveg párokból LoRA
  finomhangolás, CTranslate2 konverzióval; az egyedi modell a
  modellválasztóból aktiválható. Kötegelt tanítás hosszú hangfelvétel +
  kézirat alapján is lehetséges.
- **Filmfelirat modul** – videóból SRT felirat készítése faster-whisperrel,
  opcionális magyar fordítással (Helsinki-NLP opus-mt-en-hu).

## Telepítés

### Egyszerű telepítés (ajánlott)

Nem kell semmit előre telepíteni. Töltsd le a repót, majd **dupla katt a
`Telepito.bat` fájlra**. A telepítő mindent elintéz:

- ha nincs Python a gépen, automatikusan letölti és felteszi (3.10–3.12),
- saját, elkülönített környezetet (venv) hoz létre a mappán belül,
- felismeri az NVIDIA GPU-t, és a hozzá illő PyTorch-ot telepíti
  (RTX 50xx / Blackwell kártyához a CUDA 12.8-as változatot; GPU nélkül
  CPU-s változatot),
- telepíti az összes többi csomagot, és parancsikont tesz az asztalra.

Ezután indítás az asztali **Voicetex v3** ikonnal vagy a
`Voicetex_Inditas.bat` fájllal. Az első indításkor a Whisper modell (~3 GB)
egyszer letöltődik, utána helyi gyorsítótárból tölt, internet nélkül is.

Részletes útmutató és hibaelhárítás: [OLVASSEL_TELEPITES.txt](OLVASSEL_TELEPITES.txt).

A feliratmodul videó-hangkinyeréséhez [ffmpeg](https://ffmpeg.org/download.html)
is szükséges (ezt külön kell telepíteni).

### Kézi telepítés (haladóknak)

Ha inkább magad állítod be a környezetet, Python 3.10+ szükséges, valamint:

```
pip install faster-whisper ctranslate2 transformers peft datasets torch
pip install sounddevice scipy numpy keyboard silero-vad
pip install librosa soundfile pyttsx3 num2words
```

A `large-v3` és `large-v3-turbo` modellekhez friss faster-whisper
szükséges (`pip install -U faster-whisper ctranslate2`).

## Használat

```
python voicetex_v3.py
```

Részletek a felület súgóbuborékaiban (vidd az egeret bármelyik vezérlőre),
a változások listája a [JAVITASOK.md](JAVITASOK.md) fájlban.

## Köszönet és tisztelgés / Acknowledgements and Tribute

**Magyarul:** Ez az alkalmazás kizárólag helyben futó, nyílt modellekre és
eszközökre épül. Tiszteljük és becsüljük az alkotókat, akik ezeket a nagyszerű
szoftvereszközöket ingyen, szabadon a kezünkbe adták – az ő munkájuk nélkül ez
a program nem létezhetne. Köszönjük!

**In English:** This application is built entirely on locally-running, open
models and tools. We honour and respect the creators who put these wonderful
software tools into our hands, freely and openly – without their work this
program could not exist. Thank you!

### Az alapok / Foundations

- **Whisper Flow** – az eredeti projekt, amelyre ennek az alkalmazásnak a
  hangfelismerő alapjai épülnek / the original project this application's
  speech recognition foundations are built upon.

### Helyben futó modellek / Locally-running models

- [OpenAI Whisper](https://github.com/openai/whisper) – beszédfelismerő modell
  (large-v1/v2/v3, turbo) / speech recognition models, by **OpenAI**
- [Silero VAD](https://github.com/snakers4/silero-vad) – hangaktivitás-felismerő
  neurális háló / voice activity detection network, by the **Silero Team**
- [Helsinki-NLP OPUS-MT en-hu](https://huggingface.co/Helsinki-NLP/opus-mt-en-hu)
  – angol→magyar fordítómodell / English→Hungarian translation model, by
  **Jörg Tiedemann and the Language Technology Research Group at the
  University of Helsinki**

### Eszközök és könyvtárak / Tools and libraries

- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) – gyors Whisper
  inferencia / fast Whisper inference, by **SYSTRAN**
- [CTranslate2](https://github.com/OpenNMT/CTranslate2) – hatékony
  modellfuttató motor / efficient inference engine, by the **OpenNMT team**
- [Hugging Face Transformers, PEFT, Datasets](https://github.com/huggingface)
  – modellkezelés és LoRA finomhangolás / model handling and LoRA fine-tuning,
  by **Hugging Face**
- [PyTorch](https://pytorch.org/) – gépi tanulási keretrendszer / machine
  learning framework, by the **PyTorch team**
- [Marian NMT](https://marian-nmt.github.io/) – a fordítómodell architektúrája
  / the translation model architecture, by the **Marian NMT team**
- [FFmpeg](https://ffmpeg.org/) – hangkinyerés videóból / audio extraction,
  by the **FFmpeg team**
- valamint / as well as: [librosa](https://librosa.org/),
  [soundfile](https://github.com/bastibe/python-soundfile),
  [sounddevice](https://github.com/spatialaudio/python-sounddevice),
  [keyboard](https://github.com/boppreh/keyboard),
  [pyttsx3](https://github.com/nateshmbhat/pyttsx3),
  [num2words](https://github.com/savoirfairelinux/num2words),
  [NumPy](https://numpy.org/), [SciPy](https://scipy.org/)

## Licenc

MIT – lásd a [LICENSE](LICENSE) fájlt.
