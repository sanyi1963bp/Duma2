# VoiceTex 3 – Részletes Használati Útmutató / Complete User Guide

---

# 🇭🇺 MAGYAR VERZIÓ

## 1. RENDSZERKÖVETELMÉNYEK

### Minimális követelmények
- **Operációs rendszer:** Windows 10 vagy újabb (64-bit)
- **Processzor:** Intel Core i5 / AMD Ryzen 5 vagy jobb
- **RAM:** 8 GB (ajánlott: 16 GB)
- **SSD:** 20 GB szabad hely a modellek és tanító adatokhoz
- **Hang eszköz:** Működő mikrofon és hangszóró

### Ajánlott konfiguráció
- **GPU:** NVIDIA GPU (CUDA támogatással) – jelentős gyorsulás
  - GeForce GTX 1050 Ti vagy jobb ajánlott
  - CUDA 11.8+ szükséges
- **RAM:** 16+ GB (a nagyobb modellek és párhuzamos feldolgozáshoz)
- **SSD:** 30+ GB (több modell és hosszabb tanító adatok tárolásához)

### Mit kell még?
- **Python 3.10 vagy 3.11** (3.12+ még nem ajánlott, kompatibilitási problémák)
- **FFmpeg** (csak a felirat-modul használatához szükséges)
- **Internet kapcsolat** az első futásnál (modellek letöltéséhez, ~2-4 GB)

---

## 2. TELEPÍTÉS LÉPÉSRŐL LÉPÉSRE

### 2.1 Python telepítése

1. Menj a [python.org](https://www.python.org/downloads/) oldalra
2. Töltsd le a **Python 3.10** vagy **3.11** verzióját (Windows 64-bit)
3. Futtasd a telepítőt
4. **FONTOS:** Pipáld be az „Add Python to PATH" opciót! 
5. Válaszd: `Install Now`
6. Várj, amíg végez

**Ellenőrzés:** Nyiss egy parancssor-ablakot (`Win+R` → `cmd`) és írj:
```bash
python --version
```
Ha megjelenik `Python 3.10.x` vagy `3.11.x`, jó!

---

### 2.2 VoiceTex 3 letöltése és telepítése

#### Opció A: GitHub-ről (ajánlott)
1. Menj a GitHub repo-ra: `https://github.com/sanyi1963bp/VoiceTex-3`
2. Kattints a zöld **Code** gombra
3. Válaszd: **Download ZIP**
4. Csomagold ki egy mappába, pl.: `C:\Alkalmazások\VoiceTex-3`

#### Opció B: Git klónozása (ha van Git telepítve)
```bash
git clone https://github.com/sanyi1963bp/VoiceTex-3.git
cd VoiceTex-3
```

---

### 2.3 Függőségek telepítése

1. Nyiss egy **parancssor**-ablakot (`cmd`)
2. Navigálj a VoiceTex mappájába:
```bash
cd C:\Alkalmazások\VoiceTex-3
```

3. Telepítsd az alapvető csomagokat:
```bash
pip install faster-whisper ctranslate2 transformers peft datasets torch
pip install sounddevice scipy numpy keyboard silero-vad
pip install librosa soundfile pyttsx3 num2words
```

**Megjegyzés:** A telepítés több percet vesz igénybe (PyTorch mérete: ~500 MB).

4. Ellenőrzés: ha nem jött hiba, jó!

---

### 2.4 FFmpeg telepítése (opcionális, felirat-modulhoz szükséges)

1. Menj a [ffmpeg.org](https://ffmpeg.org/download.html) oldalra
2. Töltsd le a Windows build-et
3. Csomagold ki egy mappába, pl.: `C:\ffmpeg`
4. Add hozzá a PATH-hez:
   - `Win+X` → `Rendszertulajdonságok`
   - `Speciális` → `Környezeti változók`
   - Új változó: `Path` → `C:\ffmpeg\bin`
   - OK

**Ellenőrzés:**
```bash
ffmpeg -version
```

---

## 3. AZ ALKALMAZÁS INDÍTÁSA

### Normál indítás
```bash
python voicetex_v3.py
```

### Parancssor nélkül (gyorsbillentyű)

1. Hozz létre egy `start.bat` fájlt a VoiceTex mappában:
```batch
@echo off
python voicetex_v3.py
pause
```

2. Mentsd el és dupla kattintassal indítsd el!

3. Az első futásnál az app automatikusan letölti a szükséges modelleket (~2-4 GB).

---

## 4. INTERFÉSZ BEMUTATÁSA

### Főablak elemei

#### Felső sáv
- **Mikrofon eszköz kiválasztása:** A gépedben lévő mikrofonok listája
- **Modell választó:** Whisper modellek (tiny, base, small, medium, large) + egyedi LoRA modell

#### Középső szakasz: Diktálási módok

**PTT (Push-To-Talk) mód:**
- Tartsd nyomva: `CTRL + WIN` billentyűt → hangfelismerés indul
- Engedd el → szöveg beillesztődik a kurzor pozíciójára
- Előnye: precíz, kézzel irányított
- Hátránya: kézzel kell aktiválni

**VAD (Voice Activity Detection) mód:**
- Nyomj egy gombot, majd beszélj → automatikusan érzékeli, ha beszélsz
- Elhallgatsz ~2 mp-re → beillesztódik a szöveg
- Előnye: kéz-mentes, arról beszélsz, amit akarsz
- Hátránya: lehet, hogy rossz időben megáll

#### Alsó szakasz: Beállítások

**Hallucináció-szűrő:** ON/OFF
- Ha ON: eltávolítja a Whisper jellegzetes hallucinációit
- Pl.: „amara.org", „feliratok a közösségtől"

**Vágólap-visszaállítás:** ON/OFF
- Ha ON: a diktálás után helyreállítja az eredeti vágólapot
- (Ezért lehet biztonságosan egy másik szöveg másolása közben diktálni)

**Szöveg normalizálás:** ON/OFF
- Ha ON: helyesírást javít, számokat szavakra alakít

---

## 5. FUNKCIÓK RÉSZLETESEN

### 5.1 Hangparancsok

Ezeket a szavakat kimondva speciális karaktereket vagy formázást adhatsz be:

#### Írásjelek
- „**pont**" → `.`
- „**vessző**" → `,`
- „**pontosvessző**" → `;`
- „**kettőspont**" → `:`
- „**felkiáltójel**" → `!`
- „**kérdőjel**" → `?`

#### Szerkezet
- „**új sor**" → sortörés
- „**új bekezdés**" → üres sor + nagybetű a következő mondathoz

#### Szmájlik
- „**szmájli**" → 🙂
- „**nevető szmájli**" → 😂
- „**kacsintós szmájli**" → 😉
- „**szomorú szmájli**" → 😢
- „**dühös szmájli**" → 😠
- „**puszi szmájli**" → 😘
- „**szív szmájli**" → ❤️
- „**lájk szmájli**" → 👍

**Tipp:** A hangparancsok nem kerülnek a tanító adatokba – csak az, amit ténylegesen kiejtettél!

---

### 5.2 Saját modell tanítása (LoRA finomhangolás)

A VoiceTex képes megtanulni az **egyedi hangodat** és a mondanivalódat.

#### Lépésről lépésre

**1. Diktálás + javítás**

- Diktálsz valamit PTT vagy VAD módban
- Ha hibás, a felismerés után kattints az „Javítás" gombra
- Írj le a **helyes szöveget**
- Kattints az „Engedélyez" gombra

Az app egy **tanító párban** tárlja: (eredeti hang + javított szöveg)

**2. Tanítás indítása**

- A tanító adatok gyűjtése után: kattints a **„Tanítás indítása"** gombra
- Ez egy LoRA finomhangolást végez (~5-10 perc a számítógéptől függően)
- Az app letölt egy base modellt, finomhangol, majd konvertál CT2-re

**3. Az új modell aktiválása**

- Sikeres tanítás után automatikusan aktiválódik az egyedi modell
- Ez az „🧠 Egyedi tanított modell (LoRA→CT2)" lesz kiválasztva

**4. Visszaváltás**

- Ha szeretnél, bármikor visszaválthatsz a gyári modellre a modell-választóból

#### Tanítási tippek
- Gyűjts **legalább 20-30 diktálás + javítás párt**
- Minél több és különfélébb adat, annál jobb az eredmény
- Az alapmodelltől függ a végeredmény (large jobb, mint small)
- A tanítás után az app CT2-formátumra konvertál (gyorsabb inferencia)

---

### 5.3 Felirat-modul (SRT készítés)

#### Videóból SRT felirat készítése

1. Kattints a **„Felirat készítés"** fülre
2. Válassz egy videofájlt (MP4, WebM, stb.)
3. Beállítások:
   - **Whisper modell:** válaszd azt, ami tetszik (nagyobb = jobb, lassabb)
   - **Fordítás:** Bekapcsolod-e az angol→magyar fordítást?
   - **Párbeszéd-küszöb:** Mi számít külön mondatnak? (ajánlott: 2-3 másodperc)

4. Kattints: **„Felirat kezdete"**
5. Az app kivonja a hangot, feldolgozza és SRT fájlt hoz létre
6. Az SRT megjelenik a Video Studio vagy VLC-ben

**Hosszú videó?** Óvatosan! 2+ órás videó sok időt vesz igénybe.

---

## 6. HIBAELHÁRÍTÁS

### Probléma: „Nincs hangfelismerés"

**Okok és megoldások:**

1. **A mikrofon nem működik**
   - Windows Hangbeállítások (jobb sarok hangikon)
   - Ellenőrizd: microphon be van-e kapcsolva
   - Próbálj másik mikrofont a kiválasztóból

2. **Silero VAD nincs telepítve**
   ```bash
   pip install silero-vad
   ```

3. **A Whisper modell nem töltődött le**
   - Menj a `whisper_lora_magyar` mappába
   - Ha üres: futtasd újra az alkalmazást (újra próbálja letölteni)
   - Erős internet kapcsolat szükséges

---

### Probléma: „AttributeError: fw_model"

Ezt jelezné: a modell nem töltődött be helyesen.

**Megoldás:**
1. Zárd be az alkalmazást
2. Töröld a `whisper_lora_magyar` és `whisper_ct2_magyar` mapákat
3. Indítsd újra az appot (újra letöltödik)

---

### Probléma: Rossz beszédfelismerés

**Ok:** Az alapmodell nem ideális az egyedi hangoddal.

**Megoldások:**
1. Váltsd nagyobb modellre (tiny → small → medium)
2. Tanítsd a saját modelldet (lásd 5.2 szakasz)
3. Ellenőrizd: nincs-e zaj a háttérben (ventillátor, utca)

---

### Probléma: Az alkalmazás lassú / traverzálódik

**Okok:**
- Sok a háttér-feladat (antivírus, frissítés)
- Kevés a RAM

**Megoldások:**
1. Zárd be az egyéb alkalmazásokat
2. Használj kisebb Whisper modellt (medium helyett small)
3. GPU-t engedélyezz (CUDA, ha van NVIDIA kártyád)
4. Frissítsd az `onnxruntime`-t:
   ```bash
   pip install -U onnxruntime-gpu
   ```

---

### Probléma: „ModuleNotFoundError: keyboard"

Hiányzik a `keyboard` csomag.

**Megoldás:**
```bash
pip install keyboard
```

Újraindítás szükséges!

---

### Probléma: Felirat-modul nem működik

**Ok:** FFmpeg nincs telepítve vagy nem érhető el.

**Megoldás:**
1. Telepítsd az FFmpeg-et (lásd 2.4 szakasz)
2. Ellenőrizd: `ffmpeg -version` a parancssorban
3. Újra indítsd az alkalmazást

---

### Probléma: LoRA tanítás nem működik / hiba a konverzió

**Ok:** PyTorch vagy CUDA verzió ütközés.

**Megoldás:**
```bash
pip install -U torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Majd újraindítás.

---

## 7. GYIK (GYAKRAN ISMÉTELT KÉRDÉSEK)

### K: Működik-e Mac vagy Linux-on?

**V:** Jelenleg Windows-specifikus (keyboard hook, hangeszköz-kezelés). Mac/Linux portra később lehetség van.

---

### K: Működhet-e offline?

**V:** Az első futásnál szükséges az internet (modellek letöltéséhez). Utána teljesen offline működik!

---

### K: Milyen nagy az egyedi modell fájl?

**V:** ~200 MB CT2 formátumban. Ha LoRA marad: ~50 MB.

---

### K: Lehet-e több egyedi modellt tanítani?

**V:** Jelenleg az app egy egyedi modellt támogat. A future verzióban lehet majd több variáns.

---

### K: Mit csináljak, ha elveszítem az egyedi modellt?

**V:** A tanító párok (hang + szöveg) megmaradnak a `tanito_adatok` mappában. Újra lehet tanítani.

---

### K: Lehet-e angolul vagy más nyelveken használni?

**V:** Jelenleg Magyar és angol (Whisper támogat sok nyelvet). Más nyelvhez a fordító modult ki kell cserélni.

---

### K: Mennyi ideig tart a tanítás?

**V:** 20-30 párnál: ~5-10 perc (modern PC-n). GPU-val: ~2-3 perc.

---

### K: Lehet-e a diktálást kikapcsolni, ha nem akarom használni?

**V:** Igen! Egyszerűen ne nyomd meg a CTRL+WIN-t vagy ne kapcsold be a VAD-ot.

---

### K: A szövegem biztonságban van-e?

**V:** Teljesen lokális (offline) feldolgozás. Semmit nem küldelek GitHub-nak vagy egyéb szervereknek.

---

### K: Létezik-e beépített történet vagy napló?

**V:** A tanító párok és felismert szövegek mentésre kerülnek. Később lehet naplózás bővítés.

---

### K: Tudok-e különböző hangszínekre tanítani?

**V:** Jelenleg egy egyedi modell van. A jövőben lehet profilok (még nem implement).

---

## 8. TIPPEK ÉS TRÜKKÖK

### 1. Gyorsabb diktálás
- Használj PTT módot (CTRL+WIN) → pontosabb és gyorsabb
- VAD csak akkor, ha kézmentes munka szükséges

### 2. Jobb felismerés
- Beszélj világosan, ne mondjál 15 szót egy levegővétellel
- Mondat után szüneteljen egy kicsit
- A hangparancsokat világosan mondj ki

### 3. Tanítás előtt
- Gyűjtsd össze a tanító párokat (min. 20-30)
- Minél változatosabb mondatok, annál jobb

### 4. Vágólap-biztonság
- Ha másolsz valamit diktálás közben, a régi vágólap automatikusan visszaáll
- Biztonságos párba dolgozni!

---

## 9. TÁMOGATÁS ÉS HIBAVIZSGÁLAT

### Ahol lehet segítséget kérni:

1. **GitHub Issues:** https://github.com/sanyi1963bp/VoiceTex-3/issues
2. **Ez az útmutató:** Olvasd el újra a Hibaelhárítás szekciót
3. **Python parancssor debug:**
   ```bash
   python voicetex_v3.py 2>&1 | tee debug.log
   ```
   Ez egy `debug.log` fájlt hoz létre, amit majd megoszthat!

---

---

# 🇬🇧 ENGLISH VERSION

## 1. SYSTEM REQUIREMENTS

### Minimum Requirements
- **OS:** Windows 10 or newer (64-bit)
- **CPU:** Intel Core i5 / AMD Ryzen 5 or better
- **RAM:** 8 GB (recommended: 16 GB)
- **SSD:** 20 GB free space for models and training data
- **Audio Device:** Working microphone and speakers

### Recommended Configuration
- **GPU:** NVIDIA GPU with CUDA support – significant speedup
  - GeForce GTX 1050 Ti or better recommended
  - CUDA 11.8+ required
- **RAM:** 16+ GB (for larger models and parallel processing)
- **SSD:** 30+ GB (for multiple models and extensive training data)

### Additional Requirements
- **Python 3.10 or 3.11** (3.12+ not yet recommended due to compatibility issues)
- **FFmpeg** (only needed for subtitle module)
- **Internet connection** on first run (to download models, ~2-4 GB)

---

## 2. INSTALLATION STEP BY STEP

### 2.1 Installing Python

1. Go to [python.org](https://www.python.org/downloads/)
2. Download **Python 3.10** or **3.11** (Windows 64-bit)
3. Run the installer
4. **IMPORTANT:** Check the "Add Python to PATH" option!
5. Select: `Install Now`
6. Wait for completion

**Verification:** Open Command Prompt (`Win+R` → `cmd`) and type:
```bash
python --version
```
If you see `Python 3.10.x` or `3.11.x`, you're good!

---

### 2.2 Downloading and Installing VoiceTex 3

#### Option A: From GitHub (recommended)
1. Go to the GitHub repo: `https://github.com/sanyi1963bp/VoiceTex-3`
2. Click the green **Code** button
3. Select: **Download ZIP**
4. Extract to a folder, e.g.: `C:\Applications\VoiceTex-3`

#### Option B: Clone with Git
```bash
git clone https://github.com/sanyi1963bp/VoiceTex-3.git
cd VoiceTex-3
```

---

### 2.3 Installing Dependencies

1. Open **Command Prompt** (`cmd`)
2. Navigate to the VoiceTex folder:
```bash
cd C:\Applications\VoiceTex-3
```

3. Install core packages:
```bash
pip install faster-whisper ctranslate2 transformers peft datasets torch
pip install sounddevice scipy numpy keyboard silero-vad
pip install librosa soundfile pyttsx3 num2words
```

**Note:** Installation takes several minutes (PyTorch is ~500 MB).

4. Verification: If no errors appeared, you're good!

---

### 2.4 Installing FFmpeg (Optional, required for subtitle module)

1. Go to [ffmpeg.org](https://ffmpeg.org/download.html)
2. Download the Windows build
3. Extract to a folder, e.g.: `C:\ffmpeg`
4. Add to PATH:
   - `Win+X` → `System Properties`
   - `Advanced` → `Environment Variables`
   - New variable: `Path` → `C:\ffmpeg\bin`
   - OK

**Verification:**
```bash
ffmpeg -version
```

---

## 3. STARTING THE APPLICATION

### Normal start
```bash
python voicetex_v3.py
```

### Without Command Prompt (Quick Launch)

1. Create a `start.bat` file in the VoiceTex folder:
```batch
@echo off
python voicetex_v3.py
pause
```

2. Save and double-click to launch!

3. On first run, the app automatically downloads required models (~2-4 GB).

---

## 4. INTERFACE OVERVIEW

### Main Window Elements

#### Top Bar
- **Microphone Device Selection:** List of microphones on your computer
- **Model Chooser:** Whisper models (tiny, base, small, medium, large) + custom LoRA model

#### Middle Section: Dictation Modes

**PTT (Push-To-Talk) Mode:**
- Hold down: `CTRL + WIN` → speech recognition starts
- Release → text is inserted at cursor position
- Advantage: precise, manually controlled
- Disadvantage: requires manual activation

**VAD (Voice Activity Detection) Mode:**
- Press a button, then speak → automatically detects when you're speaking
- Fall silent for ~2 seconds → text is inserted
- Advantage: hands-free, dictate what you want
- Disadvantage: may stop at wrong times

#### Bottom Section: Settings

**Hallucination Filter:** ON/OFF
- If ON: removes Whisper's characteristic hallucinations
- E.g.: "amara.org", "subtitles from community"

**Clipboard Restoration:** ON/OFF
- If ON: restores original clipboard after dictation
- (Safe to copy other text while dictating)

**Text Normalization:** ON/OFF
- If ON: fixes spelling, converts numbers to words

---

## 5. FEATURES IN DETAIL

### 5.1 Voice Commands

Saying these words inserts special characters or formatting:

#### Punctuation
- "**pont**" (period) → `.`
- "**vessző**" (comma) → `,`
- "**pontosvessző**" (semicolon) → `;`
- "**kettőspont**" (colon) → `:`
- "**felkiáltójel**" (exclamation) → `!`
- "**kérdőjel**" (question) → `?`

#### Structure
- "**új sor**" (new line) → line break
- "**új bekezdés**" (new paragraph) → blank line + capitalized next sentence

#### Emojis
- "**szmájli**" → 🙂
- "**nevető szmájli**" → 😂
- "**kacsintós szmájli**" → 😉
- "**szomorú szmájli**" → 😢
- "**dühös szmájli**" → 😠
- "**puszi szmájli**" → 😘
- "**szív szmájli**" → ❤️
- "**lájk szmájli**" → 👍

**Tip:** Voice commands don't get added to training data – only what you actually said!

---

### 5.2 Training Custom Model (LoRA Fine-tuning)

VoiceTex can learn your **unique voice** and vocabulary.

#### Step by Step

**1. Dictate + Correct**

- Dictate something in PTT or VAD mode
- If wrong, click the "Correct" button after recognition
- Type the **correct text**
- Click "Confirm"

The app stores a **training pair**: (original audio + corrected text)

**2. Start Training**

- After collecting training data: click **"Start Training"** button
- This performs LoRA fine-tuning (~5-10 minutes depending on your computer)
- The app downloads a base model, fine-tunes it, then converts to CT2

**3. Activate the New Model**

- After successful training, the custom model activates automatically
- It becomes the "🧠 Custom Trained Model (LoRA→CT2)" option

**4. Switch Back**

- Anytime, switch back to factory model from the model chooser

#### Training Tips
- Collect **at least 20-30 dictation + correction pairs**
- More diverse data = better results
- Base model matters (large better than small)
- After training, app converts to CT2 format (faster inference)

---

### 5.3 Subtitle Module (SRT Generation)

#### Creating Subtitles from Video

1. Click the **"Subtitles"** tab
2. Select a video file (MP4, WebM, etc.)
3. Settings:
   - **Whisper Model:** choose your preference (larger = better, slower)
   - **Translation:** Enable English→Hungarian translation?
   - **Dialogue Threshold:** What counts as separate sentence? (recommended: 2-3 seconds)

4. Click: **"Start Subtitles"**
5. App extracts audio, processes it, creates SRT file
6. Open SRT in Video Studio or VLC

**Long video?** Be careful! 2+ hour video takes considerable time.

---

## 6. TROUBLESHOOTING

### Problem: "No Speech Recognition"

**Causes and Solutions:**

1. **Microphone not working**
   - Windows Sound Settings (right-click speaker icon)
   - Check: microphone is enabled
   - Try different microphone from selector

2. **Silero VAD not installed**
   ```bash
   pip install silero-vad
   ```

3. **Whisper model didn't download**
   - Check `whisper_lora_magyar` folder
   - If empty: restart app (will retry download)
   - Strong internet connection required

---

### Problem: "AttributeError: fw_model"

Indicates: model failed to load properly.

**Solution:**
1. Close the application
2. Delete `whisper_lora_magyar` and `whisper_ct2_magyar` folders
3. Restart app (will re-download)

---

### Problem: Poor Speech Recognition

**Cause:** Base model not ideal for your voice.

**Solutions:**
1. Switch to larger model (tiny → small → medium)
2. Train your own model (see section 5.2)
3. Check: no background noise (fan, street)

---

### Problem: Application is Slow / Stutters

**Causes:**
- Many background tasks (antivirus, updates)
- Insufficient RAM

**Solutions:**
1. Close other applications
2. Use smaller Whisper model (small instead of medium)
3. Enable GPU (CUDA, if you have NVIDIA card)
4. Update `onnxruntime`:
   ```bash
   pip install -U onnxruntime-gpu
   ```

---

### Problem: "ModuleNotFoundError: keyboard"

Missing `keyboard` package.

**Solution:**
```bash
pip install keyboard
```

Restart required!

---

### Problem: Subtitle Module Not Working

**Cause:** FFmpeg not installed or not accessible.

**Solution:**
1. Install FFmpeg (see section 2.4)
2. Check: `ffmpeg -version` in command prompt
3. Restart application

---

### Problem: LoRA Training Not Working / Conversion Error

**Cause:** PyTorch or CUDA version conflict.

**Solution:**
```bash
pip install -U torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Then restart.

---

## 7. FAQ (FREQUENTLY ASKED QUESTIONS)

### Q: Does it work on Mac or Linux?

**A:** Currently Windows-specific (keyboard hook, audio device handling). Mac/Linux port possible in future.

---

### Q: Can it work offline?

**A:** Internet required on first run (to download models). After that, fully offline!

---

### Q: How large is a custom model file?

**A:** ~200 MB in CT2 format. If kept as LoRA: ~50 MB.

---

### Q: Can I train multiple custom models?

**A:** Currently supports one custom model. Future versions may allow multiple variants.

---

### Q: What if I lose my custom model?

**A:** Training pairs (audio + text) remain in `tanito_adatok` folder. Can retrain.

---

### Q: Can I use it in English or other languages?

**A:** Currently Hungarian and English (Whisper supports many languages). Other languages require switching translator model.

---

### Q: How long does training take?

**A:** 20-30 pairs: ~5-10 minutes (modern PC). With GPU: ~2-3 minutes.

---

### Q: Can I turn off dictation if I don't want to use it?

**A:** Yes! Simply don't press CTRL+WIN or enable VAD.

---

### Q: Is my text secure?

**A:** Fully local (offline) processing. Nothing sent to GitHub or other servers.

---

### Q: Is there built-in history or logging?

**A:** Training pairs and recognized text are saved. Logging expansion possible in future.

---

### Q: Can I train for different accents?

**A:** Currently one custom model. Future versions may support profiles (not implemented yet).

---

## 8. TIPS AND TRICKS

### 1. Faster Dictation
- Use PTT mode (CTRL+WIN) → more accurate and faster
- VAD only when hands-free needed

### 2. Better Recognition
- Speak clearly, don't say 15 words in one breath
- Pause after sentences
- Say voice commands clearly

### 3. Before Training
- Collect training pairs (min. 20-30)
- More varied sentences = better results

### 4. Clipboard Safety
- If you copy something while dictating, old clipboard auto-restores
- Safe to work in pairs!

---

## 9. SUPPORT AND DEBUG

### Where to Get Help:

1. **GitHub Issues:** https://github.com/sanyi1963bp/VoiceTex-3/issues
2. **This Guide:** Re-read the Troubleshooting section
3. **Python Command Debug:**
   ```bash
   python voicetex_v3.py 2>&1 | tee debug.log
   ```
   Creates a `debug.log` file you can share!

---

## 10. LICENSE

This application is released under the **MIT License**. See LICENSE file for details.

---

**Thank you for using VoiceTex 3! Happy dictating! 🎤**

---

*Last updated: 2026-07-16*
