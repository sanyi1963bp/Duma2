import os
import sys
import time
import queue
import json
import threading
import re
import zipfile
import subprocess
import shutil
import numpy as np
import scipy.io.wavfile as wav
import torch
import tkinter as tk
from tkinter import messagebox, ttk, filedialog

# ── Opcionális függőségek ─────────────────────────────────────────────────────

try:
    import sounddevice as sd
    MIKROFON_ELERHETO = True
except Exception:
    MIKROFON_ELERHETO = False

try:
    import pyttsx3
    TTS_ELERHETO = True
except Exception:
    TTS_ELERHETO = False

try:
    import librosa
    import soundfile as sf
    DARABOLO_ELERHETO = True
except Exception:
    DARABOLO_ELERHETO = False

try:
    from num2words import num2words
    NUM2WORDS_ELERHETO = True
except Exception:
    NUM2WORDS_ELERHETO = False

try:
    import keyboard
    KEYBOARD_ELERHETO = True
except Exception:
    KEYBOARD_ELERHETO = False

# ── Gyors inferencia: faster-whisper ─────────────────────────────────────────
try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER_ELERHETO = True
except Exception:
    FASTER_WHISPER_ELERHETO = False

# ── VAD: Silero hangaktivitás-felismerés ──────────────────────────────────────
try:
    from silero_vad import load_silero_vad, VADIterator
    SILERO_VAD_ELERHETO = True
except Exception:
    SILERO_VAD_ELERHETO = False

# ── LoRA tanítás (HuggingFace / PEFT) ────────────────────────────────────────
# OPTIMALIZÁLÁS: a transformers / datasets / peft import ~10-20 mp-cel lassította
# az app INDULÁSÁT, pedig csak tanításkor / fordításkor kellenek. Mostantól
# lazy importtal, csak az első használatkor töltődnek be.
import importlib.util

TANITAS_ELERHETO = all(
    importlib.util.find_spec(m) is not None
    for m in ("transformers", "datasets", "peft")
)
FORDITO_ELERHETO = importlib.util.find_spec("transformers") is not None

# ── BEÁLLÍTÁSOK ───────────────────────────────────────────────────────────────
MINTAVETELI_FREKVENCIA = 16000
MODEL_NAME             = "openai/whisper-large-v3"
DEFAULT_FW_MODEL_NAME  = "large-v2"          # stabil alapértelmezett: large-v2=80 mel
MODEL_SETTINGS_FILE    = "./voicetex_model_settings.json"
FW_MODEL_CHOICES       = [
    ("tiny – nagyon gyors, gyengébb", "tiny"),
    ("base – gyors, alap pontosság", "base"),
    ("small – gyorsabb, közepes pontosság", "small"),
    ("medium – jó kompromisszum", "medium"),
    ("large-v1 – régebbi nagy modell", "large-v1"),
    ("large-v2 – stabil magyarhoz / ajánlott", "large-v2"),
    ("large-v3 – pontosabb, de lassabb", "large-v3"),
    ("large-v3-turbo – gyors modern modell", "large-v3-turbo"),
    ("🧠 Egyedi tanított modell (LoRA→CT2)", "custom-ct2"),
]
LORA_OUTPUT_DIR        = "./whisper_lora_magyar"
CT2_MODEL_DIR          = "./whisper_ct2_magyar"   # konvertált, gyors modell
CUSTOM_CT2_ID          = "custom-ct2"             # az egyedi tanított modell azonosítója a választóban
DATASET_DIR            = "./tanito_adatok"
BACKUP_DIR             = "./voicetex_backups"

# VAD hangolás
VAD_KUSZOB             = 0.45   # érzékenység (0–1, kisebb = érzékenyebb)
VAD_CSEND_MS           = 900    # ennyi ms csend után áll meg a felvétel
                                # (⚡ 500-600-ra csökkentve érezhetően fürgébb,
                                #  de hosszabb gondolkodási szünetnél elvághat)
VAD_PADDING_MS         = 120    # rövid puffer a szavak elejére/végére

# Diktálási sebesség/pontosság kompromisszum:
# beam_size=1 (greedy) a leggyorsabb, large-v2-nél magyarra általában elég jó;
# ha romlana a pontosság, állítsd vissza 3-ra.
DIKTALAS_BEAM_SIZE     = 1

# Passzív önjavító tanulás
AUTO_TANULAS_KUSZOB    = -0.55  # avg_logprob küszöb (0=tökéletes, -1=bizonytalan)
AUTO_TANULAS_KOTEG     = 15     # ennyi jó pár összegyűlése után tanít automatikusan
AUTO_TANULAS_AUTOSTART = False  # Biztonsági mód: queue gyűlik, de LoRA tanítás csak kézzel indul
AUTO_QUEUE_FILE        = "./tanito_adatok/auto_queue.json"

# Film felirat modul
FORDITO_MODELL         = "Helsinki-NLP/opus-mt-en-hu"
FORDITO_CACHE_DIR      = "./fordito_cache"
FELIRAT_KOTEG_MERET    = 8      # (⚡ GPU-n 16-32-re emelve tovább gyorsul a fordítás)


# ─────────────────────────────────────────────────────────────────────────────
#  SEGÉDFÜGGVÉNYEK
# ─────────────────────────────────────────────────────────────────────────────

# OPTIMALIZÁLÁS: előre lefordított regexek – a re.sub minden hívásnál újra
# fordította a mintákat (a re-cache segít, de a compile + modulszintű minta
# így is gyorsabb és tisztább).
_RE_IRASJEL   = re.compile(r'[.,;:!?\-"()""„"«»—_]')
_RE_SZAM      = re.compile(r'\d+')
_RE_TOBB_SZOKOZ = re.compile(r'\s+')

def szoveg_normalizalas(szoveg):
    szoveg = szoveg.lower().strip()
    szoveg = _RE_IRASJEL.sub(' ', szoveg)
    if NUM2WORDS_ELERHETO:
        szamok = _RE_SZAM.findall(szoveg)
        for szam in sorted(szamok, key=len, reverse=True):
            try:
                betus_szam = num2words(int(szam), lang='hu')
                szoveg = szoveg.replace(szam, " " + betus_szam + " ")
            except Exception:
                pass
    szoveg = _RE_TOBB_SZOKOZ.sub(' ', szoveg).strip()
    return szoveg


import difflib

def hasonlosag_arány(s1, s2):
    s1, s2 = szoveg_normalizalas(s1), szoveg_normalizalas(s2)
    if not s1 or not s2:
        return 0.0
    if s1 == s2:
        return 1.0
    return difflib.SequenceMatcher(None, s1, s2).ratio()


# JAVÍTÁS: a korábbi hallucináció-szűrő egyetlen szó ('felirat', 'közösség')
# előfordulásakor is kidobta a TELJES felismerést, pedig ezek normál magyar
# szavak. Most csak jellegzetes Whisper-hallucinációs KIFEJEZÉSEKRE szűrünk.
HALLUCINACIO_MINTAK = [
    "amara.org", "subtitles by", "subtitle by", "translated by",
    "feliratok a közösségtől", "feliratokat a közösség",
    "felirat a közösségtől", "a feliratot készítette",
]

def hallucinacio_gyanus(szoveg):
    t = szoveg.lower()
    return any(m in t for m in HALLUCINACIO_MINTAK)


# ÚJ FUNKCIÓ: diktálási hangparancsok – a kimondott parancsszó írásjellé /
# sortöréssé alakul. A sorrend fontos: az összetett kifejezések előrébb állnak,
# hogy a "pont" ne találjon bele a "pontosvessző"-be vagy a "kettőspont"-ba.
HANGPARANCSOK = [
    (r'új\s+bekezdés\b',        '\n\n'),
    (r'új\s+sor\b',             '\n'),
    (r'pontos\s*vessz[őo]t?\b', ';'),
    (r'kettős\s*pont\b',        ':'),
    (r'felkiált[óo]\s*jel\b',   '!'),
    (r'kérd[őo]\s*jel\b',       '?'),
    (r'vessz[őo]t?\b',           ','),
    (r'pont\b',                  '.'),
]
# OPTIMALIZÁLÁS: előre fordított minták – korábban minden diktálásnál 16 regex
# fordult újra a hangparancs-átalakításban.
_HANGPARANCS_COMPILED = [
    (re.compile(r'\s*\b' + minta + r'[.,!?]*', re.IGNORECASE), jel)
    for minta, jel in HANGPARANCSOK
]

# ÚJ: szmájli-hangparancsok. A "szmájli" kulcsszó kötelező, így normál
# beszédben nem okoznak téves cserét. A Whisper többféle írásmódját
# elfogadjuk (szmájli / smájli / smiley).
_SM = r'(?:sz?m[áa]jli|smiley)'
SMAJLI_PARANCSOK = [
    (r'nevető\s+' + _SM,             '😂'),
    (r'kacsint[óo]s?\s+' + _SM,      '😉'),
    (r'szomorú\s+' + _SM,            '😢'),
    (r'dühös\s+' + _SM,              '😠'),
    (r'puszi\s+' + _SM,              '😘'),
    (r'szíve?s?\s+' + _SM,           '❤️'),
    (r'(?:lájk|like)\s+' + _SM,      '👍'),
    (r'(?:mosolyg[óo]s?\s+)?' + _SM, '🙂'),
]
_SMAJLI_COMPILED = [
    (re.compile(r'\b' + minta + r'\b[.,!?]*', re.IGNORECASE), jel)
    for minta, jel in SMAJLI_PARANCSOK
]
_RE_IRASJEL_SZOKOZ = re.compile(r'([.!?;,:])(?=[^\s\d])')
_RE_SORTORES_SZOKOZ = re.compile(r'\n[ \t]+')
_RE_SORTORES_NAGYBETU = re.compile(r'(\n+)([a-záéíóöőúüű])')

def hangparancs_atalakitas(szoveg):
    """Kimondott parancsszavak írásjellé/sortöréssé/szmájlivá alakítása.
    A parancs előtti szóközt és a Whisper által mögé tett írásjelet is elnyeli."""
    for minta, jel in _HANGPARANCS_COMPILED:
        szoveg = minta.sub(jel, szoveg)
    # Szmájliknál a szóköz megmarad a szó előtt ("köszi 🙂"), csak a parancs
    # szövege és az utána tett írásjel tűnik el.
    for minta, jel in _SMAJLI_COMPILED:
        szoveg = minta.sub(jel, szoveg)
    # Írásjel után hiányzó szóköz pótlása (sortörés elé nem kell)
    szoveg = _RE_IRASJEL_SZOKOZ.sub(r'\1 ', szoveg)
    # Sortörés utáni felesleges szóköz törlése
    szoveg = _RE_SORTORES_SZOKOZ.sub('\n', szoveg)
    # Sortörés után nagybetűvel folytatódjon a szöveg
    szoveg = _RE_SORTORES_NAGYBETU.sub(
        lambda m: m.group(1) + m.group(2).upper(), szoveg)
    return szoveg


def gep_beszel_magyarul(szoveg):
    """Magyar TTS – háttérszálon."""
    if not TTS_ELERHETO:
        return
    def _run():
        try:
            engine = pyttsx3.init()
            for voice in engine.getProperty('voices'):
                v_name, v_id = voice.name.lower(), voice.id.lower()
                if any(k in v_name or k in v_id for k in ("hungary", "szabolcs", "aniko")):
                    engine.setProperty('voice', voice.id)
                    break
            engine.setProperty('rate', 155)
            engine.setProperty('volume', 0.9)
            engine.say(szoveg)
            engine.runAndWait()
        except Exception as e:
            print(f"TTS hiba: {e}")
    threading.Thread(target=_run, daemon=True).start()


def lora_automatikus_mentes():
    """LoRA adapter mappa → zip biztonsági mentés."""
    if not os.path.exists(LORA_OUTPUT_DIR):
        return
    try:
        os.makedirs(BACKUP_DIR, exist_ok=True)
        zip_nev = os.path.join(BACKUP_DIR, f"lora_backup_{time.strftime('%Y%m%d_%H%M%S')}.zip")
        with zipfile.ZipFile(zip_nev, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(LORA_OUTPUT_DIR):
                for f in files:
                    p = os.path.join(root, f)
                    zipf.write(p, os.path.relpath(p, os.path.dirname(LORA_OUTPUT_DIR)))
        print(f"✅ Biztonsági mentés: {zip_nev}")
    except Exception as e:
        print(f"⚠️ Mentési hiba: {e}")


def lora_merge_es_ct2_konvertalas(log_fn=print):
    """
    LoRA adapter beolvasztása az alap modellbe, majd konverzió
    CTranslate2 formátumba a faster-whisper számára.
    Visszatér True-val, ha sikerült.
    """
    if not os.path.exists(os.path.join(LORA_OUTPUT_DIR, "adapter_config.json")):
        log_fn("ℹ️ Nincs LoRA adapter, alap modell marad aktív.")
        return False

    merged_dir = "./whisper_merged_temp"
    try:
        # Lazy import: csak konverziókor kell a transformers/peft.
        from transformers import WhisperProcessor, WhisperForConditionalGeneration
        from peft import PeftModel

        log_fn("🔀 LoRA súlyok beolvasztása az alap modellbe...")
        # OPTIMALIZÁLÁS: low_cpu_mem_usage → a 3GB-os modell gyorsabban és
        # fele annyi RAM-mal töltődik be (nem inicializál felesleges súlyokat).
        alap = WhisperForConditionalGeneration.from_pretrained(
            MODEL_NAME, low_cpu_mem_usage=True)
        peft_m = PeftModel.from_pretrained(alap, LORA_OUTPUT_DIR)
        merged = peft_m.merge_and_unload()
        merged.save_pretrained(merged_dir)
        proc = WhisperProcessor.from_pretrained(MODEL_NAME)
        proc.save_pretrained(merged_dir)
        del alap, peft_m, merged
        import gc; gc.collect()

        log_fn("⚙️ CTranslate2 konverzió (int8 kvantálás)...")
        quant = "float16" if torch.cuda.is_available() else "int8"
        if os.path.exists(CT2_MODEL_DIR):
            shutil.rmtree(CT2_MODEL_DIR)

        result = subprocess.run(
            [
                "ct2-transformers-converter",
                "--model", merged_dir,
                "--output_dir", CT2_MODEL_DIR,
                "--quantization", quant,
                # JAVÍTÁS: a preprocessor konfig is kerüljön a CT2 mappába,
                # különben a faster-whisper 80 mel-t feltételez a large-v3
                # alapú (128 mel) modellnél → "Invalid input features shape".
                "--copy_files", "preprocessor_config.json", "tokenizer_config.json",
                "--force"
            ],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr[:300])

        log_fn("✅ CT2 konverzió sikeres – faster-whisper betölti az egyedi modellt.")
        return True
    except FileNotFoundError:
        log_fn("⚠️ ct2-transformers-converter nem található. Telepítsd: pip install ctranslate2")
        return False
    except Exception as e:
        log_fn(f"⚠️ Konverzió hiba: {e}")
        return False
    finally:
        if os.path.exists(merged_dir):
            shutil.rmtree(merged_dir)


def _audio_beolvasas_norm(hang_utvonal):
    """WAV beolvasása float32 mono 16kHz-re normalizálva."""
    sr, audio_np = wav.read(hang_utvonal)
    # JAVÍTÁS: korábban a fájl tényleges mintavételi frekvenciáját (sr) eldobtuk
    # és mindig int16-ot feltételeztünk. Idegen wav-nál ez csendben hibás
    # featuret adott a tanításnak.
    if audio_np.ndim > 1:                                  # sztereó → mono
        audio_np = audio_np.mean(axis=1)
    if np.issubdtype(audio_np.dtype, np.integer):
        audio_np = audio_np.astype(np.float32) / float(np.iinfo(audio_np.dtype).max)
    else:
        audio_np = audio_np.astype(np.float32)
    if sr != MINTAVETELI_FREKVENCIA:
        from scipy.signal import resample_poly
        audio_np = resample_poly(audio_np, MINTAVETELI_FREKVENCIA, sr).astype(np.float32)
    return audio_np


def hatter_tanitas_tobb_minta(parok, device, log_fn=None):
    """
    OPTIMALIZÁLÁS (KRITIKUS): több (hangfájl, szöveg) pár LoRA finomhangolása
    EGYETLEN futásban.

    Korábban a kötegelt tanítás és az auto-queue mondatonként hívta az
    egymintás tanítást, ami MINDEN mondatnál újratöltötte a ~3GB-os
    whisper-large-v3 HF modellt (+ mondatonként zip backup és CT2 konverzió
    is futott). 100 mondatnál ez órákat jelentett. Most:
      - a processor és az alapmodell EGYSZER töltődik be,
      - az összes minta egy Dataset-be kerül,
      - a tanítási lépésszám a minták számával skálázódik (5 lépés/minta),
    így a végeredmény tanulás szempontjából egyenértékű, de nagyságrendekkel
    gyorsabb.

    `parok`: [(wav_utvonal, javitott_szoveg), ...]
    """
    if not parok:
        return
    if not TANITAS_ELERHETO:
        raise RuntimeError(
            "A tanításhoz szükséges csomagok hiányoznak.\n"
            "Telepítsd: pip install transformers datasets peft"
        )
    # Lazy import: a nehéz csomagok csak tanításkor töltődnek be.
    from transformers import (WhisperProcessor, WhisperForConditionalGeneration,
                              Seq2SeqTrainer, Seq2SeqTrainingArguments)
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, PeftModel

    processor = WhisperProcessor.from_pretrained(
        MODEL_NAME, language="hungarian", task="transcribe")

    sorok = []
    for i, (hang_utvonal, javitott_szoveg) in enumerate(parok, 1):
        if log_fn and len(parok) > 1:
            log_fn(f"   🎼 Feature előkészítés {i}/{len(parok)}...")
        audio_np = _audio_beolvasas_norm(hang_utvonal)
        features = processor(audio_np, sampling_rate=MINTAVETELI_FREKVENCIA).input_features[0]
        labels   = processor.tokenizer(javitott_szoveg).input_ids
        # Megjegyzés: np.asarray(features) marad numpy – a .tolist() konverzió
        # feleslegesen lassú és memóriaigényes volt.
        sorok.append({"input_features": np.asarray(features, dtype=np.float32),
                      "labels": labels})

    dataset = Dataset.from_dict({
        "input_features": [s["input_features"] for s in sorok],
        "labels":         [s["labels"] for s in sorok],
    })

    alap = WhisperForConditionalGeneration.from_pretrained(
        MODEL_NAME, low_cpu_mem_usage=True).to(device)
    if os.path.exists(os.path.join(LORA_OUTPUT_DIR, "adapter_config.json")):
        model = PeftModel.from_pretrained(alap, LORA_OUTPUT_DIR, is_trainable=True).to(device)
    else:
        cfg   = LoraConfig(r=8, lora_alpha=32, target_modules=["q_proj","v_proj"],
                           lora_dropout=0.05, bias="none")
        model = get_peft_model(alap, cfg).to(device)

    def collator(features):
        bf = [{"input_features": torch.tensor(f["input_features"])} for f in features]
        bl = [{"input_ids":      torch.tensor(f["labels"])}          for f in features]
        pf = processor.feature_extractor.pad(bf, return_tensors="pt")
        pl = processor.tokenizer.pad(bl, return_tensors="pt")
        lbl = pl["input_ids"].masked_fill(pl.attention_mask.ne(1), -100)
        return {"input_features": pf["input_features"], "labels": lbl}

    args = Seq2SeqTrainingArguments(
        output_dir=LORA_OUTPUT_DIR, per_device_train_batch_size=1,
        learning_rate=2e-4,
        max_steps=5 * len(parok),                 # 5 lépés/minta, mint eddig
        fp16=torch.cuda.is_available(),
        remove_unused_columns=False, label_names=["labels"], report_to="none"
    )
    # Transformers verziófüggő kompatibilitás:
    # régebben `tokenizer=...` kellett, újabb verziókban ez megszűnt,
    # és `processing_class=...` lett belőle. Ezért futáskor ellenőrizzük
    # a Seq2SeqTrainer aktuális szignatúráját.
    import inspect
    trainer_kwargs = dict(
        args=args,
        model=model,
        train_dataset=dataset,
        data_collator=collator,
    )
    sig = inspect.signature(Seq2SeqTrainer.__init__).parameters
    if "processing_class" in sig:
        trainer_kwargs["processing_class"] = processor.feature_extractor
    elif "tokenizer" in sig:
        trainer_kwargs["tokenizer"] = processor.feature_extractor

    Seq2SeqTrainer(**trainer_kwargs).train()
    model.save_pretrained(LORA_OUTPUT_DIR)
    del model, alap
    import gc; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def hatter_tanitas_process(hang_utvonal, javitott_szoveg, device):
    """Egyetlen minta LoRA finomhangolása (a többmintás tanító wrappere)."""
    hatter_tanitas_tobb_minta([(hang_utvonal, javitott_szoveg)], device)


# ─────────────────────────────────────────────────────────────────────────────
#  TOOLTIP (SÚGÓ BUBORÉK) – sötét módban, részletes leírásokkal
# ─────────────────────────────────────────────────────────────────────────────

class ToolTip:
    """Egérrel való rámutatáskor részletes súgó buborékot jelenít meg (600 ms késéssel)."""
    def __init__(self, widget, text, delay=600, wrap=420):
        self.widget   = widget
        self.text     = text
        self.delay    = delay
        self.wrap     = wrap
        self.tip_win  = None
        self.after_id = None
        widget.bind("<Enter>",       self._schedule, add="+")
        widget.bind("<Leave>",       self._cancel,   add="+")
        widget.bind("<ButtonPress>", self._cancel,   add="+")

    def _schedule(self, event=None):
        self._cancel()
        self.after_id = self.widget.after(self.delay, self._show)

    def _cancel(self, event=None):
        if self.after_id:
            self.widget.after_cancel(self.after_id)
            self.after_id = None
        self._hide()

    def _show(self):
        if self.tip_win:
            return
        x = self.widget.winfo_rootx() + 16
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 6
        self.tip_win = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        tw.attributes("-topmost", True)
        frame = tk.Frame(tw, background="#00adb5", padx=1, pady=1)
        frame.pack()
        tk.Label(
            frame, text=self.text, justify="left",
            background="#1e2a2b", foreground="#d0f0f0",
            font=("Helvetica", 9), wraplength=self.wrap,
            padx=10, pady=8, relief="flat"
        ).pack()

    def _hide(self):
        if self.tip_win:
            self.tip_win.destroy()
            self.tip_win = None



# ─────────────────────────────────────────────────────────────────────────────
#  FŐ ALKALMAZÁS
# ─────────────────────────────────────────────────────────────────────────────

class VoicetexApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Voicetex v3 – javított kiadás (CT2 fix + szálbiztonság)")
        self.root.geometry("820x820")
        self.root.minsize(700, 700)

        # Sötét mód paletta
        self.BG_DARK  = "#121212"
        self.BG_PANEL = "#1e1e1e"
        self.FG_LIGHT = "#e0e0e0"
        self.FG_ACCENT= "#00adb5"
        self.BG_INPUT = "#252525"
        self.root.configure(bg=self.BG_DARK)

        # Állapotváltozók
        self.hang_sor             = queue.Queue()
        self.is_recording         = False
        self.hang_adatok          = []
        self.utolso_hang_utvonal  = ""
        self.record_start_time    = 0
        self.stop_factory_requested = False
        self.minden_eszkoz_lista  = []
        self.aktualis_vu_szint    = 0.0

        # VAD állapot
        self.vad_aktiv  = False   # VAD mód be/ki
        self.vad_beszel = False   # éppen beszél-e
        self.vad_model  = None    # Silero modell (betöltés után)

        # JAVÍTÁS: a Space-előtag mechanizmus holt kód volt (a hook sehol sem
        # települt fel), ezért teljesen eltávolítva.
        self._paste_target_hwnd = None # utolsó külső ablak, ahová be kell illeszteni

        # faster-whisper modellválasztó állapota
        self.fw_model_name     = self._load_selected_fw_model()
        self.model_loading     = False

        self.auto_paste_var    = tk.BooleanVar(value=True)
        self.auto_tanulas_var  = tk.BooleanVar(value=False)
        self.auto_tanulas_fut  = False   # éppen tanít-e a háttérben
        self.auto_tanulas_varakozik = False # VAD közben nem indítunk tanítást
        self.auto_queue        = self._queue_betoltes()

        # Modell-zár: diktálás és felirat nem futhat egyszerre
        self.model_lock        = threading.Lock()

        self._build_styles()
        self._build_ui()

        self.status_bar = tk.Label(
            self.root, text="⏳ Modell inicializálása...",
            bg=self.BG_PANEL, fg=self.FG_ACCENT,
            anchor="w", padx=6, pady=6, font=("Consolas", 9)
        )
        self.status_bar.pack(fill="x", side="bottom")

        threading.Thread(target=self.init_model, daemon=True).start()

    # ── Stílus ──────────────────────────────────────────────────────────────

    def _build_styles(self):
        s = ttk.Style()
        s.theme_use('default')
        s.configure(".",            background=self.BG_DARK,  foreground=self.FG_LIGHT)
        s.configure("TNotebook",    background=self.BG_DARK,  borderwidth=0)
        s.configure("TNotebook.Tab",background=self.BG_PANEL, foreground=self.FG_LIGHT,
                    padding=[12,6], font=("Helvetica",9,"bold"))
        s.map("TNotebook.Tab",
              background=[("selected", self.FG_ACCENT)],
              foreground=[("selected", "#ffffff")])
        s.configure("TLabelframe",       background=self.BG_PANEL, foreground=self.FG_ACCENT,
                    bordercolor="#333333", borderwidth=1)
        s.configure("TLabelframe.Label", background=self.BG_PANEL, foreground=self.FG_ACCENT,
                    font=("Helvetica",10,"bold"))
        s.configure("TLabel",    background=self.BG_PANEL, foreground=self.FG_LIGHT)
        s.configure("TEntry",    fieldbackground=self.BG_INPUT, foreground=self.FG_LIGHT, borderwidth=0)
        s.configure("TCombobox", fieldbackground=self.BG_INPUT, background=self.BG_PANEL,
                    foreground=self.FG_LIGHT, arrowcolor=self.FG_LIGHT,
                    selectbackground=self.BG_INPUT, selectforeground=self.FG_LIGHT)
        # JAVÍTÁS: a combobox felirata olvashatatlan volt. Két ok:
        # 1) "readonly"/"disabled" állapotban a ttk külön (alapértelmezetten
        #    világos) mezőszínt és kijelölési színt használ, ezért a világos
        #    szöveg világos mezőre került → explicit map minden állapotra;
        # 2) a lenyíló lista nem ttk, hanem sima Tk Listbox, amit csak
        #    option_add-dal lehet sötét módra színezni (lásd lentebb).
        s.map("TCombobox",
              fieldbackground=[("readonly", self.BG_INPUT), ("disabled", self.BG_PANEL)],
              foreground=[("readonly", self.FG_LIGHT), ("disabled", "#777777")],
              selectbackground=[("readonly", self.BG_INPUT)],
              selectforeground=[("readonly", self.FG_LIGHT)],
              arrowcolor=[("disabled", "#555555")])
        # A lenyíló lista (Tk Listbox) színei:
        self.root.option_add("*TCombobox*Listbox.background",       self.BG_INPUT)
        self.root.option_add("*TCombobox*Listbox.foreground",       self.FG_LIGHT)
        self.root.option_add("*TCombobox*Listbox.selectBackground", self.FG_ACCENT)
        self.root.option_add("*TCombobox*Listbox.selectForeground", "#ffffff")
        s.configure("TButton",   background=self.BG_INPUT, foreground=self.FG_LIGHT,
                    borderwidth=1, focuscolor=self.FG_ACCENT)
        s.map("TButton",
              background=[("active", self.FG_ACCENT)],
              foreground=[("active","#ffffff")])
        s.configure("Accent.TButton", background=self.FG_ACCENT, foreground="#ffffff",
                    font=("Helvetica",9,"bold"))
        s.map("Accent.TButton", background=[("active","#007a80")])
        s.configure("VAD.TButton", background="#1a4a1a", foreground="#00ff66",
                    font=("Helvetica",9,"bold"), borderwidth=1)
        s.map("VAD.TButton",
              background=[("active","#00cc44")],
              foreground=[("active","#ffffff")])
        s.configure("VADActive.TButton", background="#006622", foreground="#ffffff",
                    font=("Helvetica",9,"bold"))
        s.configure("VU.Horizontal.TProgressbar",
                    foreground="#00ff66", background="#00ff66",
                    troughcolor=self.BG_INPUT)
        self.style = s

    # ── UI felépítés ─────────────────────────────────────────────────────────

    def _build_ui(self):
        nb = ttk.Notebook(self.root)
        nb.pack(fill="both", expand=True, padx=10, pady=10)
        self.notebook = nb

        self.tab1 = ttk.Frame(nb, style="TNotebook")
        self.tab2 = ttk.Frame(nb, style="TNotebook")
        nb.add(self.tab1, text="🎤 Interaktív Diktálás")
        nb.add(self.tab2, text="🤖 Automata Kötegelt Tanítás")

        self.tab3 = ttk.Frame(nb, style="TNotebook")
        nb.add(self.tab3, text="🎬 Film Felirat")

        self._build_diktalo_tab()
        self._build_factory_tab()
        self._build_felirat_tab()
        self._build_overlay()
        self.root.after(150, self._build_tooltips)


    # ── Passzív önjavító tanulás ──────────────────────────────────────────────

    def _queue_betoltes(self):
        """Auto-tanulás queue betöltése fájlból (vagy üres lista)."""
        try:
            import json
            if os.path.exists(AUTO_QUEUE_FILE):
                with open(AUTO_QUEUE_FILE, encoding="utf-8") as f:
                    q = json.load(f)
                print(f"[AUTO] Queue betöltve: {len(q)} pár", flush=True)
                return q
        except Exception as e:
            print(f"[AUTO] Queue betöltési hiba: {e}", flush=True)
        return []

    def _queue_mentes_fajlba(self):
        """Queue mentése JSON fájlba."""
        try:
            import json
            os.makedirs(DATASET_DIR, exist_ok=True)
            with open(AUTO_QUEUE_FILE, "w", encoding="utf-8") as f:
                json.dump(self.auto_queue, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[AUTO] Queue fájl mentési hiba: {e}", flush=True)

    def _auto_queue_mentes(self, wav_path, text, confidence):
        """Egy jó minőségű pár hozzáadása az auto-tanulás queue-hoz."""
        import json
        entry = {
            "wav":        wav_path,
            "text":       text,
            "confidence": round(confidence, 4),
            "timestamp":  int(time.time())
        }
        self.auto_queue.append(entry)
        self._queue_mentes_fajlba()
        n = len(self.auto_queue)
        print(f"[AUTO] Queue: {n}/{AUTO_TANULAS_KOTEG} – '{text[:40]}'", flush=True)
        # UI frissítés
        self.root.after(0, lambda: self.auto_queue_label.config(
            text=f"📦 {n}/{AUTO_TANULAS_KOTEG}",
            fg="#00ff66" if n >= AUTO_TANULAS_KOTEG else "#888888"
        ))
        self.root.after(0, lambda: self.log_status(
            f"🧠 Auto-queue: {n}/{AUTO_TANULAS_KOTEG} "
            f"(bizalom: {confidence:.2f})"
        ))
        # Refaktor 3: a queue gyűlhet, de a LoRA tanítás alapból NEM indul automatikusan.
        # Ennek oka: élő diktálás közben a tanítás GPU-t, memóriát és modellt foglal,
        # amitől a felhasználó joggal ijed meg és a diktálás is bizonytalanná válhat.
        if n >= AUTO_TANULAS_KOTEG:
            if AUTO_TANULAS_AUTOSTART and not self.auto_tanulas_fut:
                self.root.after(0, self._auto_tanitas_indit)
            else:
                print(
                    f"[AUTO] Queue elérte a tanítási küszöböt ({n}/{AUTO_TANULAS_KOTEG}), "
                    "de az automatikus tanítás biztonsági okból ki van kapcsolva. "
                    "Használd a 'Queue tanítás most' gombot, amikor nem diktálsz.",
                    flush=True
                )
                self.root.after(0, lambda: self.log_status(
                    "🧠 Queue megtelt – tanítás csak kézzel indul, diktálás közben nem."
                ))

    def _auto_tanitas_indit(self):
        """Auto-tanítás elindítása a háttérben (főszálról hívva).

        Refaktor 2:
        VAD közben nem indítunk LoRA tanítást, mert a tanítás és a modell-újratöltés
        ugyanazt a Whisper erőforrást/GPU memóriát piszkálja, mint az élő diktálás.
        Ettől keletkezett a `fw_model` eltűnés és a beillesztés körüli bizonytalanság.
        """
        if self.auto_tanulas_fut or self.is_recording or getattr(self, "model_loading", False):
            return
        if len(self.auto_queue) < AUTO_TANULAS_KOTEG:
            return
        if getattr(self, "vad_aktiv", False):
            self.auto_tanulas_varakozik = True
            self.auto_queue_label.config(text=f"📦 {len(self.auto_queue)}/{AUTO_TANULAS_KOTEG} – vár", fg="#ffaa00")
            self.log_status("🧠 Auto-tanulás várakozik – VAD módban nem indítok modelltréninget.")
            print("[AUTO] Tanítás elhalasztva: VAD aktív, a diktálás stabilitása az első.", flush=True)
            return
        self.auto_tanulas_fut = True
        self.auto_tanulas_varakozik = False
        self.auto_queue_label.config(text="🔄 Tanítás...", fg="#ffaa00")
        self.log_status("🧠 Auto-tanítás indul a háttérben...")
        koteg = self.auto_queue[:AUTO_TANULAS_KOTEG]
        threading.Thread(
            target=self._auto_tanitas_thread,
            args=(koteg,),
            daemon=True
        ).start()

    def _auto_tanitas_thread(self, koteg):
        """
        Háttérszálon: a queue első N párjára LoRA tanítás,
        majd CT2 konverzió és modell újratöltés.
        """
        try:
            print(f"[AUTO] Tanítás indul – {len(koteg)} pár", flush=True)
            self.root.after(0, lambda: self.log_status(
                f"🧠 Auto-tanítás: {len(koteg)} pár feldolgozása..."))

            # Refaktor 2:
            # Itt korábban `del self.fw_model` futott. Ez élő VAD/diktálás mellett
            # versenyhelyzetet okozott: a run_inference néha éppen akkor indult,
            # amikor a modell attribútum már nem létezett.
            # Nem töröljük közvetlenül az aktív modellt; a betöltést/cserét
            # kizárólag a load_active_model végzi, saját model_lock alatt.
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # OPTIMALIZÁLÁS: korábban páronként külön tanítás futott, ami minden
            # párnál újratöltötte a 3GB-os alapmodellt. Most az összes érvényes
            # pár EGY tanítási menetben megy le → ~15x kevesebb modellbetöltés.
            ervenyes_parok = []
            for par in koteg:
                if os.path.exists(par["wav"]):
                    ervenyes_parok.append((par["wav"], par["text"]))
                else:
                    print(f"[AUTO] Hiányzó fájl, kihagyva: {par['wav']}", flush=True)
            if ervenyes_parok:
                self.root.after(0, lambda n=len(ervenyes_parok): self.log_status(
                    f"🧠 Auto-tanítás: {n} pár egy menetben..."))
                print(f"[AUTO] Tanítás egy menetben: {len(ervenyes_parok)} pár", flush=True)
                hatter_tanitas_tobb_minta(ervenyes_parok, self.device)

            # Zip backup + CT2 konverzió
            lora_automatikus_mentes()
            konv_ok = lora_merge_es_ct2_konvertalas(self.log_status)
            # JAVÍTÁS: sikeres konverzió után az egyedi CT2 modell lép életbe.
            self.load_active_model(CUSTOM_CT2_ID if konv_ok else None)

            # Sikeres tanítás után töröljük a feldolgozott párokat
            self.auto_queue = self.auto_queue[len(koteg):]
            self._queue_mentes_fajlba()
            n = len(self.auto_queue)

            print(f"[AUTO] Tanítás kész! Maradék queue: {n}", flush=True)
            self.root.after(0, lambda: self.auto_queue_label.config(
                text=f"📦 {n}/{AUTO_TANULAS_KOTEG}", fg="#888888"))
            self.root.after(0, lambda: self.log_status(
                "✅ Auto-tanítás kész – modell frissítve!"))
            gep_beszel_magyarul("Az automatikus tanítás sikeresen befejeződött.")

        except Exception as e:
            print(f"[AUTO] Tanítási hiba: {e}", flush=True)
            self.root.after(0, lambda m=str(e): self.log_status(
                f"⚠️ Auto-tanítás hiba: {m[:60]}"))
            try:
                self.load_active_model()
            except Exception:
                pass
        finally:
            self.auto_tanulas_fut = False
            # Ha közben VAD be van kapcsolva, nem próbáljuk újra azonnal a tanítást.
            # A felhasználó diktálása elsőbbséget élvez.

    def _build_diktalo_tab(self):
        panel = ttk.Frame(self.tab1, style="TNotebook")
        panel.pack(fill="both", expand=True)

        # ── Vezérlés panel ──────────────────────────────────────────────────
        ctrl = ttk.LabelFrame(panel, text=" Vezérlés és Mikrofon ")
        ctrl.pack(fill="x", padx=15, pady=10, ipady=5)

        dev_row = tk.Frame(ctrl, bg=self.BG_PANEL)
        dev_row.pack(fill="x", pady=(5,10))
        ttk.Label(dev_row, text="Bemenet:", font=("Helvetica",9,"bold")).pack(side="left", padx=5)
        self.device_combo = ttk.Combobox(dev_row, state="readonly", width=48)
        self.device_combo.pack(side="left", padx=5, fill="x", expand=True)

        model_row = tk.Frame(ctrl, bg=self.BG_PANEL)
        model_row.pack(fill="x", pady=(0,10))
        ttk.Label(model_row, text="Whisper modell:", font=("Helvetica",9,"bold")).pack(side="left", padx=5)
        self.model_combo = ttk.Combobox(
            model_row, state="disabled", width=48,
            values=[nev for nev, _azon in FW_MODEL_CHOICES]
        )
        self.model_combo.set(self._model_display_from_id(self.fw_model_name))
        self.model_combo.pack(side="left", padx=5, fill="x", expand=True)
        self.reload_model_btn = ttk.Button(
            model_row, text="🔄 Modell betöltése",
            command=self.reload_model_from_combo, state="disabled"
        )
        self.reload_model_btn.pack(side="left", padx=5)

        btn_row = tk.Frame(ctrl, bg=self.BG_PANEL)
        btn_row.pack(fill="x", padx=5, pady=(0,8))

        # PTT gomb
        self.record_btn = ttk.Button(
            btn_row, text="🎤 PTT – Tartsd nyomva",
            command=self.toggle_recording, state="disabled",
            style="Accent.TButton"
        )
        self.record_btn.pack(side="left", padx=5)

        # VAD gomb
        self.vad_btn = ttk.Button(
            btn_row, text="🤖 VAD mód: KI",
            command=self.toggle_vad, state="disabled",
            style="VAD.TButton"
        )
        self.vad_btn.pack(side="left", padx=5)

        self.time_label = ttk.Label(btn_row, text="⏱ 0.0 mp", font=("Helvetica",10))
        self.time_label.pack(side="left", padx=10)

        self.hotkey_label = ttk.Label(
            btn_row, text="💡 PTT: CTRL+WIN",
            font=("Helvetica",9,"bold"), foreground="#00ff66"
        )
        self.hotkey_label.pack(side="right", padx=5)

        # ── VU méter ────────────────────────────────────────────────────────
        vu_frame = ttk.LabelFrame(panel, text=" Mikrofon szint (VU) ")
        vu_frame.pack(fill="x", padx=15, pady=5)
        self.vu_meter = ttk.Progressbar(
            vu_frame, orient="horizontal", mode="determinate",
            style="VU.Horizontal.TProgressbar"
        )
        self.vu_meter.pack(fill="x", expand=True, padx=5, pady=5)

        # ── Szöveg területek ────────────────────────────────────────────────
        txt = tk.Frame(panel, bg=self.BG_DARK)
        txt.pack(fill="both", expand=True, padx=15, pady=5)

        tk.Label(txt, text="Amit a Whisper hallott:",
                 font=("Helvetica",9,"bold"), fg=self.FG_ACCENT, bg=self.BG_DARK
                 ).pack(anchor="w", pady=(5,0))
        self.ai_text = tk.Text(
            txt, height=4, bg=self.BG_INPUT, fg=self.FG_LIGHT,
            insertbackground=self.FG_LIGHT, font=("Helvetica",11),
            wrap="word", borderwidth=0, padx=5, pady=5
        )
        self.ai_text.pack(fill="x", pady=5)
        self.ai_text.config(state="disabled")

        tk.Label(txt, text="Javítás (ebből tanul):",
                 font=("Helvetica",9,"bold"), fg=self.FG_LIGHT, bg=self.BG_DARK
                 ).pack(anchor="w", pady=(10,0))
        self.user_text = tk.Text(
            txt, height=6, bg=self.BG_INPUT, fg=self.FG_LIGHT,
            insertbackground=self.FG_LIGHT, font=("Helvetica",11),
            wrap="word", borderwidth=0, padx=5, pady=5
        )
        self.user_text.pack(fill="both", expand=True, pady=5)

        btn_f = tk.Frame(panel, bg=self.BG_DARK)
        btn_f.pack(fill="x", padx=15, pady=10)

        self.auto_paste_cb = tk.Checkbutton(
            btn_f,
            text="📋 Auto-beillesztés",
            variable=self.auto_paste_var,
            bg=self.BG_DARK, fg=self.FG_LIGHT,
            selectcolor=self.BG_INPUT,
            activebackground=self.BG_DARK, activeforeground=self.FG_ACCENT,
            font=("Helvetica", 9)
        )
        self.auto_paste_cb.pack(side="left", padx=5)

        self.auto_tanulas_cb = tk.Checkbutton(
            btn_f,
            text="🧠 Tanító queue gyűjtés",
            variable=self.auto_tanulas_var,
            bg=self.BG_DARK, fg="#00ff66",
            selectcolor=self.BG_INPUT,
            activebackground=self.BG_DARK, activeforeground="#00ff66",
            font=("Helvetica", 9)
        )
        self.auto_tanulas_cb.pack(side="left", padx=5)

        self.auto_queue_label = tk.Label(
            btn_f, text="📦 0/15",
            bg=self.BG_DARK, fg="#888888",
            font=("Helvetica", 9)
        )
        self.auto_queue_label.pack(side="left", padx=2)

        self.auto_queue_train_btn = ttk.Button(
            btn_f, text="🧠 Queue tanítás most",
            command=self._auto_tanitas_indit
        )
        self.auto_queue_train_btn.pack(side="left", padx=5)

        self.train_btn = ttk.Button(
            btn_f, text="💾 Összehasonlítás, Tanulás & Backup",
            command=self.start_learning, state="disabled"
        )
        self.train_btn.pack(side="right", padx=5)

    def _build_factory_tab(self):
        panel = ttk.Frame(self.tab2, style="TNotebook")
        panel.pack(fill="both", expand=True)

        fajl_f = ttk.LabelFrame(panel, text=" Fájlok ")
        fajl_f.pack(fill="x", padx=15, pady=10, ipady=5)

        for row, (lbl, attr, cmd) in enumerate([
            ("1. Hanganyag (.wav):", "wave_path_entry", "browse_wave"),
            ("2. Kézirat (.txt):",   "txt_path_entry",  "browse_txt"),
        ]):
            tk.Label(fajl_f, text=lbl, font=("Helvetica",9,"bold"),
                     bg=self.BG_PANEL, fg=self.FG_LIGHT
                     ).grid(row=row, column=0, sticky="w", padx=10, pady=8)
            entry = ttk.Entry(fajl_f, width=42)
            entry.grid(row=row, column=1, padx=5, pady=8)
            setattr(self, attr, entry)
            ttk.Button(fajl_f, text="Tallózás...",
                       command=getattr(self, cmd)
                       ).grid(row=row, column=2, padx=5, pady=8)

        proc_f = tk.Frame(panel, bg=self.BG_DARK)
        proc_f.pack(fill="both", expand=True, padx=15, pady=5)

        self.factory_log = tk.Text(
            proc_f, height=14, bg="#111111", fg="#00ff66",
            insertbackground="#00ff66", font=("Consolas",10),
            borderwidth=0, padx=5, pady=5
        )
        self.factory_log.pack(fill="both", expand=True, pady=5)
        self.factory_log.tag_configure("warning", foreground="#ffaa00")
        self.factory_log.tag_configure("error",   foreground="#ff4444")
        self.factory_log.tag_configure("success",  foreground="#00ff66")
        self.factory_log.tag_configure("info",     foreground="#00adb5")

        self.factory_progress = ttk.Progressbar(proc_f, mode="determinate")
        self.factory_progress.pack(fill="x", pady=5)

        act_f = tk.Frame(proc_f, bg=self.BG_DARK)
        act_f.pack(fill="x", pady=5)
        self.stop_factory_btn = ttk.Button(
            act_f, text="🛑 Leállítás",
            command=self.request_factory_stop, state="disabled"
        )
        self.stop_factory_btn.pack(side="left")
        self.start_factory_btn = ttk.Button(
            act_f, text="🚀 Kötegelt Tanítás Indítása",
            command=self.start_batch_processing,
            state="disabled", style="Accent.TButton"
        )
        self.start_factory_btn.pack(side="right")


    # ── Film Felirat fül ─────────────────────────────────────────────────────

    def _build_felirat_tab(self):
        panel = ttk.Frame(self.tab3, style="TNotebook")
        panel.pack(fill="both", expand=True)

        fajl_f = ttk.LabelFrame(panel, text=" Videófájl ")
        fajl_f.pack(fill="x", padx=15, pady=10, ipady=5)
        tk.Label(fajl_f, text="Videó:", font=("Helvetica", 9, "bold"),
                 bg=self.BG_PANEL, fg=self.FG_LIGHT
                 ).grid(row=0, column=0, sticky="w", padx=10, pady=8)
        self.felirat_video_entry = ttk.Entry(fajl_f, width=50)
        self.felirat_video_entry.grid(row=0, column=1, padx=5, pady=8)
        ttk.Button(fajl_f, text="Tallózás...",
                   command=self._felirat_talloz
                   ).grid(row=0, column=2, padx=5, pady=8)

        opt_f = ttk.LabelFrame(panel, text=" Beállítások ")
        opt_f.pack(fill="x", padx=15, pady=5, ipady=5)
        tk.Label(opt_f, text="Forrás nyelv:", font=("Helvetica", 9),
                 bg=self.BG_PANEL, fg=self.FG_LIGHT
                 ).grid(row=0, column=0, sticky="w", padx=10, pady=6)
        self.felirat_forras_var = tk.StringVar(value="en")
        ttk.Combobox(opt_f, textvariable=self.felirat_forras_var,
                     values=["en","de","fr","es","it","ru","ja","zh"],
                     width=6, state="readonly").grid(row=0, column=1, sticky="w", padx=5, pady=6)
        tk.Label(opt_f, text="→  Magyar fordítás:",
                 font=("Helvetica", 9), bg=self.BG_PANEL, fg=self.FG_LIGHT
                 ).grid(row=0, column=2, sticky="w", padx=10, pady=6)
        self.felirat_forditas_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(opt_f, variable=self.felirat_forditas_var
                        ).grid(row=0, column=3, sticky="w", padx=5)
        tk.Label(opt_f, text="Csak fordítás (meglévő SRT):",
                 font=("Helvetica", 9), bg=self.BG_PANEL, fg=self.FG_LIGHT
                 ).grid(row=0, column=4, sticky="w", padx=10, pady=6)
        self.felirat_csak_forditas_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(opt_f, variable=self.felirat_csak_forditas_var
                        ).grid(row=0, column=5, sticky="w", padx=5)

        tk.Label(opt_f, text="💬 Párbeszéd összevonás:",
                 font=("Helvetica", 9), bg=self.BG_PANEL, fg=self.FG_LIGHT
                 ).grid(row=1, column=0, columnspan=2, sticky="w", padx=10, pady=6)
        self.felirat_parbeszed_var = tk.DoubleVar(value=0.8)
        self._parbeszed_ertek_lbl = tk.Label(
            opt_f, text="0.8 mp", width=6,
            font=("Helvetica", 9, "bold"), bg=self.BG_PANEL, fg="#00adb5")
        self._parbeszed_ertek_lbl.grid(row=1, column=3, sticky="w", padx=5)
        def _parbeszed_frissit(v):
            val = float(v)
            if val < 0.1:
                self._parbeszed_ertek_lbl.config(text="Ki", fg="#888888")
            else:
                self._parbeszed_ertek_lbl.config(text=f"{val:.1f} mp", fg="#00adb5")
        tk.Scale(
            opt_f, from_=0.0, to=2.0, resolution=0.1,
            orient="horizontal", length=200,
            variable=self.felirat_parbeszed_var,
            command=_parbeszed_frissit,
            bg=self.BG_PANEL, fg=self.FG_LIGHT,
            troughcolor="#2d333b", highlightthickness=0,
            showvalue=False
        ).grid(row=1, column=2, sticky="w", padx=5, pady=4)
        tk.Label(opt_f, text="(0 = kikapcsolva)",
                 font=("Helvetica", 8), bg=self.BG_PANEL, fg="#666666"
                 ).grid(row=1, column=4, columnspan=2, sticky="w", padx=5)

        log_f = tk.Frame(panel, bg=self.BG_DARK)
        log_f.pack(fill="both", expand=True, padx=15, pady=5)
        self.felirat_log = tk.Text(
            log_f, height=12, bg="#0d1117", fg="#c9d1d9",
            insertbackground="#c9d1d9", font=("Consolas", 10),
            borderwidth=0, padx=6, pady=6
        )
        self.felirat_log.pack(fill="both", expand=True, pady=5)
        self.felirat_log.tag_configure("ok",      foreground="#3fb950")
        self.felirat_log.tag_configure("info",    foreground="#58a6ff")
        self.felirat_log.tag_configure("warning", foreground="#d29922")
        self.felirat_log.tag_configure("error",   foreground="#f85149")
        self.felirat_progress = ttk.Progressbar(log_f, mode="determinate")
        self.felirat_progress.pack(fill="x", pady=4)

        dikt_f = tk.Frame(panel, bg=self.BG_DARK)
        dikt_f.pack(fill="x", padx=15, pady=(0, 4))
        tk.Label(dikt_f, text="🎤 Utolsó diktálás:",
                 font=("Helvetica", 8), bg=self.BG_DARK, fg="#888888"
                 ).pack(side="left", padx=(0, 6))
        self.felirat_diktalt_label = tk.Label(
            dikt_f, text="–", font=("Helvetica", 9, "italic"),
            bg=self.BG_DARK, fg="#00adb5", anchor="w", wraplength=550
        )
        self.felirat_diktalt_label.pack(side="left", fill="x", expand=True)

        btn_f = tk.Frame(panel, bg=self.BG_DARK)
        btn_f.pack(fill="x", padx=15, pady=8)
        self.felirat_stop_btn = ttk.Button(
            btn_f, text="🛑 Leállítás",
            command=self._felirat_stop, state="disabled"
        )
        self.felirat_stop_btn.pack(side="left")
        self.felirat_start_btn = ttk.Button(
            btn_f, text="🎬 Felirat Generálása",
            command=self._felirat_indit, style="Accent.TButton"
        )
        self.felirat_start_btn.pack(side="right")
        self._felirat_leall = False
        self._fordito_modell = None
        self._fordito_tokenizer = None

    def _felirat_talloz(self):
        fajl = filedialog.askopenfilename(
            title="Videófájl kiválasztása",
            filetypes=[
                ("Videófájlok", "*.mp4 *.mkv *.avi *.mov *.wmv *.flv *.webm *.ts *.m4v"),
                ("SRT felirat", "*.srt"),
                ("Minden fájl", "*.*"),
            ]
        )
        if fajl:
            self.felirat_video_entry.delete(0, tk.END)
            self.felirat_video_entry.insert(0, fajl)

    def _felirat_log_write(self, uzenet, tag=""):
        def _do():
            self.felirat_log.config(state="normal")
            self.felirat_log.insert(tk.END, uzenet + "\n", tag)
            self.felirat_log.see(tk.END)
            self.felirat_log.config(state="disabled")
        self.root.after(0, _do)

    def _felirat_stop(self):
        self._felirat_leall = True
        self._felirat_log_write("⏹ Leállítás kérve...", "warning")

    def _felirat_indit(self):
        video_ut = self.felirat_video_entry.get().strip()
        if not video_ut or not os.path.exists(video_ut):
            messagebox.showerror("Hiba", "Kérlek válassz ki egy létező videó- vagy SRT-fájlt!")
            return
        self._felirat_leall = False
        # JAVÍTÁS: Tk-változók kiolvasása a főszálon; a háttérszál csak ezt
        # a pillanatképet használja (Tk szálbiztonság).
        self._felirat_opciok = {
            "csak_forditas":    self.felirat_csak_forditas_var.get(),
            "forditas_kell":    self.felirat_forditas_var.get(),
            "forras_nyelv":     self.felirat_forras_var.get(),
            "parbeszed_kuszob": self.felirat_parbeszed_var.get(),
        }
        self.felirat_start_btn.config(state="disabled")
        self.felirat_stop_btn.config(state="normal")
        self.felirat_log.config(state="normal")
        self.felirat_log.delete("1.0", tk.END)
        self.felirat_log.config(state="disabled")
        self.felirat_progress.config(value=0)
        threading.Thread(target=self._felirat_thread, args=(video_ut,), daemon=True).start()

    def _felirat_thread(self, video_ut):
        try:
            opciok = getattr(self, "_felirat_opciok", {})
            csak_forditas = opciok.get("csak_forditas", False)
            forditas_kell = opciok.get("forditas_kell", True)
            forras_nyelv  = opciok.get("forras_nyelv", "en")
            alap_nev      = os.path.splitext(video_ut)[0]
            if csak_forditas:
                srt_ut = video_ut if video_ut.lower().endswith(".srt") else alap_nev + ".srt"
                if not os.path.exists(srt_ut):
                    self._felirat_log_write(f"❌ SRT nem található: {srt_ut}", "error")
                    return
                self._felirat_log_write(f"📂 Meglévő SRT betöltve: {srt_ut}", "info")
                szegmensek = self._srt_beolvasas(srt_ut)
            else:
                self._felirat_log_write("🔊 1/3 – Hang kinyerése a videóból...", "info")
                temp_wav = alap_nev + "_temp_audio.wav"
                if not self._hang_kinyeres(video_ut, temp_wav):
                    return
                if self._felirat_leall:
                    return
                self._felirat_log_write("✍️  2/3 – Átírás faster-whisper-rel...", "info")
                srt_ut = alap_nev + f"_{forras_nyelv}.srt"
                szegmensek = self._atiras_srt(temp_wav, srt_ut, forras_nyelv)
                if szegmensek is None:
                    return
                if os.path.exists(temp_wav):
                    os.remove(temp_wav)
                self._felirat_log_write(f"✅ SRT mentve: {srt_ut}", "ok")
            if self._felirat_leall:
                return
            if forditas_kell and FORDITO_ELERHETO:
                self._felirat_log_write("🇭🇺 3/3 – Magyar fordítás...", "info")
                srt_hu_ut = alap_nev + "_hu.srt"
                self._srt_forditas(szegmensek, srt_hu_ut)
                if not self._felirat_leall:
                    self._felirat_log_write(f"✅ Magyar SRT mentve: {srt_hu_ut}", "ok")
            elif forditas_kell and not FORDITO_ELERHETO:
                self._felirat_log_write("⚠️ transformers nem elérhető – fordítás kihagyva.", "warning")
            if not self._felirat_leall:
                self._felirat_log_write("\n🏁 KÉSZ! A feliratfájl a videó mappájában van.", "ok")
                self.root.after(0, lambda: messagebox.showinfo("Kész", f"Felirat elkészült!\n{srt_ut}"))
        except Exception as e:
            import traceback
            self._felirat_log_write(f"❌ Hiba: {e}", "error")
            self._felirat_log_write(traceback.format_exc(), "error")
        finally:
            self.root.after(0, lambda: self.felirat_start_btn.config(state="normal"))
            self.root.after(0, lambda: self.felirat_stop_btn.config(state="disabled"))

    def _hang_kinyeres(self, video_ut, cel_wav):
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            self._felirat_log_write(
                "❌ ffmpeg nincs telepítve!\n   https://ffmpeg.org/download.html", "error")
            return False
        try:
            cmd = [ffmpeg, "-y", "-i", video_ut,
                   "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", cel_wav]
            eredmeny = subprocess.run(cmd, capture_output=True, text=True)
            if eredmeny.returncode != 0:
                self._felirat_log_write(f"❌ ffmpeg hiba:\n{eredmeny.stderr[-800:]}", "error")
                return False
            self._felirat_log_write(f"   ✔ Hang kinyerve: {os.path.basename(cel_wav)}", "ok")
            return True
        except Exception as e:
            self._felirat_log_write(f"❌ ffmpeg indítási hiba: {e}", "error")
            return False

    def _atiras_srt(self, wav_ut, srt_ut, nyelv="en"):
        if not hasattr(self, "fw_model") or self.fw_model is None:
            self._felirat_log_write("⏳ Whisper modell betöltése...", "info")
            self.load_active_model()
        try:
            self._felirat_log_write(f"   Átírás indul ({nyelv})...", "info")
            szegmensek = []
            with self.model_lock:
                fw = getattr(self, "fw_model", None)
                if fw is None:
                    raise RuntimeError("A Whisper modell nincs betöltve.")
                szegmensek_gen, info = fw.transcribe(
                    wav_ut, language=nyelv, beam_size=5,
                    vad_filter=True, word_timestamps=True,
                )
                for s in szegmensek_gen:
                    if self._felirat_leall:
                        break
                    szegmensek.append(s)
                    if len(szegmensek) % 20 == 0:
                        self._felirat_log_write(f"   ... {len(szegmensek)} szegmens kész...", "info")
            if not szegmensek:
                self._felirat_log_write("⚠️ Nem sikerült szöveget felismerni.", "warning")
                return None
            self._felirat_log_write(
                f"   ✔ {len(szegmensek)} szegmens felismerve ({info.duration:.1f}s)", "ok")
            kuszob = getattr(self, "_felirat_opciok", {}).get("parbeszed_kuszob", 0.8)
            items  = self._parbeszed_merge(szegmensek, kuszob)
            parbeszed_db = sum(1 for it in items if it['tipus']=='parbeszed')
            if parbeszed_db:
                self._felirat_log_write(
                    f"   💬 {parbeszed_db} párbeszéd-pár összevonva", "info")
            with open(srt_ut, "w", encoding="utf-8") as f:
                for idx, it in enumerate(items, 1):
                    kezd = it['start']
                    veg  = (min(it['end_raw'] + 2.0, it['next_start'])
                            if it['next_start'] else it['end_raw'] + 2.0)
                    f.write(f"{idx}\n")
                    f.write(f"{self._mp_ido(kezd)} --> {self._mp_ido(veg)}\n")
                    f.write(f"{it['text']}\n\n")
            self.root.after(0, lambda: self.felirat_progress.config(value=33))
            return szegmensek
        except Exception as e:
            self._felirat_log_write(f"❌ Átírási hiba: {e}", "error")
            return None

    def _parbeszed_merge(self, szegmensek, kuszob):
        """Szomszéd szegmenseket párbeszéddé von össze, ha a köztük lévő szünet
        kisebb mint kuszob másodperc. Visszatér item-listával:
        {'start', 'end_raw', 'text', 'texts': [str,...], 'next_start'}"""
        def _seg_kezd(s):
            return s.words[0].start if (hasattr(s,'words') and s.words) else s.start
        def _seg_veg(s):
            return s.words[-1].end  if (hasattr(s,'words') and s.words) else s.end

        items = []
        i = 0
        while i < len(szegmensek):
            s = szegmensek[i]
            sk = _seg_kezd(s)
            sv = _seg_veg(s)
            if kuszob > 0 and i + 1 < len(szegmensek):
                ns = szegmensek[i + 1]
                gap = _seg_kezd(ns) - sv
                if 0 <= gap < kuszob:
                    # párbeszéd pár
                    nsk = _seg_kezd(ns)
                    nsv = _seg_veg(ns)
                    next_start = _seg_kezd(szegmensek[i+2]) if i+2 < len(szegmensek) else None
                    items.append({
                        'start':      sk,
                        'end_raw':    nsv,
                        'text':       f"– {s.text.strip()}\n– {ns.text.strip()}",
                        'texts':      [s.text.strip(), ns.text.strip()],
                        'next_start': next_start,
                        'tipus':      'parbeszed'
                    })
                    i += 2
                    continue
            next_start = _seg_kezd(szegmensek[i+1]) if i+1 < len(szegmensek) else None
            items.append({
                'start':      sk,
                'end_raw':    sv,
                'text':       s.text.strip(),
                'texts':      [s.text.strip()],
                'next_start': next_start,
                'tipus':      'egyedi'
            })
            i += 1
        return items

    def _mp_ido(self, masodpercek):
        h  = int(masodpercek // 3600)
        m  = int((masodpercek % 3600) // 60)
        s  = int(masodpercek % 60)
        ms = int((masodpercek - int(masodpercek)) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    def _srt_beolvasas(self, srt_ut):
        szegmensek = []
        try:
            with open(srt_ut, encoding="utf-8", errors="ignore") as f:
                tartalom = f.read()
            blokkok = re.split(r'\n\s*\n', tartalom.strip())
            for blokk in blokkok:
                sorok = blokk.strip().splitlines()
                if len(sorok) < 3:
                    continue
                szegmensek.append({"idosor": sorok[1], "szoveg": " ".join(sorok[2:]).strip()})
        except Exception as e:
            self._felirat_log_write(f"⚠️ SRT beolvasási hiba: {e}", "warning")
        return szegmensek

    def _fordito_betoltes(self):
        if self._fordito_modell is not None:
            return True
        try:
            # Lazy import: a transformers csak itt töltődik be, nem app-indításkor.
            from transformers import MarianMTModel, MarianTokenizer
            os.makedirs(FORDITO_CACHE_DIR, exist_ok=True)
            helyi_ok = os.path.exists(os.path.join(FORDITO_CACHE_DIR, "config.json"))
            if helyi_ok:
                self._felirat_log_write("   ⏳ Fordítómodell betöltése helyi cache-ből...", "info")
                self._fordito_tokenizer = MarianTokenizer.from_pretrained(
                    FORDITO_CACHE_DIR, local_files_only=True)
                self._fordito_modell = MarianMTModel.from_pretrained(
                    FORDITO_CACHE_DIR, local_files_only=True)
            else:
                self._felirat_log_write(
                    f"   ⏳ Fordítómodell letöltése (csak egyszer): {FORDITO_MODELL}\n"
                    "      (~307MB – ezután helyi cache-ből tölt)", "info")
                self._fordito_tokenizer = MarianTokenizer.from_pretrained(FORDITO_MODELL)
                self._fordito_modell    = MarianMTModel.from_pretrained(FORDITO_MODELL)
                self._fordito_tokenizer.save_pretrained(FORDITO_CACHE_DIR)
                self._fordito_modell.save_pretrained(FORDITO_CACHE_DIR)
                self._felirat_log_write("   💾 Modell elmentve helyi cache-be.", "ok")
            self._fordito_modell.eval()
            # OPTIMALIZÁLÁS: korábban a fordítás CUDA mellett is CPU-n futott.
            # GPU-n fp16-tal a feliratfordítás nagyságrenddel gyorsabb.
            self._fordito_device = "cuda" if torch.cuda.is_available() else "cpu"
            if self._fordito_device == "cuda":
                self._fordito_modell = self._fordito_modell.half().to("cuda")
            self._felirat_log_write(
                f"   ✔ Fordítómodell kész ({self._fordito_device}).", "ok")
            return True
        except Exception as e:
            self._felirat_log_write(f"❌ Fordítómodell betöltési hiba: {e}", "error")
            return False

    def _forditas_koteg(self, mondatok):
        if not mondatok:
            return []
        tokenizalt = self._fordito_tokenizer(
            mondatok, return_tensors="pt",
            padding=True, truncation=True, max_length=512
        )
        dev = getattr(self, "_fordito_device", "cpu")
        if dev == "cuda":
            tokenizalt = {k: v.to("cuda") for k, v in tokenizalt.items()}
        with torch.inference_mode():
            forditas_ids = self._fordito_modell.generate(**tokenizalt, num_beams=4, max_length=512)
        return [self._fordito_tokenizer.decode(t, skip_special_tokens=True) for t in forditas_ids]

    def _srt_forditas(self, szegmensek, cel_srt):
        if not self._fordito_betoltes():
            return
        # item-alapú fordítás (párbeszéd-tudatos)
        kuszob = getattr(self, "_felirat_opciok", {}).get("parbeszed_kuszob", 0.8)
        if any(not isinstance(s, dict) for s in szegmensek):
            items = self._parbeszed_merge(szegmensek, kuszob)
        else:
            # dict-alapú (csak-fordítás mód) – eredeti idosor megtartva
            items = [{'start': None, 'end_raw': None,
                      'text': s['szoveg'], 'texts': [s['szoveg']],
                      'next_start': None, 'idosor': s['idosor'],
                      'tipus': 'dict'} for s in szegmensek]

        # összes fordítandó mondat összegyűjtése
        osszes_mondat = []
        for it in items:
            osszes_mondat.extend(it['texts'])

        osszes = len(osszes_mondat)
        ford_lista = []
        for kezdet in range(0, osszes, FELIRAT_KOTEG_MERET):
            if self._felirat_leall:
                break
            vege = min(kezdet + FELIRAT_KOTEG_MERET, osszes)
            try:
                ford_lista.extend(self._forditas_koteg(osszes_mondat[kezdet:vege]))
            except Exception as e:
                self._felirat_log_write(f"⚠️ Fordítási hiba: {e}", "warning")
                ford_lista.extend(osszes_mondat[kezdet:vege])
            haladas = int((vege / osszes) * 66) + 33
            self.root.after(0, lambda v=haladas: self.felirat_progress.config(value=v))
            self._felirat_log_write(f"   🔄 {vege}/{osszes} mondat lefordítva...", "info")

        # fordítások visszaosztása itemekre
        ptr = 0
        with open(cel_srt, "w", encoding="utf-8") as f:
            for idx, it in enumerate(items, 1):
                n = len(it['texts'])
                ford_szovegek = ford_lista[ptr:ptr+n]
                ptr += n
                # JAVÍTÁS: leállításkor a fordításlista rövidebb lehet – a hiányzó
                # fordítás helyére az eredeti szöveg kerül (nincs IndexError).
                if len(ford_szovegek) < n:
                    ford_szovegek = ford_szovegek + it['texts'][len(ford_szovegek):]
                f.write(f"{idx}\n")
                if it['tipus'] == 'dict':
                    f.write(f"{it['idosor']}\n")
                    f.write(f"{' '.join(ford_szovegek)}\n\n")
                elif it['tipus'] == 'parbeszed':
                    veg = (min(it['end_raw'] + 2.0, it['next_start'])
                           if it['next_start'] else it['end_raw'] + 2.0)
                    f.write(f"{self._mp_ido(it['start'])} --> {self._mp_ido(veg)}\n")
                    f.write(f"– {ford_szovegek[0]}\n– {ford_szovegek[1]}\n\n")
                else:
                    veg = (min(it['end_raw'] + 2.0, it['next_start'])
                           if it['next_start'] else it['end_raw'] + 2.0)
                    f.write(f"{self._mp_ido(it['start'])} --> {self._mp_ido(veg)}\n")
                    f.write(f"{ford_szovegek[0]}\n\n")
        self.root.after(0, lambda: self.felirat_progress.config(value=100))

    def _build_overlay(self):
        self.overlay = tk.Toplevel(self.root)
        self.overlay.title("Voicetex HUD")
        sw, sh = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        self.overlay.geometry(f"260x55+{sw-290}+{sh-80}")
        self.overlay.attributes("-topmost", True)
        self.overlay.overrideredirect(True)
        self.overlay.attributes("-alpha", 0.92)
        self.overlay.configure(bg="#1a1a1a")

        self.overlay_label = tk.Label(
            self.overlay, text="🎙️ PTT STANDBY",
            font=("Helvetica",10,"bold"), fg="#888888", bg="#1a1a1a"
        )
        self.overlay_label.pack(fill="x", pady=4)
        self.overlay_vu = ttk.Progressbar(
            self.overlay, orient="horizontal", mode="determinate",
            length=240, style="VU.Horizontal.TProgressbar"
        )
        self.overlay_vu.pack(padx=10, fill="x")

        self.overlay.bind("<Button-1>", lambda e: setattr(self, '_drag_xy', (e.x, e.y)))
        self.overlay.bind("<B1-Motion>", self._drag_overlay)
        self.overlay.withdraw()
        self.root.after(100, self._make_overlay_noactivate)
        self._keep_overlay_top()


    def _build_tooltips(self):
        """Részletes súgó buborékok minden vezérlőhöz."""
        T = ToolTip

        T(self.device_combo,
          "MIKROFON / BEMENETI ESZKÖZ\n\n"
          "Ebből a listából választhatod ki, melyik hangforrást használja az alkalmazás. "
          "Az app automatikusan előnyben részesíti az USB-s, AudioBox vagy '96'-os nevű "
          "eszközt, ha van ilyen csatlakoztatva.\n\n"
          "Ha több mikrofont is használsz, itt válthatsz közöttük. "
          "A változtatás a következő felvételnél lép életbe.")

        T(self.model_combo,
          "WHISPER MODELLVÁLASZTÓ\n\n"
          "Itt választhatod ki, melyik faster-whisper modellt használja a diktálás és a feliratmodul. "
          "A kisebb modellek gyorsabbak, de gyengébbek. A large-v2 általában stabil magyar diktáláshoz, "
          "a large-v3 pontosabb lehet, de lassabb és több memóriát kér.\n\n"
          "Választás után nyomd meg a 'Modell betöltése' gombot. A betöltés idejére a diktálás letiltódik.")

        T(self.reload_model_btn,
          "MODELL BETÖLTÉSE / CSERÉJE\n\n"
          "A kiválasztott Whisper modellt tölti be újra. Diktálás vagy VAD közben nem érdemes modellt váltani, "
          "ezért az app ilyenkor leállítja vagy letiltja a felvételt. Az utoljára kiválasztott modell külön JSON beállításfájlba mentődik.")

        T(self.record_btn,
          "PUSH-TO-TALK (PTT) – KÉZZEL VEZÉRELT FELVÉTEL\n\n"
          "Kattints egyszer a felvétel indításához, majd újra a leállításhoz. "
          "Vagy tartsd lenyomva a CTRL+WIN billentyűkombinációt amíg beszélsz, "
          "és engedd el amikor végzel.\n\n"
          "A Whisper AI automatikusan szöveggé alakítja amit mondtál. Ha az "
          "'Automatikus beillesztés' be van kapcsolva, Ctrl+V-vel beilleszti oda "
          "ahol éppen a kurzored áll (pl. Facebook, Word, e-mail).")

        T(self.vad_btn,
          "VAD MÓD – HANGAKTIVITÁS ALAPÚ AUTOMATIKUS FELVÉTEL\n\n"
          "Ha bekapcsolod, az alkalmazás folyamatosan figyeli a mikrofont "
          "a Silero VAD (Voice Activity Detection) neurális hálóval.\n\n"
          "• Amint megszólalsz → automatikusan elindul a felvétel\n"
          "• Amint kb. 0.9 másodpercig csend van → megáll és azonnal átírja\n"
          "• Ezután azonnal újra figyel a következő mondatra\n\n"
          "Ebben a módban nem kell semmilyen gombot nyomni.")

        T(self.time_label,
          "FELVÉTEL IDŐTARTAMA\n\n"
          "PTT módban mutatja az aktuális felvétel hosszát másodpercekben. "
          "Whisper large optimálisan ~30 másodperces szegmenseket kezel. "
          "Hosszabb felvételeknél a felismerés pontossága csökkenhet.")

        T(self.hotkey_label,
          "BILLENTYŰPARANCS (HOTKEY)\n\n"
          "CTRL + WIN lenyomva tartva = felvétel indítása PTT módban\n"
          "Felengedésre = felvétel leállítása és azonnali átírás\n\n"
          "Ez a kombináció akkor is működik, ha az alkalmazás ablaka nincs előtérben – "
          "tehát közvetlenül abba az ablakba diktálhatsz, ahol a kurzored áll.")

        T(self.vu_meter,
          "VU-MÉTER – ÉLŐ MIKROFON SZINTJELZŐ\n\n"
          "Valós időben mutatja a mikrofon bemeneti jelszintjét.\n\n"
          "• Ha felvétel közben mozog → a mikrofon rendesen veszi a hangot\n"
          "• Ha végig nulla marad → valószínűleg rossz hangeszköz van kiválasztva, "
          "a mikrofon némítva van, vagy nem csatlakozik fizikailag")

        T(self.ai_text,
          "WHISPER AI KIMENET – AMIT A GÉP HALLOTT\n\n"
          "Ide kerül a faster-whisper modell által felismert szöveg. "
          "Ez a mező nem szerkeszthető.\n\n"
          "Ha a szöveg helyes → nem kell semmit csinálni, már be is lett illesztve.\n"
          "Ha a szöveg hibás → az alatta lévő javítás mezőben módosítsd, "
          "majd nyomd meg a Tanulás gombot.")

        T(self.user_text,
          "JAVÍTÁS SZÖVEGDOBOZ – EBBŐL TANUL AZ AI\n\n"
          "Ide kerül automatikusan a Whisper kimenete, de szabadon szerkeszthető.\n\n"
          "Ha a gép valamit rosszul értett (pl. személynevet, helynevet, szakkifejezést), "
          "javítsd ki itt a helyes változatra, majd nyomd meg a Tanulás gombot.\n\n"
          "A modell a hang + helyes szöveg párból tanul LoRA finomhangolással.")

        T(self.auto_paste_cb,
          "AUTOMATIKUS BEILLESZTÉS A KURZORNÁL\n\n"
          "Ha be van pipálva:\n"
          "Az átírás után az app a vágólapra másolja a szöveget és Ctrl+V-vel beilleszti "
          "oda ahol éppen a kurzored áll. A vágólap EREDETI tartalma automatikusan "
          "visszaáll beillesztés után (~250 ms késéssel), így nem vész el semmi.\n\n"
          "Ha nincs bepipálva:\n"
          "A szöveg csak az alkalmazás szövegdobozában jelenik meg.")

        T(self.train_btn,
          "TANULÁS ÉS BIZTONSÁGI MENTÉS\n\n"
          "Megnyomva az app összehasonlítja a Whisper eredeti tippjét a te javításoddal, "
          "majd LoRA finomhangolással betanítja a modellt.\n\n"
          "A folyamat lépései:\n"
          "1. LoRA tanítás (5 lépés az adott hangra)\n"
          "2. Zip biztonsági mentés\n"
          "3. LoRA súlyok beolvasztása + CTranslate2 konverzió\n"
          "4. faster-whisper újratöltése az egyedi modellel")

        T(self.wave_path_entry,
          "HOSSZÚ HANGANYAG (.WAV) FÁJL\n\n"
          "Egy hosszabb, tisztán rögzített WAV fájl elérési útja a kötegelt tanításhoz. "
          "Lehet felolvasott szöveg, hangos feljegyzés, előadás részlet.\n\n"
          "Követelmények: WAV formátum, 16kHz+, tiszta felvétel.")

        T(self.txt_path_entry,
          "KÉZIRAT SZÖVEGFÁJL (.TXT)\n\n"
          "A hanganyaghoz tartozó pontos szöveges kézirat.\n\n"
          "Fontos: UTF-8 kódolás, a szöveg tartalmában és sorrendjében pontosan "
          "egyezzen a hanggal. Az app mondatonként párosítja a hangszeleteket.")

        T(self.factory_log,
          "FELDOLGOZÁSI NAPLÓ\n\n"
          "Lépésről lépésre követheted a kötegelt tanítás folyamatát:\n\n"
          "🟡 Sárga = figyelmeztetés (gyenge egyezés, leállítás kérve)\n"
          "🔴 Piros = hiba\n"
          "🟢 Zöld = sikeres lépés\n"
          "🔵 Kék = folyamatban lévő művelet")

        T(self.factory_progress,
          "HALADÁSI SÁV\n\n"
          "Megmutatja hány mondatot dolgozott fel az app az összes feldolgozandóból.")

        T(self.stop_factory_btn,
          "KÖTEGELT TANÍTÁS LEÁLLÍTÁSA\n\n"
          "Biztonságos leállítás – az app befejezi az éppen futó tanítási lépést, "
          "elmenti az eddig elkészült modellt, zip mentést készít, majd megáll.\n\n"
          "Semmi nem vész el az eddig feldolgozottakból.")

        T(self.start_factory_btn,
          "AUTOMATA KÖTEGELT TANÍTÁS INDÍTÁSA\n\n"
          "Egy hosszú hangfájlt és kéziratot dolgoz fel automatikusan.\n\n"
          "A folyamat: mondatonkénti darabolás → Whisper átírás → egyezés mérése → "
          "ha ≥40%: LoRA tanítás. Ha <40%: kihagyás (szinkronhiba ellen).")

    def _make_overlay_noactivate(self):
        """
        Windows-fix: a HUD maradjon felül, de SOHA ne vegye el a fókuszt
        attól az alkalmazástól, amelyikbe diktálunk.
        """
        if os.name != "nt":
            return
        try:
            import ctypes
            user32 = ctypes.windll.user32
            hwnd = int(self.overlay.winfo_id())

            GWL_EXSTYLE      = -20
            WS_EX_TOOLWINDOW = 0x00000080
            WS_EX_NOACTIVATE = 0x08000000

            # 64 bites Python alatt SetWindowLongPtrW a biztosabb.
            get_long = getattr(user32, "GetWindowLongPtrW", user32.GetWindowLongW)
            set_long = getattr(user32, "SetWindowLongPtrW", user32.SetWindowLongW)
            ex_style = int(get_long(hwnd, GWL_EXSTYLE))
            set_long(hwnd, GWL_EXSTYLE, ex_style | WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE)
            self._set_overlay_topmost_noactivate()
            print("[FOCUS] HUD no-activate mód bekapcsolva", flush=True)
        except Exception as e:
            print(f"[FOCUS] HUD no-activate hiba: {e}", flush=True)

    def _set_overlay_topmost_noactivate(self):
        """Topmost állítás fókuszlopás nélkül. Fontos: nincs overlay.lift()."""
        try:
            if not self.overlay.winfo_exists():
                return
            if os.name == "nt":
                import ctypes
                user32 = ctypes.windll.user32
                hwnd = int(self.overlay.winfo_id())
                HWND_TOPMOST   = -1
                SWP_NOSIZE     = 0x0001
                SWP_NOMOVE     = 0x0002
                SWP_NOACTIVATE = 0x0010
                SWP_SHOWWINDOW = 0x0040
                user32.SetWindowPos(
                    hwnd, HWND_TOPMOST, 0, 0, 0, 0,
                    SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE | SWP_SHOWWINDOW
                )
            else:
                self.overlay.attributes("-topmost", True)
        except Exception:
            try:
                self.overlay.attributes("-topmost", True)
            except Exception:
                pass

    def _drag_overlay(self, event):
        dx, dy = getattr(self, '_drag_xy', (0,0))
        self.overlay.geometry(f"+{self.overlay.winfo_x()-dx+event.x}+{self.overlay.winfo_y()-dy+event.y}")

    def _keep_overlay_top(self):
        try:
            if self.overlay.winfo_exists():
                # FONTOS: lift() nincs, mert fókuszt lophat a célalkalmazástól.
                self._set_overlay_topmost_noactivate()
        except Exception:
            pass
        self.root.after(1500, self._keep_overlay_top)

    # ── Tallózók ─────────────────────────────────────────────────────────────

    def browse_wave(self):
        f = filedialog.askopenfilename(filetypes=[("WAV", "*.wav")])
        if f:
            self.wave_path_entry.delete(0, tk.END)
            self.wave_path_entry.insert(0, f)

    def browse_txt(self):
        f = filedialog.askopenfilename(filetypes=[("Szöveg", "*.txt")])
        if f:
            self.txt_path_entry.delete(0, tk.END)
            self.txt_path_entry.insert(0, f)

    # ── Log / státusz ─────────────────────────────────────────────────────────

    def log_to_factory(self, text, color=None):
        self.root.after(0, lambda: self._safe_log(text, color))

    def _safe_log(self, text, color):
        tag = color if color in ("warning","error","success","info") else None
        if tag:
            self.factory_log.insert(tk.END, text+"\n", tag)
        else:
            self.factory_log.insert(tk.END, text+"\n")
        self.factory_log.see(tk.END)

    def log_status(self, text):
        self.root.after(0, lambda: self.status_bar.config(text=text))

    def _remember_paste_target_window(self):
        """
        Megjegyzi azt a külső Windows-ablakot, amelyik a diktálás indításakor aktív volt.
        Ez kell ahhoz, hogy a felismerés után a Ctrl+V ne a Voicetex saját ablakába menjen.
        """
        if os.name != "nt":
            return
        try:
            import ctypes
            user32 = ctypes.windll.user32
            hwnd = int(user32.GetForegroundWindow())
            if not hwnd:
                return

            # Ne saját Tk ablakot jegyezzünk meg célként.
            sajat_hwndk = set()
            for w in (getattr(self, "root", None), getattr(self, "overlay", None)):
                try:
                    sajat_hwndk.add(int(w.winfo_id()))
                except Exception:
                    pass
            if hwnd not in sajat_hwndk:
                self._paste_target_hwnd = hwnd
                print(f"[PASTE] Célablak megjegyezve: hwnd={hwnd}", flush=True)
        except Exception as e:
            print(f"[PASTE] Célablak mentési hiba: {e}", flush=True)

    def _restore_paste_target_window(self):
        """
        Visszateszi fókuszba a diktálás előtt aktív külső ablakot.
        Windows néha tiltja a SetForegroundWindow hívást, ezért AttachThreadInput is van benne.
        """
        if os.name != "nt":
            return False
        hwnd = getattr(self, "_paste_target_hwnd", None)
        if not hwnd:
            return False
        try:
            import ctypes
            from ctypes import wintypes
            user32 = ctypes.windll.user32
            kernel32 = ctypes.windll.kernel32

            user32.IsWindow.argtypes = [wintypes.HWND]
            user32.IsWindow.restype = wintypes.BOOL
            user32.IsIconic.argtypes = [wintypes.HWND]
            user32.IsIconic.restype = wintypes.BOOL
            user32.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
            user32.SetForegroundWindow.argtypes = [wintypes.HWND]
            user32.GetForegroundWindow.restype = wintypes.HWND
            user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
            user32.GetWindowThreadProcessId.restype = wintypes.DWORD
            user32.AttachThreadInput.argtypes = [wintypes.DWORD, wintypes.DWORD, wintypes.BOOL]

            hwnd = wintypes.HWND(hwnd)
            if not user32.IsWindow(hwnd):
                return False

            SW_RESTORE = 9
            if user32.IsIconic(hwnd):
                user32.ShowWindow(hwnd, SW_RESTORE)

            current_thread = wintypes.DWORD(kernel32.GetCurrentThreadId())
            target_pid = wintypes.DWORD()
            target_thread = wintypes.DWORD(user32.GetWindowThreadProcessId(hwnd, ctypes.byref(target_pid)))

            fg = user32.GetForegroundWindow()
            fg_pid = wintypes.DWORD()
            fg_thread = wintypes.DWORD(user32.GetWindowThreadProcessId(fg, ctypes.byref(fg_pid))) if fg else wintypes.DWORD(0)

            attached = []
            for tid in {int(target_thread.value), int(fg_thread.value)}:
                if tid and tid != int(current_thread.value):
                    if user32.AttachThreadInput(current_thread, wintypes.DWORD(tid), True):
                        attached.append(tid)
            try:
                # FONTOS: Itt NEM hívunk ShowWindow(hwnd, SW_RESTORE)-t.
                # Ha a böngésző/Word/egyéb célablak maximálva volt, az SW_RESTORE
                # normál méretre zsugorítaná, és sok webes textbox ilyenkor elveszíti
                # a belső kurzorfókuszt. Csak akkor restaurálunk, ha tényleg minimalizált.
                ok = bool(user32.SetForegroundWindow(hwnd))
            finally:
                for tid in attached:
                    user32.AttachThreadInput(current_thread, wintypes.DWORD(tid), False)

            time.sleep(0.05)
            aktiv = int(user32.GetForegroundWindow())
            print(f"[PASTE] Fókusz visszaállítás / méretváltoztatás nélkül: ok={ok}, aktiv={aktiv}", flush=True)
            return aktiv == int(hwnd.value)
        except Exception as e:
            print(f"[PASTE] Fókusz-visszaállítási hiba: {e}", flush=True)
            return False

    # ── Whisper modellválasztó ──────────────────────────────────────────────────

    def _valid_fw_model_ids(self):
        return [azon for _nev, azon in FW_MODEL_CHOICES]

    def _model_display_from_id(self, model_id):
        for nev, azon in FW_MODEL_CHOICES:
            if azon == model_id:
                return nev
        return self._model_display_from_id(DEFAULT_FW_MODEL_NAME)

    def _model_id_from_display(self, display_text):
        for nev, azon in FW_MODEL_CHOICES:
            if nev == display_text:
                return azon
        return DEFAULT_FW_MODEL_NAME

    def _load_selected_fw_model(self):
        """Utoljára használt faster-whisper modell betöltése JSON beállításból."""
        try:
            if os.path.exists(MODEL_SETTINGS_FILE):
                with open(MODEL_SETTINGS_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                model_id = data.get("fw_model_name", DEFAULT_FW_MODEL_NAME)
                # JAVÍTÁS: ha az egyedi CT2 modell volt elmentve, de a mappája
                # időközben eltűnt, gyári alapértelmezettre esünk vissza.
                if model_id == CUSTOM_CT2_ID and not os.path.exists(
                        os.path.join(CT2_MODEL_DIR, "model.bin")):
                    return DEFAULT_FW_MODEL_NAME
                if model_id in self._valid_fw_model_ids():
                    return model_id
        except Exception as e:
            print(f"[MODELL] Beállítás betöltési hiba: {e}", flush=True)
        return DEFAULT_FW_MODEL_NAME

    def _save_selected_fw_model(self, model_id):
        """Kiválasztott faster-whisper modell mentése JSON beállításba."""
        try:
            with open(MODEL_SETTINGS_FILE, "w", encoding="utf-8") as f:
                json.dump({"fw_model_name": model_id}, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[MODELL] Beállítás mentési hiba: {e}", flush=True)

    def _set_model_controls_state(self, state):
        """Modellválasztó vezérlők biztonságos engedélyezése/tiltása."""
        try:
            if hasattr(self, "model_combo"):
                self.model_combo.config(state=state)
            if hasattr(self, "reload_model_btn"):
                self.reload_model_btn.config(state="normal" if state == "readonly" else "disabled")
        except Exception:
            pass

    def reload_model_from_combo(self):
        """UI-gombról indított modellcsere háttérszálon."""
        if self.model_loading:
            return
        if self.is_recording:
            messagebox.showwarning("Modellcsere", "Felvétel közben nem lehet modellt váltani.")
            return
        if self.vad_aktiv:
            self._stop_vad()

        display = self.model_combo.get().strip() if hasattr(self, "model_combo") else ""
        uj_model = self._model_id_from_display(display)
        regi_model = getattr(self, "fw_model_name", DEFAULT_FW_MODEL_NAME)
        if uj_model == regi_model and getattr(self, "fw_model", None) is not None:
            self.log_status(f"✅ Ez a modell már aktív: {uj_model}")
            return

        self.model_loading = True
        self.record_btn.config(state="disabled")
        self.vad_btn.config(state="disabled")
        self.start_factory_btn.config(state="disabled")
        self._set_model_controls_state("disabled")
        self.log_status(f"🔄 Modellcsere indul: {regi_model} → {uj_model}")
        threading.Thread(
            target=self._reload_model_thread,
            args=(uj_model, regi_model),
            daemon=True
        ).start()

    def _reload_model_thread(self, uj_model, regi_model):
        try:
            self.load_active_model(uj_model)
            self.root.after(0, lambda: self._finish_model_reload(True, uj_model, regi_model, None))
        except Exception as e:
            print(f"[MODELL] Modellcsere hiba: {e}", flush=True)
            self.root.after(0, lambda m=str(e): self._finish_model_reload(False, uj_model, regi_model, m))

    def _finish_model_reload(self, siker, uj_model, regi_model, hiba):
        self.model_loading = False
        self.model_combo.set(self._model_display_from_id(getattr(self, "fw_model_name", regi_model)))
        self.record_btn.config(state="normal")
        self.start_factory_btn.config(state="normal")
        if self.vad_model is not None:
            self.vad_btn.config(state="normal")
        else:
            self.vad_btn.config(state="disabled")
        self._set_model_controls_state("readonly")
        if siker:
            self.log_status(f"✅ Aktív Whisper modell: {self.fw_model_name}")
            print(f"[MODELL] Aktív modell: {self.fw_model_name}", flush=True)
        else:
            self.log_status(f"❌ Modellcsere sikertelen: {hiba[:80] if hiba else ''}")
            messagebox.showerror(
                "Modellcsere hiba",
                "Nem sikerült betölteni a kiválasztott modellt.\n\n"
                f"Kért modell: {uj_model}\n"
                f"Hiba: {hiba[:500] if hiba else 'ismeretlen'}"
            )

    # ── Modell inicializálás ──────────────────────────────────────────────────

    def init_model(self):
        """Háttérszálon: faster-whisper + opcionálisan Silero VAD betöltése."""
        try:
            self.device      = "cuda" if torch.cuda.is_available() else "cpu"
            self.compute_type = "float16" if torch.cuda.is_available() else "int8"
            self.log_status("⚙️ faster-whisper modell betöltése...")
            self.load_active_model()

            if SILERO_VAD_ELERHETO:
                self.log_status("🎙️ Silero VAD betöltése...")
                self.vad_model = load_silero_vad()
            else:
                self.vad_model = None

            self.root.after(0, self.enable_ui_after_load)
        except Exception as e:
            self.root.after(0, lambda m=str(e): self._show_error(m))

    def _unload_fw_model(self):
        """JAVÍTÁS: a futó faster-whisper modell biztonságos, lock alatti
        kiürítése. Korábban `del self.fw_model` futott lock nélkül, ami
        versenyhelyzetet okozott az élő inferenciával."""
        with self.model_lock:
            self.fw_model = None
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def load_active_model(self, model_name=None):
        """faster-whisper modell betöltése a kiválasztott modellnév alapján."""
        if not FASTER_WHISPER_ELERHETO:
            raise RuntimeError(
                "A faster-whisper nincs telepítve!\n"
                "Telepítsd: pip install faster-whisper"
            )
        cel_model = model_name or getattr(self, "fw_model_name", DEFAULT_FW_MODEL_NAME)
        if cel_model not in self._valid_fw_model_ids():
            cel_model = DEFAULT_FW_MODEL_NAME

        # JAVÍTÁS (KRITIKUS): korábban a ct2_ok csak logba került, a WhisperModel
        # mindig a gyári modellnevet kapta – a betanított modell SOSEM töltődött be.
        ct2_ok = os.path.exists(os.path.join(CT2_MODEL_DIR, "model.bin"))
        print(f"[MODELL] CT2 mappa létezik: {ct2_ok}  →  {CT2_MODEL_DIR}", flush=True)
        if cel_model == CUSTOM_CT2_ID:
            if not ct2_ok:
                raise RuntimeError(
                    "Nincs betanított egyedi modell "
                    f"(hiányzik: {os.path.join(CT2_MODEL_DIR, 'model.bin')}). "
                    "Előbb taníts, vagy válassz gyári modellt."
                )
            betoltendo = CT2_MODEL_DIR
        else:
            betoltendo = cel_model
        print(f"[MODELL] {cel_model} töltése faster-whisper-rel...", flush=True)
        self.log_status(f"🌱 faster-whisper modell töltése: {cel_model}")

        with self.model_lock:
            regi_model = getattr(self, "fw_model_name", DEFAULT_FW_MODEL_NAME)
            # GPU memória felszabadítása modellcsere előtt.
            if getattr(self, "fw_model", None) is not None:
                try:
                    old = self.fw_model
                    self.fw_model = None
                    del old
                except Exception:
                    self.fw_model = None
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            try:
                uj_fw_model = WhisperModel(
                    betoltendo, device=self.device,
                    compute_type=self.compute_type
                )
            except Exception:
                # Ha a választott modell nem tölthető, megpróbáljuk visszahozni a régi stabil modellt.
                if regi_model != cel_model:
                    print(f"[MODELL] Visszaállási kísérlet erre: {regi_model}", flush=True)
                    try:
                        regi_betoltendo = (CT2_MODEL_DIR if regi_model == CUSTOM_CT2_ID
                                           else regi_model)
                        self.fw_model = WhisperModel(
                            regi_betoltendo, device=self.device,
                            compute_type=self.compute_type
                        )
                        self.fw_model_name = regi_model
                    except Exception as vissza_hiba:
                        print(f"[MODELL] Visszaállás sikertelen: {vissza_hiba}", flush=True)
                raise

            self.fw_model = uj_fw_model
            self.fw_model_name = cel_model
            self._save_selected_fw_model(cel_model)

        n = getattr(getattr(self.fw_model, "feature_extractor", None), "n_mels", "?")
        print(f"[MODELL] Betöltve: {self.fw_model_name} – n_mels={n}", flush=True)
        self.log_status(f"✅ Modell kész: {self.fw_model_name} – n_mels={n}")

    def enable_ui_after_load(self):
        self.record_btn.config(state="normal")
        self.start_factory_btn.config(state="normal")
        if self.vad_model is not None:
            self.vad_btn.config(state="normal")
        else:
            self.vad_btn.config(
                state="disabled",
                text="🤖 VAD (silero-vad nincs telepítve)"
            )
        self._refresh_vu()
        self._populate_devices()
        if hasattr(self, "model_combo"):
            self.model_combo.set(self._model_display_from_id(self.fw_model_name))
        self._set_model_controls_state("readonly")
        self.overlay.deiconify()
        self._make_overlay_noactivate()
        self._set_overlay_topmost_noactivate()
        if KEYBOARD_ELERHETO:
            try:
                keyboard.on_press_key("win",   self._hotkey_down)
                keyboard.on_release_key("win", self._hotkey_up)
            except Exception as e:
                print(f"Hotkey hiba: {e}")
        self.log_status(f"🚀 Voicetex v2 kész – aktív modell: {self.fw_model_name}")

    def _show_error(self, msg):
        self.status_bar.config(text="❌ Indítási hiba!")
        messagebox.showerror("Hiba", f"Nem sikerült elindítani a modellt.\n\n{msg}")

    # ── Eszköz lista ──────────────────────────────────────────────────────────

    def _populate_devices(self):
        if not MIKROFON_ELERHETO:
            self.device_combo.config(values=["Nincs hangeszköz!"])
            self.device_combo.current(0)
            return
        try:
            eszkozok = sd.query_devices()
            nevek, self.minden_eszkoz_lista = [], []
            pref, pref_score, cnt = 0, 0, 0
            for i, d in enumerate(eszkozok):
                if d['max_input_channels'] > 0:
                    nevek.append(f"{i}: {d['name']} ({int(d['max_input_channels'])} ch)")
                    self.minden_eszkoz_lista.append(i)
                    n = d['name'].lower()
                    # Prioritás: AudioBox/96 > USB mikrofon > egyéb
                    if "audiobox" in n or "96" in n:
                        score = 3
                    elif "usb" in n and "mikrofon" in n:
                        score = 2
                    elif "usb" in n:
                        score = 1
                    else:
                        score = 0
                    if score > pref_score:
                        pref_score = score
                        pref = cnt
                    cnt += 1
            if nevek:
                self.device_combo.config(values=nevek)
                self.device_combo.current(pref)
        except Exception as e:
            self.log_status(f"⚠️ Eszköz hiba: {e}")

    # ── VAD mód ───────────────────────────────────────────────────────────────

    def toggle_vad(self):
        if self.vad_aktiv:
            self._stop_vad()
        else:
            self._start_vad()

    def _start_vad(self):
        if self.vad_model is None:
            messagebox.showwarning("VAD", "A silero-vad nincs betöltve!")
            return
        self.vad_aktiv  = True
        self.vad_beszel = False
        self.record_btn.config(state="disabled")
        self._set_model_controls_state("disabled")
        self.vad_btn.config(text="🤖 VAD mód: BE  (katt a leállításhoz)", style="VADActive.TButton")
        self.log_status("🎙️ VAD mód aktív – automatikusan figyeli a hangot...")
        self.overlay_label.config(text="🤖 VAD FIGYELÉS", fg="#00ff66", bg="#1a1a1a")
        self.overlay.deiconify()
        # JAVÍTÁS: a mikrofon-eszközt a főszálon olvassuk ki (Tk widgethez
        # háttérszálból nyúlni nem szálbiztos), és paraméterként adjuk át.
        idx = self.device_combo.current()
        dev = self.minden_eszkoz_lista[idx] if 0 <= idx < len(self.minden_eszkoz_lista) else None
        threading.Thread(target=self._vad_thread, args=(dev,), daemon=True).start()

    def _stop_vad(self):
        self.vad_aktiv  = False
        self.vad_beszel = False
        self.record_btn.config(state="normal")
        self._set_model_controls_state("readonly")
        self.vad_btn.config(text="🤖 VAD mód: KI", style="VAD.TButton")
        self.log_status("🎙️ VAD mód leállítva.")
        self._overlay_standby()
        # Ha VAD közben elhalasztott auto-tanulás gyűlt össze, csak a VAD leállítása után indítjuk.
        if getattr(self, "auto_tanulas_varakozik", False) and self.auto_tanulas_var.get():
            self.root.after(1200, self._auto_tanitas_indit)

    def _vad_thread(self, dev):
        """
        Folyamatos VAD figyelés háttérszálon.
        512 mintás (32ms) chunkokat olvas, Silero VAD dönt a hangkezdetről/végéről.
        """
        try:
            vad_iter = VADIterator(
                self.vad_model,
                threshold=VAD_KUSZOB,
                sampling_rate=MINTAVETELI_FREKVENCIA,
                min_silence_duration_ms=VAD_CSEND_MS,
                speech_pad_ms=VAD_PADDING_MS
            )

            audio_buf = []
            CHUNK = 512  # silero: pontosan 512 minta kell 16kHz-en

            with sd.InputStream(
                samplerate=MINTAVETELI_FREKVENCIA,
                channels=1, dtype='float32',
                device=dev, blocksize=CHUNK
            ) as stream:
                while self.vad_aktiv:
                    chunk, _ = stream.read(CHUNK)
                    flat = chunk.flatten()

                    # VU frissítés
                    self.aktualis_vu_szint = float(np.sqrt(np.mean(flat**2)))

                    speech_dict = vad_iter(torch.from_numpy(flat), return_seconds=False)

                    if speech_dict:
                        if 'start' in speech_dict and not self.vad_beszel:
                            self._remember_paste_target_window()
                            self.vad_beszel = True
                            audio_buf = [flat.copy()]
                            self.root.after(0, self._vad_ui_recording)

                        elif 'end' in speech_dict and self.vad_beszel:
                            audio_buf.append(flat.copy())
                            self.vad_beszel = False
                            self.root.after(0, self._vad_ui_processing)
                            # Felvett audio feldolgozása
                            self._vad_save_and_transcribe(audio_buf.copy())
                            audio_buf = []

                    elif self.vad_beszel:
                        audio_buf.append(flat.copy())

        except Exception as e:
            self.root.after(0, lambda m=str(e): self.log_status(f"❌ VAD hiba: {m}"))
        finally:
            self.vad_aktiv  = False
            self.vad_beszel = False
            self.root.after(0, self._vad_ui_standby_restore)

    def _vad_ui_recording(self):
        self.overlay.configure(bg="#440000")
        self.overlay_label.config(text="🔴 VAD – RÖGZÍT...", fg="#ffffff", bg="#440000")

    def _vad_ui_processing(self):
        self.overlay.configure(bg="#333300")
        self.overlay_label.config(text="⏳ VAD – FELDOLGOZ...", fg="#ffff00", bg="#333300")
        self.aktualis_vu_szint = 0.0

    def _vad_ui_standby_restore(self):
        self.record_btn.config(state="normal")
        self.vad_btn.config(text="🤖 VAD mód: KI", style="VAD.TButton")
        self._overlay_standby()

    def _vad_save_and_transcribe(self, buf):
        """VAD által összegyűjtött audio mentése és átírás."""
        try:
            os.makedirs(DATASET_DIR, exist_ok=True)
            fajlnev = os.path.join(DATASET_DIR, f"vad_{int(time.time())}.wav")
            audio = np.concatenate(buf, axis=0)
            wav.write(fajlnev, MINTAVETELI_FREKVENCIA, (audio * 32767).astype(np.int16))
            self.utolso_hang_utvonal = fajlnev
            # OPTIMALIZÁLÁS: a memóriában lévő hangot adjuk át, nem a fájlt.
            self.run_inference(audio_np=audio.astype(np.float32))
        except Exception as e:
            self.root.after(0, lambda m=str(e): self.log_status(f"❌ VAD mentési hiba: {m}"))

    # ── PTT felvétel ──────────────────────────────────────────────────────────

    def toggle_recording(self):
        if not self.is_recording:
            self.start_recording_process()
        else:
            self.stop_recording_process()

    def start_recording_process(self):
        if self.is_recording or self.vad_aktiv: return
        self._remember_paste_target_window()
        self.is_recording = True
        self.record_btn.config(text="🛑 Engedd el")
        self.device_combo.config(state="disabled")
        self._set_model_controls_state("disabled")
        self.overlay.configure(bg="#440000")
        self.overlay_label.config(text="🔴 PTT – RÖGZÍT...", fg="#ffffff", bg="#440000")
        self.overlay.deiconify()
        self._make_overlay_noactivate()
        self._set_overlay_topmost_noactivate()
        self.hang_adatok       = []
        self.record_start_time = time.time()
        self._update_timer()
        # JAVÍTÁS: eszközindex kiolvasása a főszálon (Tk szálbiztonság).
        idx = self.device_combo.current()
        threading.Thread(target=self._record_thread, args=(idx,), daemon=True).start()

    def stop_recording_process(self):
        if not self.is_recording: return
        self.is_recording = False
        self.record_btn.config(state="disabled", text="⏳ Whisper AI...")
        self.overlay.configure(bg="#333300")
        self.overlay_label.config(text="⏳ FELDOLGOZÁS...", fg="#ffff00", bg="#333300")

    def _record_thread(self, idx):
        fajlnev = os.path.join(DATASET_DIR, f"ptt_{int(time.time())}.wav")
        os.makedirs(DATASET_DIR, exist_ok=True)
        self.utolso_hang_utvonal = fajlnev
        try:
            if 0 <= idx < len(self.minden_eszkoz_lista):
                dev      = self.minden_eszkoz_lista[idx]
                max_ch   = int(sd.query_devices(dev)['max_input_channels'])
            else:
                dev, max_ch = None, 1

            def cb(indata, frames, t, status):
                if self.is_recording:
                    self.hang_sor.put(indata.copy())
                    self.aktualis_vu_szint = float(np.sqrt(np.mean(indata**2)))

            if MIKROFON_ELERHETO:
                # Ha a kiválasztott eszköz érvénytelen, fallback az alapértelmezettre
                for try_dev, try_ch in ([(dev, max_ch)] if dev is not None else []) + [(None, 1)]:
                    try:
                        with sd.InputStream(samplerate=MINTAVETELI_FREKVENCIA,
                                            channels=try_ch, device=try_dev, callback=cb):
                            print(f"[MIK] Felvétel – eszköz: {try_dev}", flush=True)
                            self.root.after(0, lambda d=try_dev: self.log_status(
                                f"🎙️ Eszköz: {'#'+str(d) if d is not None else 'alapértelmezett'}"))
                            while self.is_recording or not self.hang_sor.empty():
                                try:
                                    self.hang_adatok.append(self.hang_sor.get(timeout=0.05))
                                except queue.Empty:
                                    pass
                        break  # sikeres
                    except Exception as dev_err:
                        print(f"[MIK] Eszköz {try_dev} nem elérhető: {dev_err}", flush=True)
                        if try_dev is None:
                            raise RuntimeError(f"Egyik mikrofon sem elérhető: {dev_err}")
            else:
                while self.is_recording:
                    time.sleep(0.1)
                self.hang_adatok.append(np.zeros((MINTAVETELI_FREKVENCIA*3,1), dtype=np.float32))

            if self.hang_adatok:
                total = np.concatenate(self.hang_adatok, axis=0)
                if total.ndim > 1 and total.shape[1] > 1:
                    mono = total[:, np.argmax(
                        [np.max(np.abs(total[:,c])) for c in range(total.shape[1])]
                    )]
                else:
                    mono = total.flatten()
                wav.write(fajlnev, MINTAVETELI_FREKVENCIA, (mono*32767).astype(np.int16))
                # OPTIMALIZÁLÁS: a memóriában lévő hangot adjuk át, nem a fájlt.
                self.run_inference(audio_np=mono.astype(np.float32))
            else:
                self.root.after(0, self._reset_ptt_btn)
        except Exception as rec_err:
            import traceback
            traceback.print_exc()
            self.root.after(0, lambda m=str(rec_err): self.log_status(f"❌ Felvételi hiba: {m}"))
            self.root.after(0, self._reset_ptt_btn)

    # ── Whisper inferencia (faster-whisper) ───────────────────────────────────

    def run_inference(self, audio_np=None):
        """Hangfájl átírása faster-whisper-rel, majd megjelenítés.

        OPTIMALIZÁLÁS: ha a hívó már memóriában tartja a hangot (PTT/VAD),
        azt kapja meg a Whisper közvetlenül – korábban a frissen kiírt wav
        fájlt olvastuk vissza, majd a faster-whisper PyAV-val MÉG EGYSZER
        dekódolta ugyanazt. Ez minden diktálásnál felesleges lemez- és
        dekódolási kört jelentett."""
        try:
            print(f"[INFO] Átírás indul: {self.utolso_hang_utvonal}", flush=True)
            if audio_np is None:
                sr, audio_np = wav.read(self.utolso_hang_utvonal)
                audio_np = audio_np.astype(np.float32) / 32767.0
            rms = float(np.sqrt(np.mean(audio_np**2)))
            print(f"[INFO] RMS={rms:.5f}", flush=True)

            if rms < 0.0015:
                print("[INFO] Túl halk – süket csend", flush=True)
                self.root.after(0, lambda: self.display_result("[Süket csend – hangosabban?]"))
                return

            with self.model_lock:
                fw = getattr(self, "fw_model", None)
                if fw is None:
                    raise RuntimeError("A Whisper modell éppen nincs betöltve vagy modellcsere/tanítás alatt áll. Próbáld újra pár másodperc múlva.")
                segments_gen, _ = fw.transcribe(
                    audio_np,
                    language="hu",
                    beam_size=DIKTALAS_BEAM_SIZE,
                    vad_filter=False,
                    # OPTIMALIZÁLÁS: rövid diktálásnál az előző szövegre
                    # kondicionálás lassít és ismétlés-hallucinációt okozhat.
                    condition_on_previous_text=False,
                )
                segments_list = list(segments_gen)
            text_result = " ".join(s.text.strip() for s in segments_list).strip()
            print(f"[INFO] Whisper eredmény: '{text_result}'", flush=True)

            # Bizalom számítás (avg_logprob: 0=tökéletes, -1=bizonytalan)
            if segments_list:
                avg_conf = sum(s.avg_logprob for s in segments_list) / len(segments_list)
            else:
                avg_conf = -1.0
            print(f"[INFO] Bizalom: {avg_conf:.3f}  (küszöb: {AUTO_TANULAS_KUSZOB})", flush=True)

            # Hallucináció szűrő (JAVÍTÁS: csak kifejezésekre, nem egyes szavakra)
            if hallucinacio_gyanus(text_result):
                text_result = ""

            # Auto-tanulás: ha elég biztos volt és engedélyezett
            # (a queue a NYERS szöveget kapja, a hangparancs-átalakítás ELŐTT,
            # hogy a tanítópár azt tükrözze, ami ténylegesen elhangzott)
            if (text_result.strip()
                    and avg_conf >= AUTO_TANULAS_KUSZOB
                    and self.auto_tanulas_var.get()):
                self._auto_queue_mentes(self.utolso_hang_utvonal, text_result, avg_conf)

            # ÚJ: hangparancsok átalakítása (pont, vessző, új bekezdés, ...)
            text_result = hangparancs_atalakitas(text_result)

            def _paste(szoveg=text_result):
                # ÚJ: a csak sortörésből álló eredmény (kimondott "új bekezdés" /
                # "új sor" külön mondatként) is érvényes, beillesztendő tartalom.
                van_tartalom = bool(szoveg.strip()) or ("\n" in szoveg)
                if szoveg.strip():
                    paste_text_for_ui = szoveg
                elif "\n" in szoveg:
                    paste_text_for_ui = "[új bekezdés]" if "\n\n" in szoveg else "[új sor]"
                else:
                    paste_text_for_ui = "[Nem értett semmit]"
                autopaste_kell = self.auto_paste_var.get() and KEYBOARD_ELERHETO and van_tartalom

                if autopaste_kell:
                    # Vágólap eredeti tartalmának mentése.
                    # Megjegyzés: a visszaállítást késleltetjük, mert Chrome/Word/Electron
                    # alkalmazások néha nem azonnal olvassák ki a vágólapot a Ctrl+V után.
                    try:
                        regi = self.root.clipboard_get()
                    except Exception:
                        regi = None

                    # Beillesztendő szöveg a vágólapra.
                    # Csak update_idletasks(), mert a teljes root.update() fókuszt lophat.
                    self.root.clipboard_clear()
                    self.root.clipboard_append(szoveg)
                    try:
                        self.root.update_idletasks()
                    except Exception:
                        pass

                    # FONTOS REFAKTOR-FIX:
                    # Korábban a saját Voicetex szövegdobozait azonnal frissítettük,
                    # miközben a háttérszál még nem küldte el a Ctrl+V-t. Ez versenyhelyzetet
                    # okozott: a Tk UI-frissítés néha visszavette a fókuszt, ezért a paste
                    # egyszer átment, egyszer nem. Most a saját UI csak a Ctrl+V elküldése után frissül.
                    def _do(r=regi, ui_text=paste_text_for_ui):
                        try:
                            self.root.after(0, self.overlay.withdraw)
                        except Exception:
                            pass

                        # Hagyunk időt a célprogramnak / Windowsnak, aztán visszaadjuk a fókuszt.
                        time.sleep(0.30)
                        self._restore_paste_target_window()
                        time.sleep(0.10)

                        try:
                            # A PTT hotkey miatt biztosra megyünk: a Ctrl/Win ne maradjon
                            # logikailag lenyomott állapotban, amikor elküldjük a Ctrl+V-t.
                            for _k in ("ctrl", "left ctrl", "right ctrl", "windows", "left windows", "right windows"):
                                try:
                                    keyboard.release(_k)
                                except Exception:
                                    pass
                            time.sleep(0.05)
                            aktiv_elotte = None
                            try:
                                if os.name == "nt":
                                    import ctypes
                                    aktiv_elotte = int(ctypes.windll.user32.GetForegroundWindow())
                            except Exception:
                                pass
                            keyboard.send("ctrl+v")
                            print(f"[PASTE] Ctrl+V elküldve; aktiv_elotte={aktiv_elotte}", flush=True)
                        except Exception as e:
                            print(f"[PASTE] Ctrl+V hiba: {e}", flush=True)
                            self.root.after(0, lambda s=ui_text: self.display_result(s))
                            return

                        # A Ctrl+V után visszatesszük a fókuszt, majd csak EZUTÁN frissítjük
                        # a saját Voicetex ablakát. Így a saját Text widgetek nem előzik meg a paste-et.
                        time.sleep(0.15)
                        self._restore_paste_target_window()
                        self.root.after(0, lambda s=ui_text: self.display_result(s))

                        # Konzervatív vágólap-visszaállítás.
                        # 3 mp szándékosan hosszú: webes textboxoknál stabilabb, mint az 1.2 mp.
                        time.sleep(3.00)
                        def _restore(rv=r, mienk=szoveg):
                            try:
                                # JAVÍTÁS: csak akkor állítjuk vissza a régi vágólapot,
                                # ha még a mi beillesztett szövegünk van rajta. Ha a
                                # felhasználó közben mást másolt, nem írjuk felül.
                                try:
                                    aktualis = self.root.clipboard_get()
                                except Exception:
                                    aktualis = None
                                if aktualis is not None and aktualis != mienk:
                                    return
                                self.root.clipboard_clear()
                                if rv is not None:
                                    self.root.clipboard_append(rv)
                                self.root.update_idletasks()
                            except Exception:
                                pass
                            threading.Thread(
                                target=lambda: (time.sleep(0.05), self._restore_paste_target_window()),
                                daemon=True
                            ).start()
                        self.root.after(0, _restore)

                    threading.Thread(target=_do, daemon=True).start()
                else:
                    self.display_result(paste_text_for_ui)

            self.root.after(0, _paste)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.root.after(0, lambda m=str(e): self._inference_error(m))

    def display_result(self, text):
        self.ai_text.config(state="normal")
        self.ai_text.delete("1.0", tk.END)
        self.ai_text.insert("1.0", text)
        self.ai_text.config(state="disabled")
        self.user_text.delete("1.0", tk.END)
        self.user_text.insert("1.0", text)
        self._reset_ptt_btn()
        self.train_btn.config(state="normal")
        self.log_status("📋 Szöveg vágólapra téve, beillesztés elküldve a célablaknak!")
        # A saját UI frissítése után ne a Voicetex maradjon aktív, hanem a diktálás célablaka.
        if getattr(self, "_paste_target_hwnd", None):
            self.root.after(80, self._restore_paste_target_window)
        if hasattr(self, "felirat_diktalt_label"):
            rovid = text if len(text) <= 80 else text[:77] + "..."
            self.felirat_diktalt_label.config(text=rovid)

    def _inference_error(self, msg):
        print(f"[HIBA] _inference_error: {msg}", flush=True)
        self.log_status(f"❌ Hiba: {msg[:80]}")
        messagebox.showerror("Átírási hiba", f"A Whisper nem tudott átírni:\n\n{msg[:300]}")
        self._reset_ptt_btn()

    def _reset_ptt_btn(self):
        self.is_recording      = False
        self.aktualis_vu_szint = 0.0
        self.record_btn.config(state="normal", text="🎤 PTT – Tartsd nyomva")
        self.device_combo.config(state="readonly")
        self._set_model_controls_state("readonly")
        self._overlay_standby()

    def _overlay_standby(self):
        self.overlay.configure(bg="#1a1a1a")
        txt = "🤖 VAD FIGYELÉS" if self.vad_aktiv else "🎙️ PTT STANDBY"
        clr = "#00ff66"       if self.vad_aktiv else "#888888"
        self.overlay_label.config(text=txt, fg=clr, bg="#1a1a1a")

    # ── Timer / VU ────────────────────────────────────────────────────────────

    def _update_timer(self):
        if self.is_recording:
            self.time_label.config(text=f"⏱ {time.time()-self.record_start_time:.1f} mp")
            self.root.after(100, self._update_timer)

    def _refresh_vu(self):
        # OPTIMALIZÁLÁS: ez a ciklus 30ms-onként örökké fut; ha a szint nem
        # változott (pl. tétlen app), nem piszkáljuk feleslegesen a widgeteket.
        val = min(int(self.aktualis_vu_szint * 350), 100)
        if val != getattr(self, "_utolso_vu_val", -1):
            self._utolso_vu_val = val
            self.vu_meter.config(value=val)
            self.overlay_vu.config(value=val)
        self.root.after(30, self._refresh_vu)

    # ── Hotkey ────────────────────────────────────────────────────────────────

    def _hotkey_down(self, event):
        if keyboard.is_pressed("ctrl") and not self.is_recording and not self.vad_aktiv:
            self._remember_paste_target_window()
            self.root.after(0, self.start_recording_process)

    def _hotkey_up(self, event):
        if self.is_recording:
            self.root.after(0, self.stop_recording_process)

    # ── Tanítás ───────────────────────────────────────────────────────────────

    def start_learning(self):
        javitott = self.user_text.get("1.0", tk.END).strip()
        if not javitott:
            messagebox.showwarning("Figyelem", "Üres szövegből nem tudok tanulni!")
            return
        if not self.utolso_hang_utvonal or not os.path.exists(self.utolso_hang_utvonal):
            messagebox.showwarning("Nincs hangfájl",
                "Még nem rögzítettél hangot!\nElőbb vegyél fel valamit.")
            return
        self.train_btn.config(state="disabled")
        self.record_btn.config(state="disabled")
        self.log_status("🧠 LoRA finomhangolás indul...")
        threading.Thread(target=self._train_wrapper, args=(javitott,), daemon=True).start()

    def _train_wrapper(self, javitott):
        try:
            # 1. Tanítás
            # JAVÍTÁS: lock alatti modell-ürítés a `del self.fw_model` helyett.
            self._unload_fw_model()
            hatter_tanitas_process(self.utolso_hang_utvonal, javitott, self.device)

            # 2. Zip backup
            self.log_status("💾 LoRA zip backup...")
            lora_automatikus_mentes()

            # 3. LoRA → CT2 konverzió
            self.log_status("⚙️ faster-whisper konverzió folyamatban...")
            konv_ok = lora_merge_es_ct2_konvertalas(self.log_status)

            # 4. faster-whisper újratöltés
            # JAVÍTÁS: sikeres konverzió után az egyedi CT2 modellt töltjük be,
            # különben a tanításnak nem lenne látható hatása.
            self.load_active_model(CUSTOM_CT2_ID if konv_ok else None)
            self.log_status("✅ Tanulás és konverzió kész – egyedi modell aktív!")
            gep_beszel_magyarul("Megtanultam és aktiváltam az egyedi faster-whisper modellt.")
            self.root.after(0, lambda: messagebox.showinfo(
                "Siker", "Tanulás kész!\nAz egyedi modell mostantól aktív."))
        except Exception as e:
            self.log_status(f"❌ Tanítási hiba: {e}")
            self.load_active_model()
        self.root.after(0, lambda: self.record_btn.config(state="normal"))

    # ── Kötegelt tanítás ──────────────────────────────────────────────────────

    def request_factory_stop(self):
        self.stop_factory_requested = True
        self.log_to_factory("🛑 Leállítás kérve...", "warning")
        self.stop_factory_btn.config(state="disabled")

    def start_batch_processing(self):
        if not DARABOLO_ELERHETO:
            messagebox.showerror("Hiányzó csomag",
                "Kötegelt feldolgozáshoz szükséges:\n  pip install librosa soundfile")
            return
        wave = self.wave_path_entry.get().strip()
        txt  = self.txt_path_entry.get().strip()
        if not wave or not txt:
            messagebox.showwarning("Figyelem", "Add meg mindkét fájlt!")
            return
        self.start_factory_btn.config(state="disabled")
        self.stop_factory_btn.config(state="normal")
        self.record_btn.config(state="disabled")
        self.factory_log.delete("1.0", tk.END)
        self.stop_factory_requested = False
        threading.Thread(target=self._batch_thread, args=(wave, txt), daemon=True).start()

    def _batch_thread(self, wave_path, txt_path):
        try:
            self.log_to_factory("📖 Kézirat beolvasása...")
            with open(txt_path, encoding="utf-8") as f:
                szoveg = f.read().replace("\n"," ").strip()
            mondatok = [m.strip() for m in re.split(r'(?<=[.?!])\s+', szoveg) if m.strip()]
            self.log_to_factory(f"✨ Mondatok: {len(mondatok)}")

            self.log_to_factory("🎵 Hanganyag szeletelése...")
            y, _ = librosa.load(wave_path, sr=MINTAVETELI_FREKVENCIA)
            intervals = librosa.effects.split(y, top_db=26, frame_length=2048, hop_length=512)
            self.log_to_factory(f"✂️ Szeletek: {len(intervals)}")

            n = min(len(mondatok), len(intervals))
            self.root.after(0, lambda: self.factory_progress.config(max=n, value=0))

            temp = "./temp_szeletek"
            os.makedirs(temp, exist_ok=True)

            # OPTIMALIZÁLÁS (KRITIKUS): korábban MINDEN elfogadott mondatnál
            # lefutott a teljes lánc: 3GB-os modell betöltése + LoRA tanítás
            # + zip backup + merge + CT2 konverzió + faster-whisper újratöltés.
            # 100 mondatnál ez órákat jelentett és a diktáló modellt is
            # folyamatosan ki-be rángatta. Most:
            #   1. fázis: darabolás + átírás + egyezésvizsgálat (a betöltött
            #      fw modellel, ami végig a memóriában marad),
            #   2. fázis: az ÖSSZES elfogadott páron EGY tanítás, EGY backup,
            #      EGY konverzió, EGY modell-újratöltés.
            tanito_parok = []   # [(szelet_ut, mondat), ...]

            for i in range(n):
                if self.stop_factory_requested:
                    self.log_to_factory("🛑 Leállítva.", "warning"); break

                mondat = mondatok[i]
                self.log_to_factory(f"\n[{i+1}/{n}] {mondat[:60]}...")

                s, e = intervals[i]
                szelet = y[s:e]

                if np.sqrt(np.mean(szelet**2)) < 0.005:
                    self.log_to_factory("🤫 Túl halk, kihagyva.", "warning")
                    self.root.after(0, lambda v=i+1: self.factory_progress.config(value=v))
                    continue

                szelet_ut = os.path.join(temp, f"sz_{i}.wav")
                sf.write(szelet_ut, szelet, MINTAVETELI_FREKVENCIA)

                # faster-whisper inferencia a szelethez
                # JAVÍTÁS: lock alatt, None-ellenőrzéssel (versenyhelyzet ellen).
                # OPTIMALIZÁLÁS: a hangot memóriából adjuk át (nem fájl-útvonalat),
                # így elmarad a felesleges újradekódolás.
                with self.model_lock:
                    fw = getattr(self, "fw_model", None)
                    if fw is None:
                        raise RuntimeError("A Whisper modell nincs betöltve.")
                    segs, _ = fw.transcribe(szelet.astype(np.float32),
                                            language="hu",
                                            beam_size=3, vad_filter=True)
                    tipp = " ".join(seg.text.strip() for seg in segs).strip()
                if hallucinacio_gyanus(tipp):
                    tipp = ""

                hasonlo = hasonlosag_arány(tipp, mondat)
                self.log_to_factory(f"🔍 Tipp: '{tipp}'")
                self.log_to_factory(f"📊 Egyezés: {hasonlo*100:.1f}%")

                if hasonlo < 0.40:
                    self.log_to_factory("⚠️ Gyenge egyezés, kihagyva.", "warning")
                    if os.path.exists(szelet_ut): os.remove(szelet_ut)
                    self.root.after(0, lambda v=i+1: self.factory_progress.config(value=v))
                    continue

                tanito_parok.append((szelet_ut, mondat))
                self.root.after(0, lambda v=i+1: self.factory_progress.config(value=v))
                self.log_to_factory("✅ Elfogadva a tanítókötegbe.", "success")

            # ── 2. fázis: egyetlen tanítás az összes elfogadott páron ────────
            if tanito_parok:
                self.log_to_factory(
                    f"\n🧠 Tanítás indul: {len(tanito_parok)} pár EGY menetben...", "info")
                self._unload_fw_model()
                hatter_tanitas_tobb_minta(tanito_parok, self.device,
                                          log_fn=self.log_to_factory)
                self.log_to_factory("💾 Zip backup...", "info")
                lora_automatikus_mentes()
                konv_ok = lora_merge_es_ct2_konvertalas(self.log_to_factory)
                self.load_active_model(CUSTOM_CT2_ID if konv_ok else None)
                for szelet_ut, _m in tanito_parok:
                    if os.path.exists(szelet_ut):
                        os.remove(szelet_ut)
            else:
                self.log_to_factory("\nℹ️ Nem volt elfogadható tanítópár.", "warning")

            if os.path.exists(temp) and not os.listdir(temp):
                os.rmdir(temp)

            self.log_to_factory("\n🏁 FOLYAMAT KÉSZ!", "success")
            gep_beszel_magyarul("A kötegelt tanítás sikeresen befejeződött.")
            self.root.after(0, lambda: messagebox.showinfo("Kész","Kötegelt tanítás lefutott!"))

        except Exception as e:
            self.log_to_factory(f"❌ Hiba: {e}", "error")
            self.load_active_model()
        finally:
            self.root.after(0, lambda: self.start_factory_btn.config(state="normal"))
            self.root.after(0, lambda: self.stop_factory_btn.config(state="disabled"))
            self.root.after(0, lambda: self.record_btn.config(state="normal"))


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    root = tk.Tk()
    VoicetexApp(root)
    root.mainloop()