# Voicetex v3 – javított kiadás

**Fájlok:**
- `voicetex_v3.py` – a javított alkalmazás (ezt futtasd)
- `archiv/voicetex_v2_space_model_combo_refactor3_manual_training.py` – az eredeti, érintetlen változat
- Minden módosítás a kódban `JAVÍTÁS` kommenttel van jelölve (18 helyen).

## Kritikus javítás

### 1. A betanított (LoRA→CT2) modell mostantól ténylegesen betöltődik
Az eredetiben a `load_active_model` ellenőrizte ugyan a CT2 mappa létezését, de a
`WhisperModel` mindig a gyári modellnevet kapta – a tanítás eredménye **sosem
aktivizálódott**. Mostantól:

- A modellválasztóban megjelent egy új elem: **„🧠 Egyedi tanított modell (LoRA→CT2)"**.
- Sikeres tanítás + konverzió után az app automatikusan erre vált.
- Bármikor visszaválthatsz gyári modellre a választóból.
- Ha az egyedi modell volt elmentve, de a CT2 mappa eltűnt, az app gyári
  alapértelmezettre (large-v2) esik vissza hibaüzenet helyett induláskor.

## További hibajavítások

### 2. `del self.fw_model` versenyhelyzet (2 helyen)
A kézi tanítás (`_train_wrapper`) és a kötegelt tanítás (`_batch_thread`) lock
nélkül törölte a futó modellt – élő diktálás mellett ez okozta a korábbi
`fw_model` eltűnéses hibákat. Új, lock alatti `_unload_fw_model()` metódus
végzi az ürítést.

### 3. IndexError a feliratfordítás leállításakor
Ha a fordítást menet közben leállítottad, a fordításlista rövidebb lett, mint a
felirat-itemek – a `_srt_forditas` összeomlott. Most a hiányzó fordítások
helyére az eredeti (forrásnyelvi) szöveg kerül.

### 4. Tanítási hangfeldolgozás: mintavételi frekvencia és formátum
A `hatter_tanitas_process` eldobta a wav tényleges mintavételi frekvenciáját és
mindig int16-ot feltételezett. Most: sztereó→mono konverzió, dtype-helyes
normalizálás, és szükség esetén újramintavételezés 16 kHz-re.

### 5. Túl agresszív hallucináció-szűrő
Korábban ha a diktált szövegben szerepelt a „felirat" vagy „közösség" szó, a
teljes eredmény törlődött. Most csak jellegzetes Whisper-hallucinációs
kifejezésekre szűr (pl. „amara.org", „subtitles by", „feliratok a közösségtől").

### 6. Vágólap-visszaállítás
A 3 mp-es késleltetett visszaállítás felülírta, amit közben másoltál. Most csak
akkor állítja vissza a régi vágólaptartalmat, ha még a beillesztett szöveg van
rajta.

### 7. Tkinter szálbiztonság
Háttérszálak többé nem nyúlnak közvetlenül Tk widgetekhez/változókhoz:
- mikrofon-eszközindex kiolvasása a főszálon (PTT és VAD indításkor), paraméterként átadva,
- a feliratmodul beállításai (nyelv, fordítás, párbeszéd-küszöb) induláskor
  pillanatképbe kerülnek, a szál csak azt olvassa.

### 8. Hiányzó modell elleni védelem
A kötegelt tanítás és a felirat-átírás inferenciája is lock alatt,
None-ellenőrzéssel fut (mint a diktálásé).

### 9. Holt kód eltávolítva
A Space-előtag mechanizmus (`_space_pressed`, `_sor_elotag`, `_space_szamlalo`,
`_kozotti_allapot`, `_space_hook`) maradványai törölve – a hook már semmilyen
úton nem települt fel, a kód hatástalan volt.

## Új funkció: hangparancsok diktálás közben

A kimondott parancsszavak írásjellé, sortöréssé vagy szmájlivá alakulnak.
A tanító queue a nyers (átalakítás előtti) szöveget kapja, így a tanítópárok
azt tükrözik, ami ténylegesen elhangzott.

**Írásjelek:** „pont" → `.` „vessző" → `,` „pontosvessző" → `;`
„kettőspont" → `:` „felkiáltójel" → `!` „kérdőjel" → `?`

**Szerkezet:** „új sor" → sortörés, „új bekezdés" → üres sor + a következő
mondat nagybetűvel kezdődik. Külön mondatként kimondva is működik (ilyenkor
csak a sortörés kerül beillesztésre).

**Szmájlik** (a „szmájli" kulcsszó kötelező, így normál beszédben nincs téves
csere; szmájli/smájli/smiley írásmód is jó):
„szmájli" → 🙂 „nevető szmájli" → 😂 „kacsintós szmájli" → 😉
„szomorú szmájli" → 😢 „dühös szmájli" → 😠 „puszi szmájli" → 😘
„szív szmájli" → ❤️ „lájk szmájli" → 👍

Figyelem: a „pont" és „vessző" valódi szóként is átalakul, ha önállóan
szerepel (pl. „pont ott volt" → „. ott volt"). A ragozott alakok („pontos",
„vesszük") nem érintettek. Ha ez zavaró, a lista a `HANGPARANCSOK` /
`SMAJLI_PARANCSOK` konstansokban szabadon szerkeszthető.

A CT2 konverzió mostantól a `preprocessor_config.json`-t is bemásolja a
modellmappába (128 mel-es, large-v3 alapú modellek helyes betöltéséhez).

## Ami szándékosan NEM változott (későbbi kör)

- A per-mondatos tanítási ciklus (LoRA → merge → CT2 konverzió mondatonként)
  továbbra is lassú; kötegesített átalakítása külön feladat.
- A katasztrofális felejtés elleni védelem (régi jó minták visszakeverése).
- A 2400 soros fájl modulokra bontása (audio / inference / training / winutils / gui).
