following Gollie:

**Wie du bisher Shortcuts "löst":**

Ehrlich gesagt — du löst sie bisher NICHT. Du UMGEHST sie.

In RQ1+RQ2 trainierst du nicht. Kein Training = keine Shortcuts. Das Modell nutzt sein Pre-Training + den Kontext den du ihm gibst. Das ist elegant aber es ist keine Lösung für das Shortcut-Problem — es ist eine Vermeidungsstrategie.

Feger's Frage war: "Modelle lernen Shortcuts WENN man sie trainiert." Deine Antwort bisher: "Dann trainieren wir halt nicht." Das ist valide als praktischer Ansatz, aber es beantwortet nicht ob man Shortcut-frei trainieren KANN.

**Genau das ist der Job von RQ4.**

**Was der beste Ansatz für Shortcut-freies Fine-Tuning wäre:**

Das Problem ist klar: Standard LoRA gibt dem Modell `sentence → "Argument"`. Der kürzeste Weg zum richtigen Label ist: finde das Content-Wort das am stärksten korreliert. Das IST der Shortcut.

**Die Lösung hat drei Ebenen die du KOMBINIEREN solltest:**

**Ebene 1: Definition-Conditioning (verhindert Dataset-Memorierung)**

Jeder Trainingsatz bekommt seine Dataset-spezifische Definition im System-Prompt. Über 10 Datasets hinweg sieht das Modell 10 verschiedene Definitionen. Es KANN nicht "ABSTRCT = medizinische Wörter → Argument" lernen, weil der nächste Batch FINARG mit komplett anderen Wörtern aber gleicher Prompt-Struktur ist. Die Definition wechselt ständig → das Modell muss die Definition LESEN.

```
System: "Classify based on this definition: [ABSTRCT definition]"
User: "Therefore, single-fraction radiotherapy should be considered..."
Assistant: "Argument"
```

Nächster Batch:

```
System: "Classify based on this definition: [FINARG definition]"
User: "I can take the first one, Brian, on the time spent metric."
Assistant: "No-Argument"
```

**Ebene 2: Reasoning-Targets (erzwingt strukturelles Verstehen)**

Statt `"Argument"` als Target gibst du dem Modell:

```
Assistant: "This sentence draws a conclusion about treatment
effectiveness based on study results, matching the definition
of a claim. Therefore: Argument"
```

Der Loss wird auf ~30 Tokens berechnet statt auf 1. Das Modell MUSS "conclusion", "treatment effectiveness", "claim" korrekt generieren. Es kann nicht abkürzen — die Reasoning-Kette erzwingt dass es die Definition auf den Satz anwendet.

**Ebene 3: GoLLIE-Style Regularisierung (verhindert Definition-Memorierung)**

Ohne Regularisierung könnte das Modell lernen: "Wenn Definition X im Prompt steht → Dataset Y → bekannte Content-Wörter." Die Definition wird zum Dataset-Identifier statt zur Instruktion.

GoLLIE löst das durch:

**Definition-Paraphrasierung:** Generiere 3-5 Umformulierungen jeder Definition mit GPT-4.1. Beim Training wird zufällig eine Variante gewählt. Das Modell kann sich nicht auf exakte Formulierungen verlassen.

**Definition-Dropout:** In 10% der Trainingsbeispiele wird die Definition weggelassen. Das Modell muss auch ohne Definition einen vernünftigen Versuch machen. Das verhindert dass es die Definition komplett ignoriert wenn sie da ist (weil es merkt dass es OHNE schlechter performed).

**Der komplette RQ4 Setup:**

```python
# Für jeden Trainingsatz:
definition = random.choice(definition_paraphrases[dataset])  # 1 von 5 Varianten
if random.random() < 0.1:
    definition = ""  # 10% Dropout

# Standard LoRA Target:
target = label  # "Argument" oder "No-Argument"

# Reasoning LoRA Target:
target = f"{reasoning_chain} Therefore: {label}"
```

**Zwei Modelle trainieren, gleiche Daten, gleiche Config:**

| Variante                                 | Definition im Prompt | Target            | Paraphrasierung  | Dropout  |
| ---------------------------------------- | -------------------- | ----------------- | ---------------- | -------- |
| A: Standard LoRA                         | Nein                 | Nur Label         | Nein             | Nein     |
| B: Definition-conditioned Reasoning LoRA | Ja (wechselnd)       | Reasoning + Label | Ja (5 Varianten) | Ja (10%) |

Dann Δ_content messen für beide. Die Differenz ist dein Beitrag.

**Warum das "Hammer Performance" liefern sollte:**

Variante B kombiniert drei Mechanismen die ALLE gegen Shortcuts arbeiten: wechselnde Definitionen verhindern Dataset-Memorierung, Reasoning-Targets erzwingen strukturelles Verstehen, und Regularisierung verhindert Definition-Memorierung.

Auf dem neuen GAIC Evaluation-Dataset bekommt das Modell eine NIE gesehene Definition — aber es hat gelernt WIE man Definitionen anwendet. Das ist Transfer auf Task-Ebene, nicht auf Content-Ebene.

**Aufwand:**

| Schritt                                                | Zeit        | Kosten   |
| ------------------------------------------------------ | ----------- | -------- |
| 5 Paraphrasen pro Definition generieren (10 Datasets)  | 1h          | ~$1      |
| 17k Reasoning-Ketten generieren (GPT-4.1)              | 3h          | ~$16     |
| 50 Ketten validieren                                   | 2h          | $0       |
| Training Variante A (Standard LoRA)                    | 2h auf A10G | ~$3      |
| Training Variante B (Definition-conditioned Reasoning) | 4h auf A10G | ~$6      |
| Evaluation + Δ Messung                                 | 1h          | ~$2      |
| **Total**                                              | **~2 Tage** | **~$28** |

2 Tage Arbeit. $28. Für eine Ablation die potenziell zeigt dass Definition-Conditioning + Reasoning-Targets = Shortcut-freies Training. Das ist der stärkste einzelne Beitrag deiner Thesis.
