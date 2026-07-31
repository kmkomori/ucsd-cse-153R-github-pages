# Sonic Co-Performer Agent — Project Report

**Student Name:** Kumiko Komori
**Student ID:** A18547845

*AI Usage: I used Claude for the coding and conceptual portions of this assignment, and also to help me organize and write this report. All of the results reported here come from running my own `submission.py`, and I checked the claims against the trace files and the sample audio myself.*

---

## 1. What the assignment is asking for (in my own words)

The goal of this project is to build an agent that "performs" music with a human partner. The human plays a short sound (a **cue**), and my agent has to do two things:

1. **Listen** to the cue and figure out which of six categories it belongs to (this is a classification problem — I hear a sound and label it).
2. **Respond** with the musically correct **action**, and then play a short sound back.

There are six cues, and each one has exactly one "correct" musical response. Here is the full mapping, which is basically the heart of the whole assignment:

| Cue | Correct action | What it means musically |
|-----|----------------|-------------------------|
| `call` | `mirror_topic` | Echo the phrase back |
| `answer` | `complement_topic` | Reply in the same key |
| `hold` | `wait` | Leave space, don't play over them |
| `interrupt` | `yield_repair` | Stop, then gracefully recover |
| `drift` | `repair_topic` | They wandered off — steer back to the theme |
| `end` | `close` | Land the final cadence (the ending) |

The tricky cue is **`drift`**. A `drift` cue is a *loud, attention-grabbing, off-topic lure*. The "obvious" but wrong thing to do is to mirror whatever is loudest — which means the agent gets dragged off topic. The correct thing is to notice the drift and **repair** back to the shared theme. Because a naive agent fails here, getting `drift` right is worth dedicated points, so I paid special attention to it.

So the whole program is a small pipeline: **audio in → features → cue label → action → audio out**, with a little bit of memory (state) carried between turns.

---

## 2. How the whole thing fits together (the pipeline)

Before diving into each piece, here is the overall flow, because it helped me a lot to keep the big picture in mind while coding:

```
raw audio (one cue)
      │
      ▼
extract_features()   → a fixed list of 27 numbers describing the sound
      │
      ▼
predict_cue()        → one of the six cue labels ("call", "drift", …)
      │
      ▼
choose_action()      → the correct action + updated state (memory)
      │
      ▼
synthesize_response()→ a short waveform the agent "plays" back
```

The four functions above (`extract_features`, `predict_cue`/`fit_cue_model`, `choose_action`/`reset_policy`, and `synthesize_response`) are the four parts of my `submission.py`. I'll walk through each one.

A quick note on terms, since I'm still fairly new to this: a **waveform** is just a long list of numbers, where each number is how far the speaker cone is pushed in or out at one tiny instant. My audio is sampled at **16,000 samples per second (16 kHz)**, so one second of sound is 16,000 numbers. Everything my code does is really just math on these lists of numbers.

---

## 3. Part 1 — Feature extraction (`extract_features`)

### 3.1 Why we need features at all

A one-second cue is 16,000 raw numbers. I can't just hand 16,000 raw numbers to a classifier and expect it to learn well — that's way too much, and most of it is redundant. Instead, I **summarize** each cue into a short list of meaningful measurements called **features**. Good features are ones that are *different* for different cues, so the classifier can tell them apart.

My `extract_features` function turns each cue into a fixed-length vector of **27 numbers**. "Fixed-length" matters: every cue, no matter how long, becomes exactly 27 numbers, so the classifier always sees the same shape. The vector is also **deterministic** (no randomness) and **finite** (I clean up any `NaN` or infinity at the end with `np.nan_to_num`), which keeps the results stable and reproducible.

I grouped my 27 features into six intuitive families. I'll explain each family in plain language and say *which cue* it's meant to help catch.

### 3.2 The six feature families

**(a) Loudness / energy — 7 features**
These measure how loud the clip is overall. The main one is **RMS** (root-mean-square), which is basically the average loudness of the whole clip. I also take its log (`log_rms`) so quiet sounds spread out more, the **peak** (loudest single sample), the 90th and 50th percentiles of the absolute signal (`p90`, `p50` — "how loud is a loud moment" vs "how loud is a typical moment"), and the **crest factor** (peak divided by RMS). Crest factor is high when a sound is *bursty* — a sharp spike over a quiet background — which is exactly what an `interrupt` looks like. Loudness in general helps a lot: `drift` is loud, and `hold` is nearly silent, so these two sit at opposite ends of this family.

**(b) Temporal energy envelope — 3 features**
Here I split the clip into 8 equal segments and measure the loudness of each. Then I fit a straight line through those 8 loudness values and take its **slope** (`env_slope`). A rising slope means the sound gets louder over time (like a `call` that builds), and a falling slope means it dies away (like an `answer` or `end` that settles). I also compute `env_ratio` (energy at the end divided by energy at the start) and `env_std` (how much the loudness bounces around). This family is about the *shape of the loudness over time*, not just the average.

**(c) Onset density / burstiness — 3 features**
This family looks at very short ~20 ms frames and asks "how punchy is this sound?" I compute the variance of the frame energies (`frame_energy_var`) and an **onset rate** (`onset_rate`), which counts how often the energy suddenly jumps up. Percussive, stop-start sounds — like `interrupt` — score high here, while smooth sustained tones score low.

**(d) Zero-crossing rate — 1 feature**
The **zero-crossing rate (ZCR)** counts how often the waveform crosses zero (flips from positive to negative). A clean, low tone crosses zero slowly; a noisy or high-pitched sound crosses zero much more often. So ZCR is a cheap way to tell "tonal vs noisy" and also acts as a rough pitch proxy.

**(e) Spectral shape — 5 features**
For this family I use the **FFT** (Fast Fourier Transform), which takes the waveform and tells me how much energy sits at each frequency — think of it as breaking the sound into its ingredient pitches. From that frequency picture I compute:
- **centroid** — the "center of mass" of the frequencies, i.e. how *bright* the sound is;
- **bandwidth** — how spread out the frequencies are around that center;
- **rolloff (85%)** — the frequency below which 85% of the energy lives;
- **flatness** — whether the sound is tonal (one clear pitch → low flatness) or noise-like (energy everywhere → high flatness);
- **dominant frequency** — the single loudest frequency, a rough estimate of the pitch.

**(f) Frequency motion — 6 features**
Finally, I split the clip into a first half and a second half and compare their pitch and brightness. `dom_shift` is (second-half dominant frequency − first-half dominant frequency): a **positive** value means the pitch *rose*, a **negative** value means it *fell*. This directly helps separate cues that rise (like `call`) from cues that fall (like `answer` and `end`). I keep the raw first/second-half values too (`dom1`, `dom2`, `cen1`, `cen2`) plus the centroid shift (`cen_shift`).

### 3.3 Summary of what each family "catches"

To keep it all straight, here's the one-line intuition for each cue:

- `drift` → very loud → loudness family
- `hold` → near-silent → loudness family (opposite end)
- `interrupt` → bursty / percussive → crest factor, onset rate, burstiness
- `end` → decays away → envelope slope going down
- `call` → rises → frequency-motion `dom_shift` positive
- `answer` → descends / settles in key → frequency-motion `dom_shift` negative

I like this design because the features are **interpretable** — each one has a physical meaning I can explain, rather than being a black box.

---

## 4. Part 2 — The cue classifier (`fit_cue_model`, `predict_cue`)

### 4.1 The model I chose

Once every cue is a vector of 27 numbers, I train a classifier to map those numbers to one of the six labels. I used a **scikit-learn `Pipeline`** with two steps:

1. **`StandardScaler`** — this rescales every feature so they're all on a comparable scale (roughly mean 0, spread 1). This matters because my features have wildly different units — duration is a small number like 1.0, while a frequency like the centroid can be in the thousands. Without scaling, the big-number features would unfairly dominate. Putting everything on the same footing lets the model treat each feature fairly.
2. **`RandomForestClassifier`** — a random forest is a collection of many **decision trees** (I used 100) that each vote on the answer, and the majority vote wins. Each tree learns simple "if this feature is above/below some threshold" rules. I picked a random forest because it's robust, handles features on different scales well, doesn't need much tuning, and works nicely on smallish datasets like this one. It's a good "default" classifier when you're starting out.

I set `random_state=0` so the training is **deterministic** — it gives the exact same model every time, which is important for reproducibility and for the autograder.

### 4.2 The fallback

I wrapped the training in a `try/except`. If scikit-learn is missing or the fit fails for some reason, the code doesn't crash — it **degrades gracefully** to a "majority-class baseline," which just predicts whichever cue was most common in training. That way the program always runs end to end, even in a broken environment. (In normal conditions the real random forest is always used.)

### 4.3 Predicting

`predict_cue` takes one waveform, runs it through the same `extract_features` function, reshapes it into a single row, and asks the trained pipeline for its prediction. It's important that training and prediction use the **exact same feature function**, otherwise the model would be looking at inconsistent inputs.

---

## 5. Part 3 — The interaction policy (`reset_policy`, `choose_action`)

### 5.1 The mapping

The policy is the part that decides what to *do* once I know the cue. My `choose_action` function looks up the cue in the `CUE_TO_ACTION` dictionary and returns the matching action. This is the correct mapping from the table in Section 1. If an unknown cue somehow shows up, the agent defaults to `wait` (leave space), which is the safest, least disruptive thing to do.

### 5.2 The drift decision (the important one)

This is where I made sure not to fall into the trap. It would be tempting to make the agent respond to whatever is loudest by mirroring it — but a `drift` cue is *designed* to be loud and off-topic. So mirroring it would drag the performance off the shared theme, which is exactly the failure the assignment warns about.

Instead, my policy maps `drift → repair_topic`. In plain terms: when the agent detects that the partner has wandered off, it does **not** follow them; it plays a gesture that steers back to the original theme. This is the single most important design decision in the policy, and getting it right is worth dedicated points.

### 5.3 Memory / state

The agent isn't stateless — it remembers a little bit between turns, which is what makes it feel like an ongoing performance rather than a series of disconnected reactions. `reset_policy` returns the starting state:

```python
{"turn": 0, "last_action": None, "repair_count": 0}
```

Then every call to `choose_action` returns an **updated** copy of the state (I deliberately copy it with `dict(state)` instead of editing the old one in place, so I don't accidentally cause side effects). On each turn it:

- increments `turn` (a simple turn counter),
- records `last_action` (what it just did), and
- increments `repair_count` whenever the action is a repairing one (`repair_topic` or `yield_repair`).

The `repair_count` is a nice summary of how "chaotic" the performance has been — how many times the agent had to recover or steer things back on track.

---

## 6. Part 4 — Response synthesis (`synthesize_response`)

### 6.1 The idea

After choosing an action, the agent has to *play something back*. `synthesize_response` renders a short mono waveform whose musical gesture matches the action's meaning. Each response is **peak-normalized to ±0.98** (so it's loud enough to hear but never clips), and I add a short 20 ms fade-in and fade-out so there are no clicks at the edges.

Every gesture is built around a **topic pitch**, which is derived from the `topic_id` so that different topics sound like different "keys." The base pitch is `220 Hz` moved up in whole-tone steps depending on the topic.

### 6.2 The six gestures

Each action gets its own distinctive gesture:

- **`mirror_topic`** — a steady, clear tone at the topic pitch. It literally echoes the topic back.
- **`complement_topic`** — the topic tone plus a **major third** on top (4 semitones up). Two notes that belong together, so it sounds like a reply "in the same key."
- **`wait`** — a very faint, slow "breath" (a quiet low tone gently pulsing). It's near-silence on purpose: the point is to *leave space* and not step on the partner.
- **`yield_repair`** — a single short **pluck** that decays quickly and then goes silent. This reduces the density right after an overlap, which is the "stop, then recover" idea.
- **`repair_topic`** — a pitch that starts *off* the topic (higher) and then **glides back down onto the topic pitch**. This is the sonic version of "you drifted, let me pull us back."
- **`close`** — a **descending cadence** that falls an octave and decays to zero. It sounds final, like the end of a piece.

### 6.3 Verifying the audio actually matches

I didn't just trust that the gestures were right — I looked at the sample output (`sample_responses.wav`) to confirm. That file is mono, 16 kHz, 16-bit, and 7.5 seconds long, and it contains the six gestures rendered one after another (each about 1.25 seconds). When I analyzed the pitch of each gesture over time, they matched the design exactly:

| Gesture | What I measured in the audio |
|---------|------------------------------|
| `mirror_topic` | steady tone (~221 Hz), constant loudness |
| `complement_topic` | topic tone plus the major third (~277 Hz) held together |
| `wait` | faint steady low tone (~109 Hz) |
| `yield_repair` | a ~221 Hz pluck that decays to silence about halfway through |
| `repair_topic` | pitch glides **down** from ~326 Hz back toward the topic (~230 Hz) |
| `close` | pitch descends from ~218 Hz down to ~128 Hz and fades out |

So the sweeps go the right direction, the silence appears where it should, and the "breath" really is faint. This gave me confidence that the synthesis half of the assignment works as intended.

---

## 7. Results

### 7.1 The policy trace

To evaluate the whole system, I ran a **15-turn trace** (saved in `policy_trace.json` / `policy_trace.csv`). Each turn records the true cue, my predicted cue, whether the prediction was correct, the action I chose, the target action, and whether the action was correct — plus the state (turn number, last action, repair count).

The headline result is that **every single turn is correct**:

- **Cue prediction: 15 / 15 correct** (`cue_correct = 1` on every turn)
- **Action choice: 15 / 15 correct** (`action_correct = 1` on every turn)

The trace exercises all six cues, including three separate `drift` turns (turns 5, 9, and 12). On each of those, the predicted cue is `drift` and the chosen action is `repair_topic` — never `mirror_topic`. That's the exact behavior the assignment is testing for, so I'm happy the drift handling holds up across multiple examples and different topics.

### 7.2 The repair counter over the trace

Because `repair_count` goes up on every `repair_topic` or `yield_repair`, I can read the trace as a story of the performance. It climbs at turn 5 (first drift → repair), turn 7 (an interrupt → yield), turns 9 and 12 (two more drifts → repairs), and ends at **4 repairs** total by the final `close` on turn 15. So over the 15-turn performance the agent had to steer back on topic or recover four times, and it landed the ending cleanly.

### 7.3 Runtime, determinism, and memory

A few practical properties I verified by running the code:

- **It runs end to end** with no errors, using the `if __name__ == "__main__"` smoke test at the bottom of the file.
- **It's deterministic.** The feature extraction has no randomness, and the classifier uses `random_state=0`, so I get identical results on every run — which is exactly what you want for grading and debugging.
- **The feature vector is always length 27**, finite, and shaped consistently, so nothing downstream breaks.
- **Memory is lightweight.** The state is just a tiny dictionary with three fields, copied cleanly each turn, so there are no growing lists or leaks over a long performance.

---

## 8. The drift problem, discussed on its own

Since the assignment calls out `drift` specifically, I want to explain my reasoning in one place.

A `drift` cue is a deliberate trap: it's loud and off-topic, so any strategy of "match the loudest / most salient thing" will chase it and lose the shared theme. My design resists this on two fronts:

1. **In the features**, I don't rely on loudness alone to decide anything — loudness is just one family among six. The classifier also sees the frequency motion, spectral shape, burstiness, and envelope, so a loud-but-off-topic sound doesn't automatically look like a `call` to be mirrored.
2. **In the policy**, the mapping is explicit and hard-coded to be correct: `drift → repair_topic`. There is no path where a `drift` cue leads to `mirror_topic`.

And in the **synthesis**, the `repair_topic` gesture makes the recovery audible — it literally starts off-pitch and glides back home. Across all three drift turns in the trace, this worked, so I'm confident the "steer back instead of follow" behavior is solid.

---

## 9. Limitations and what I would try next

I want to be honest about what could be better, since I'm still learning:

- **The trace is small and clean.** 15 turns with perfect scores is encouraging, but a bigger, noisier test set (with quieter recordings, background noise, or cues that blur together) would be a tougher and fairer test. Perfect accuracy on a small trace doesn't guarantee perfect accuracy in general.
- **My features are hand-designed.** I chose them because I could reason about them, but more standard audio features like **MFCCs** (a common way to summarize the shape of a sound's spectrum) might capture things I'm missing, especially for the subtle differences between `call` and `answer`.
- **The policy is a fixed lookup table.** That's exactly right for this assignment, but a more advanced agent might use the state more — for example, adapting its response if the partner keeps drifting, instead of treating every turn independently.
- **Synthesis is simple.** The gestures are basic sine tones. They clearly communicate the right musical intent, but with more time I'd add richer timbres so the responses sound less synthetic.

If I extended this project, my first two steps would be to (1) test on a larger, noisier dataset to see how well the classifier really generalizes, and (2) try adding MFCC features to see if they improve the harder-to-separate cues.

---

## 10. Conclusion

I built a small but complete co-performer agent that listens to a musical cue, classifies it into one of six categories using 27 interpretable acoustic features and a random-forest classifier, chooses the musically correct action with a stateful policy, and plays back a matching sound. The most important design goal — handling `drift` by **repairing back to the theme instead of being pulled off topic** — works correctly across every drift example in the trace. On the 15-turn evaluation, the system predicted every cue correctly (15/15) and chose every action correctly (15/15), it runs deterministically end to end, and the synthesized audio matches each gesture's intended musical meaning. Working through this project taught me how a full audio-ML pipeline fits together, from raw samples all the way to a musical response.
