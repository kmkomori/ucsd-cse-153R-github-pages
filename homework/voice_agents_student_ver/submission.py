"""
submission.py -- Sonic Co-Performer Agent  (STUDENT STARTER SCAFFOLD)
=====================================================================

This is the file you edit and submit. It already RUNS end to end and passes the
API, synthesis, runtime, and memory checks, but the two parts that matter --
hearing the cue and choosing the musically correct response -- are left as
clearly marked ``TODO``s. Out of the box this scaffold scores roughly 30/80:

    setup/API   3      synthesis 12      runtime 8      memory 7
    cue         0  <-- YOUR JOB (train a real classifier)
    policy      0  <-- YOUR JOB (map each cue to the right action)

Your agent listens to a short audio "cue" from a human co-performer and answers
with an action. The six cues and their musically correct actions are:

    call      -> mirror_topic        (echo the phrase back)
    answer    -> complement_topic    (respond in the same key)
    hold      -> wait                (leave space; don't step on them)
    interrupt -> yield_repair        (stop, then gracefully recover)
    drift     -> repair_topic        (they wandered off -- steer back)
    end       -> close               (land the final cadence)

The interesting one is ``drift``: it is a LOUD, attention-grabbing off-topic
lure. A naive agent mirrors whatever is loudest and gets pulled off topic; a
good agent recognizes the drift and *repairs* back to the shared theme. Getting
drift right is worth dedicated points.

Do NOT change the function names or signatures below -- the autograder calls
them exactly as written. You MAY add helper functions, features, and imports
from numpy / scipy / scikit-learn / joblib (CPU only).
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

# Canonical mappings (provided for convenience; the grader uses its own copy).
SR_DEFAULT = 16000
CUES = ("call", "answer", "hold", "interrupt", "drift", "end")
CUE_TO_ACTION = {
    "call": "mirror_topic",
    "answer": "complement_topic",
    "hold": "wait",
    "interrupt": "yield_repair",
    "drift": "repair_topic",
    "end": "close",
}
ACTIONS = tuple(CUE_TO_ACTION[c] for c in CUES)


# --------------------------------------------------------------------------- #
# 1. Feature extraction
# --------------------------------------------------------------------------- #

def _spectral_stats(x: np.ndarray, sr: int) -> Tuple[float, float, float, float, float]:
    """Return (centroid, bandwidth, rolloff85, flatness, dominant_freq) in Hz.

    Computed from the magnitude spectrum of a Hann-windowed frame. All values are
    finite; a silent frame collapses to zeros.
    """
    n = x.size
    if n < 2:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    win = np.hanning(n)
    spec = np.abs(np.fft.rfft(x * win))
    freqs = np.fft.rfftfreq(n, d=1.0 / float(sr if sr else SR_DEFAULT))
    total = float(spec.sum())
    if total <= 1e-12:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    p = spec / total                                   # spectral prob. mass
    centroid = float(np.sum(freqs * p))
    bandwidth = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * p)))
    cumulative = np.cumsum(spec)
    rolloff_idx = int(np.searchsorted(cumulative, 0.85 * total))
    rolloff_idx = min(rolloff_idx, freqs.size - 1)
    rolloff = float(freqs[rolloff_idx])
    # spectral flatness = geometric mean / arithmetic mean of the power spectrum
    power = spec ** 2 + 1e-12
    flatness = float(np.exp(np.mean(np.log(power))) / np.mean(power))
    dominant = float(freqs[int(np.argmax(spec))])
    return centroid, bandwidth, rolloff, flatness, dominant


def extract_features(audio: np.ndarray, sr: int) -> np.ndarray:
    """Turn one cue waveform into a fixed-length feature vector.

    Interpretable acoustic features chosen to separate the six cues:

      * loudness / energy          -- drift is loud; hold is near-silent
      * temporal energy envelope   -- end decays; interrupt is bursty
      * onset / burstiness         -- interrupt is percussive
      * zero-crossing rate         -- noisy vs tonal, rough pitch proxy
      * spectral shape             -- centroid / bandwidth / rolloff / flatness
      * frequency motion           -- call rises, answer / end descend

    The vector is 1-D, finite and deterministic (no randomness).
    """
    x = np.asarray(audio, dtype=np.float64).ravel()
    if x.size == 0:
        x = np.zeros(1, dtype=np.float64)
    fs = float(sr if sr else SR_DEFAULT)
    n = x.size
    duration_s = n / fs

    # --- global loudness / energy ------------------------------------------
    rms = float(np.sqrt(np.mean(x ** 2)))
    log_rms = float(np.log(rms + 1e-8))
    peak = float(np.max(np.abs(x)))
    abs_x = np.abs(x)
    p90 = float(np.percentile(abs_x, 90))
    p50 = float(np.percentile(abs_x, 50))
    crest = peak / (rms + 1e-8)                         # peak-to-RMS (bursty -> high)

    # --- temporal energy envelope (per-segment RMS) ------------------------
    n_seg = 8
    seg = np.array_split(x, n_seg)
    seg_rms = np.array([np.sqrt(np.mean(s ** 2)) if s.size else 0.0 for s in seg])
    # slope of the loudness envelope: rising (call) vs falling (answer/end)
    idx = np.arange(n_seg, dtype=np.float64)
    env_slope = float(np.polyfit(idx, seg_rms, 1)[0]) if seg_rms.std() > 0 else 0.0
    env_ratio = float((seg_rms[-1] + 1e-8) / (seg_rms[0] + 1e-8))   # end/start energy
    env_std = float(seg_rms.std())

    # --- onset density / burstiness (short frames) -------------------------
    frame = max(1, int(0.02 * fs))                     # ~20 ms frames
    hop = frame
    n_frames = max(1, n // hop)
    fe = np.array([np.sqrt(np.mean(x[i * hop:i * hop + frame] ** 2))
                   for i in range(n_frames)])
    fe = fe if fe.size else np.zeros(1)
    frame_energy_var = float(fe.var())
    # count sharp positive energy jumps (percussive onsets)
    if fe.size > 1:
        diffs = np.diff(fe)
        thresh = 0.5 * (fe.max() + 1e-8)
        onset_rate = float(np.mean(diffs > thresh))
    else:
        onset_rate = 0.0

    # --- zero-crossing rate ------------------------------------------------
    signs = np.sign(x)
    signs[signs == 0] = 1
    zcr = float(np.mean(np.abs(np.diff(signs)) > 0)) if n > 1 else 0.0

    # --- spectral shape (whole clip) ---------------------------------------
    centroid, bandwidth, rolloff, flatness, dominant = _spectral_stats(x, sr)

    # --- frequency motion: first half vs second half -----------------------
    half = n // 2
    if half >= 2:
        cen1, _, _, _, dom1 = _spectral_stats(x[:half], sr)
        cen2, _, _, _, dom2 = _spectral_stats(x[half:], sr)
    else:
        dom1 = dom2 = dominant
        cen1 = cen2 = centroid
    dom_shift = float(dom2 - dom1)                     # rising (+) vs falling (-)
    cen_shift = float(cen2 - cen1)

    feats = np.array([
        duration_s, rms, log_rms, peak, p90, p50, crest,
        env_slope, env_ratio, env_std,
        seg_rms[0], seg_rms[len(seg_rms) // 2], seg_rms[-1],
        frame_energy_var, onset_rate, zcr,
        centroid, bandwidth, rolloff, flatness, dominant,
        dom1, dom2, dom_shift, cen1, cen2, cen_shift,
    ], dtype=np.float64)
    return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)


# --------------------------------------------------------------------------- #
# 2. Cue classifier
# --------------------------------------------------------------------------- #

def fit_cue_model(train_items: List[Dict[str, Any]]):
    """Train a model that maps a cue waveform to one of the six cue labels.

    Extracts ``extract_features`` for every training item and fits a
    ``Pipeline(StandardScaler, RandomForestClassifier)``. Deterministic via a
    fixed ``random_state``. Falls back to a majority-class baseline only if
    scikit-learn is unavailable.
    """
    labels = [it["cue"] for it in train_items]
    counts = {c: labels.count(c) for c in CUES}
    majority = max(CUES, key=lambda c: counts[c])

    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        X = np.array([extract_features(it["audio"], it.get("sr", SR_DEFAULT))
                      for it in train_items], dtype=np.float64)
        y = np.array(labels)
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("rf", RandomForestClassifier(
                n_estimators=300, random_state=0, n_jobs=-1)),
        ])
        pipe.fit(X, y)
        return {"model_type": "rf", "pipeline": pipe,
                "classes": sorted(set(labels)), "majority_class": majority}
    except Exception:
        # Dependency missing or fit failed -- degrade to majority baseline.
        return {"model_type": "majority-baseline", "majority_class": majority,
                "classes": sorted(set(labels))}


def predict_cue(model, audio: np.ndarray, sr: int) -> str:
    """Predict the cue label for one waveform."""
    pipe = model.get("pipeline") if isinstance(model, dict) else None
    if pipe is not None:
        feats = extract_features(audio, sr).reshape(1, -1)
        return str(pipe.predict(feats)[0])
    # No trained pipeline -- fall back to the stored majority class.
    return str(model.get("majority_class", "hold"))


# --------------------------------------------------------------------------- #
# 3. Interaction policy
# --------------------------------------------------------------------------- #

def reset_policy() -> dict:
    """Return the agent's initial conversational state (fresh performance)."""
    return {"turn": 0, "last_action": None, "repair_count": 0}


def choose_action(cue: str, features: np.ndarray, state: dict) -> Tuple[str, dict]:
    """Given the (predicted) cue and current state, choose an action.

    Maps each cue to its musically correct action -- crucially, ``drift`` is
    *repaired* back to topic rather than mirrored. Advances the turn counter,
    records the last action, and tracks how many repairs have happened.
    """
    new_state = dict(state)
    new_state["turn"] = int(state.get("turn", 0)) + 1

    # Correct cue -> action mapping; unknown cues leave space (wait).
    action = CUE_TO_ACTION.get(cue, "wait")

    if action in ("repair_topic", "yield_repair"):
        new_state["repair_count"] = int(state.get("repair_count", 0)) + 1
    new_state["last_action"] = action
    return action, new_state


# --------------------------------------------------------------------------- #
# 4. Response synthesis
# --------------------------------------------------------------------------- #

def synthesize_response(action: str, topic_id: int, sr: int = 16000,
                        duration: float = 1.0) -> np.ndarray:
    """Render a short mono waveform whose gesture matches the action's function.

    Six distinct gestures (each peak-normalized to [-1, 1], finite, mono):

      mirror_topic     -- a clear tone at the topic pitch (references the topic)
      complement_topic -- the topic tone plus a related interval (a major third)
      wait             -- near-silence: a faint breath that preserves space
      yield_repair     -- a brief decaying pluck: reduces density after overlap
      repair_topic     -- an off-topic wobble that resolves back to the topic
      close            -- a descending cadence that decays to zero (final)
    """
    n = max(1, int(round(duration * sr)))
    t = np.arange(n) / float(sr)
    # Topic id -> a base pitch (four topics over a musical range).
    topic = 220.0 * (2.0 ** ((int(topic_id) % 4) * 2 / 12.0))  # whole-tone steps

    if action == "mirror_topic":
        # echo the topic: a steady tone at the topic pitch
        wave = np.sin(2 * np.pi * topic * t)
    elif action == "complement_topic":
        # same key, a related interval on top (major third, 4 semitones)
        third = topic * (2.0 ** (4 / 12.0))
        wave = 0.6 * np.sin(2 * np.pi * topic * t) + 0.6 * np.sin(2 * np.pi * third * t)
    elif action == "wait":
        # preserve space: a very faint, slow breath (near-silence)
        wave = 0.05 * np.sin(2 * np.pi * 110.0 * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 1.5 * t))
    elif action == "yield_repair":
        # reduce density: a single short decaying pluck, then silence
        pluck = np.sin(2 * np.pi * topic * t) * np.exp(-6.0 * t)
        wave = np.zeros(n)
        cut = int(0.4 * n)
        wave[:cut] = pluck[:cut]
    elif action == "repair_topic":
        # wander off then glide back to the topic pitch (return to topic)
        f = np.linspace(topic * 1.5, topic, n)          # sweep down onto the topic
        wave = np.sin(2 * np.pi * np.cumsum(f) / float(sr))
    elif action == "close":
        # cadential ending: descend an octave and decay to zero
        f = np.linspace(topic, topic / 2.0, n)
        wave = np.sin(2 * np.pi * np.cumsum(f) / float(sr)) * np.exp(-2.5 * t)
    else:
        wave = np.sin(2 * np.pi * topic * t)

    # gentle fade in/out so it is click-free
    fade = max(1, int(0.02 * sr))
    if 2 * fade < n:
        env = np.ones(n)
        env[:fade] = np.linspace(0.0, 1.0, fade)
        env[-fade:] = np.linspace(1.0, 0.0, fade)
        wave = wave * env

    wave = np.nan_to_num(wave, nan=0.0, posinf=0.0, neginf=0.0)
    peak = float(np.max(np.abs(wave)))
    if peak > 0:
        wave = 0.98 * wave / peak                       # peak-normalize to [-1, 1]
    return wave.astype(np.float64)


if __name__ == "__main__":  # tiny manual smoke test
    sr = SR_DEFAULT
    demo = np.sin(2 * np.pi * 440 * np.arange(sr) / sr).astype(np.float64)
    feats = extract_features(demo, sr)
    print("features:", feats)
    model = fit_cue_model([{"cue": c, "audio": demo, "sr": sr} for c in CUES])
    print("predict_cue:", predict_cue(model, demo, sr))
    st = reset_policy()
    for c in CUES:
        a, st = choose_action(c, feats, st)
        print(f"  {c:9s} -> {a}")
    for a in ACTIONS:
        w = synthesize_response(a, 0, sr, 1.0)
        print(f"  synth {a:16s} len={len(w)} peak={np.max(np.abs(w)):.3f}")
