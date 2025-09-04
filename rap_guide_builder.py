#!/usr/bin/env python3
"""
Rap Guide Builder
- Generates a 1-bar hi-hat count-in
- Synthesizes a neutral mid-tone robotic rap guide for 16 bars (pyttsx3, offline)
- Aligns each bar evenly across the instrumental (or uses --bpm/--timesig if provided)
- Mixes a balanced "beat + guide" MP3 and a "guide only" MP3
- Cuts the beat immediately after bar 16
Requires:
  pip install pydub pyttsx3 numpy
Also install ffmpeg (so pydub can export MP3 at 320 kbps)
"""
import os
import argparse
import tempfile
import math
import numpy as np
from pydub import AudioSegment, effects
from pydub.playback import play

try:
    import pyttsx3
except Exception as e:
    pyttsx3 = None

# ------------------------- Lyrics (16 bars) -------------------------
LYRICS_16 = [
    "Brown eyes jaise mocha, bitter sweet ka nasha,",
    "Sip karun dheere, par dil uska jalta bhari ghunt jaisa.",
    "Kehte rishte uske liye chain aur cage,",
    "Main locksmith, bars todun jaise verse on stage.",
    "She say main broken, main bolun tu mosaic,",
    "Toot ke bhi khubsurat, jaise shattered glass aesthetic.",
    "Past ka weight uske kandhe pe hai still,",
    "Main spotter, gym floor pe uthata har guilt.",
    "Commitment ka darr, jaise shayari mein pause,",
    "Main comma banke ruk jaun, par full stop kabhi na hoon boss.",
    "Uske zakhm, likhe diary pe invisible ink,",
    "Main torchlight, har lafz ko karun visible quick.",
    "Kehte pyaar khatam, main rhyme ka revival,",
    "Uske dil mein likhun hook, jaise chorus ka survival.",
    "She drown deep, main metaphor ka float,",
    "Dooba hoon usmein jaise paper pe quote."
]

# ------------------------- Helpers -------------------------

def dBFS(segment: AudioSegment) -> float:
    return segment.dBFS if segment.dBFS != float("-inf") else -90.0

def make_hihat_tick(duration_ms=80, sr=44100, peak_db=-6):
    """Approximate hi-hat tick using shaped noise with exponential decay + high-pass feel."""
    n = int(sr * (duration_ms/1000.0))
    noise = np.random.randn(n).astype(np.float32)

    # Exponential decay envelope for a crisp tick
    t = np.linspace(0, 1, n, endpoint=False)
    env = np.exp(-t*40.0)  # fast decay
    sig = noise * env

    # High-pass flavor: subtract a moving-average (very crude HP filter)
    k = max(1, int(sr*0.001))  # ~1ms moving average
    kernel = np.ones(k)/k
    low = np.convolve(sig, kernel, mode='same')
    hp = sig - low

    # Normalize
    hp = hp / (np.max(np.abs(hp)) + 1e-9)
    # Convert to 16-bit PCM
    pcm = (hp * 32767).astype(np.int16).tobytes()

    seg = AudioSegment(
        data=pcm,
        sample_width=2,
        frame_rate=sr,
        channels=1
    )
    # Set level
    seg = seg.apply_gain(peak_db - seg.max_dBFS if hasattr(seg, "max_dBFS") else -3)
    return seg

def change_speed_to_match_length(seg: AudioSegment, target_ms: int) -> AudioSegment:
    """Time-stretch (pitch-shifted) by changing frame_rate so that len(seg) == target_ms."""
    if len(seg) == 0 or target_ms <= 0:
        return AudioSegment.silent(duration=target_ms, frame_rate=seg.frame_rate if seg.frame_rate else 44100)
    rate = len(seg) / float(target_ms)
    new_frame_rate = int(seg.frame_rate * rate)
    # Speed up/down by altering frame rate, then reset to original frame rate
    stretched = seg._spawn(seg.raw_data, overrides={'frame_rate': new_frame_rate}).set_frame_rate(seg.frame_rate)
    # If due to rounding it's slightly off, pad/trim
    if len(stretched) < target_ms:
        stretched += AudioSegment.silent(duration=target_ms - len(stretched), frame_rate=stretched.frame_rate)
    else:
        stretched = stretched[:target_ms]
    return stretched

def synth_tts_line(text: str, voice_pref: str = "neutral"):
    """Synthesize a single line to WAV temp file and return AudioSegment. Uses pyttsx3 offline."""
    if pyttsx3 is None:
        raise RuntimeError("pyttsx3 is not installed. Please run: pip install pyttsx3")

    engine = pyttsx3.init()
    # Try to pick a neutral mid-tone voice if available
    try:
        voices = engine.getProperty("voices")
        chosen_id = None
        for v in voices:
            name = (v.name or "").lower()
            lang = ",".join(v.languages) if hasattr(v, "languages") else ""
            if "female" in name or "neutral" in name:
                chosen_id = v.id
                break
        if chosen_id is None and voices:
            chosen_id = voices[0].id
        if chosen_id:
            engine.setProperty("voice", chosen_id)
    except Exception:
        pass

    # Slightly faster speech for rap bounce, adjust rate modestly
    rate = engine.getProperty('rate')
    engine.setProperty('rate', int(rate * 1.05))  # subtle bump

    # Slightly increase volume to avoid getting buried in the mix
    vol = engine.getProperty('volume')
    engine.setProperty('volume', min(1.0, vol * 1.0))

    tmp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp_wav_path = tmp_wav.name
    tmp_wav.close()

    engine.save_to_file(text, tmp_wav_path)
    engine.runAndWait()

    seg = AudioSegment.from_file(tmp_wav_path)
    os.unlink(tmp_wav_path)
    # Mild dynamic processing: normalize to -3 dBFS
    seg = effects.normalize(seg).apply_gain(-3 - seg.max_dBFS if hasattr(seg, "max_dBFS") else -3)
    return seg

def build_count_in(bar_ms: int):
    """Create a 1-bar hi-hat count-in: tik tik tik tik (4/4), last tick accented."""
    sr = 44100
    tick = make_hihat_tick(duration_ms=75, sr=sr, peak_db=-6)
    accent = make_hihat_tick(duration_ms=85, sr=sr, peak_db=-3)
    quarter = bar_ms // 4
    silence_gap = max(0, quarter - len(tick))
    one = tick + AudioSegment.silent(duration=silence_gap, frame_rate=sr)
    two = tick + AudioSegment.silent(duration=silence_gap, frame_rate=sr)
    three = tick + AudioSegment.silent(duration=silence_gap, frame_rate=sr)
    four = accent + AudioSegment.silent(duration=max(0, quarter - len(accent)), frame_rate=sr)
    return one + two + three + four

def main():
    parser = argparse.ArgumentParser(description="Build a 16-bar robotic rap guide over your beat.")
    parser.add_argument("--beat", required=True, help="Path to your instrumental MP3/WAV (the one you uploaded).")
    parser.add_argument("--out_dir", default=".", help="Output directory for MP3s.")
    parser.add_argument("--bpm", type=float, default=None, help="If provided, used to compute exact bar length (4/4).")
    parser.add_argument("--timesig_top", type=int, default=4, help="Time signature numerator (default 4).")
    parser.add_argument("--timesig_bottom", type=int, default=4, help="Time signature denominator (default 4).")
    parser.add_argument("--voice", default="neutral", help="Voice preference hint (neutral|male|female).")
    parser.add_argument("--bitrate", default="320k", help="MP3 export bitrate (default 320k).")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Load beat
    beat = AudioSegment.from_file(args.beat)
    sr = beat.frame_rate

    # Decide bar length
    if args.bpm is not None and args.bpm > 0:
        # bar length (ms) = (60_000 ms per minute * time signature (beats per bar)) / BPM
        beats_per_bar = args.timesig_top
        bar_ms = int((60000.0 * beats_per_bar) / args.bpm)
    else:
        # Distribute evenly across 16 bars using the current beat length
        bar_ms = int(len(beat) / 16.0)
    # 1-bar count in (silence + hihat)
    count_in = build_count_in(bar_ms)

    # Prepare timeline: silence for count-in + 16 bars vocal/beat
    guide_len_ms = bar_ms * 16
    pre_roll_silence = AudioSegment.silent(duration=bar_ms, frame_rate=sr)  # silence under count-in
    # Trim beat to 16 bars exactly (start after count-in)
    beat_cut = beat[:guide_len_ms]

    # Synthesize lines and time-stretch to bar length with a tiny swing
    swing_offsets_ms = [0, 60, 30, 45, 0, 70, 40, 50, 0, 60, 30, 45, 0, 70, 40, 50]  # light bounce
    vocal_track = AudioSegment.silent(duration=guide_len_ms, frame_rate=sr)

    for i, line in enumerate(LYRICS_16):
        # Synthesize
        seg = synth_tts_line(line, voice_pref=args.voice)
        # Aim to leave a touch of breathing room at end of each bar
        target_ms = max(250, bar_ms - 150)
        seg = change_speed_to_match_length(seg, target_ms)

        # Place with slight swing offset (but keep within bar)
        start = i * bar_ms + min(swing_offsets_ms[i], int(bar_ms*0.25))
        vocal_track = vocal_track.overlay(seg, position=start)

    # Build master: [count-in over silence] + [beat_cut + vocals]
    body_mix = beat_cut.overlay(vocal_track, position=0)
    full_mix = count_in.overlay(pre_roll_silence, position=0) + body_mix  # count-in then music+vocals
    # cut beat exactly after 16 bars -> done by beat_cut length

    # Exports
    out_mix = os.path.join(args.out_dir, "beat_plus_guide_16bars.mp3")
    out_vox = os.path.join(args.out_dir, "guide_vocal_only_16bars.mp3")

    # Guide only export: count-in + vocals only
    guide_only = count_in + vocal_track

    print(f"[i] Bar length: {bar_ms} ms | Total vocal body: {guide_len_ms/1000.0:.2f} s")
    print(f"[i] Exporting MP3s at {args.bitrate}...")

    full_mix.export(out_mix, format="mp3", bitrate=args.bitrate)
    guide_only.export(out_vox, format="mp3", bitrate=args.bitrate)

    print(f"[✓] Wrote: {out_mix}")
    print(f"[✓] Wrote: {out_vox}")
    print("[Tip] If export fails, install ffmpeg and ensure it's on your PATH.")

if __name__ == "__main__":
    main()
