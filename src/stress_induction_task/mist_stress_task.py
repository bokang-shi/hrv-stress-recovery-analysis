import tkinter as tk
from tkinter import ttk, messagebox
import time
import random
import csv
import os
from datetime import datetime

# =========================
# CONFIG (safe to edit)
# =========================
PRACTICE_SEC = 30
STRESS_SEC = 10 * 60

MIN_DIFFICULTY = 1
MAX_DIFFICULTY = 4
START_DIFFICULTY = 2

STREAK_UP = 3
STREAK_DOWN = 3

# Time limit bounds
TL_MIN = 3.0
TL_MAX = 7.0          # max time cap
TL_DEFAULT = 5.0      # stress starts fresh at this

# Cohort average (green) bar fixed level
EXPECTED_AVG = 0.85

# Deceptive performance bar bounds (red)
DECEPTIVE_MIN = 0.40
DECEPTIVE_MAX = 0.60

# Performance bar behaviour
FIRST_PERF_PCT = 0.53          # first shown red bar after first submitted answer in stress (53%)

# Reduced jump sizes make the displayed bar smoother / less obvious.
PERF_STEP_RIGHT = 0.007        # +0.5% on correct
PERF_STEP_WRONG = 0.015       # -0.7% on incorrect/timeout
PERF_JITTER = 0.0025           # small noise to avoid robotic moves (+/-0.15%)

ITI_SEC = 0.6
ALLOW_MULTIPLICATION = False
WINDOW_GEOMETRY = "920x720"
LOG_DIR = os.path.join("outputs", "mist_logs")


# =========================
# Helpers
# =========================
def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def now_epoch():
    return time.time()

def fmt_mmss(seconds):
    seconds = max(0, int(seconds))
    m = seconds // 60
    s = seconds % 60
    return f"{m:02d}:{s:02d}"

def generate_problem(difficulty: int):
    if difficulty == 1:
        a = random.randint(1, 9)
        b = random.randint(1, 9)
        op = random.choice(["+", "-"])
    elif difficulty == 2:
        a = random.randint(10, 99)
        b = random.randint(1, 9) if random.random() < 0.5 else random.randint(10, 99)
        op = random.choice(["+", "-"])
    elif difficulty == 3:
        a = random.randint(30, 99)
        b = random.randint(30, 99)
        op = random.choice(["+", "-"])
    else:
        a = random.randint(100, 999)
        b = random.randint(10, 99)
        op = random.choice(["+", "-"])

    if ALLOW_MULTIPLICATION and difficulty >= 3 and random.random() < 0.15:
        a = random.randint(6, 19)
        b = random.randint(3, 9)
        op = "*"

    if op == "+":
        ans = a + b
    elif op == "-":
        ans = a - b
    else:
        ans = a * b

    return f"{a} {op} {b} = ?", ans


# =========================
# Canvas progress bar (guaranteed colours)
# =========================
class CanvasBar(tk.Canvas):
    """
    A simple coloured progress bar drawn on a Canvas.
    percent: 0..100
    """
    def __init__(self, parent, height=18, fill="#ff9900", trough="#d9d9d9", border="#b0b0b0", **kwargs):
        super().__init__(parent, height=height, highlightthickness=0, bd=0, **kwargs)
        self._fill = fill
        self._trough = trough
        self._border = border
        self._percent = 0
        self.bind("<Configure>", lambda e: self._redraw())
        self._redraw()

    def set_percent(self, percent: float):
        self._percent = clamp(percent, 0.0, 100.0)
        self._redraw()

    def _redraw(self):
        w = max(1, self.winfo_width())
        h = max(1, self.winfo_height())
        self.delete("all")
        self.create_rectangle(0, 0, w, h, fill=self._trough, outline=self._border)
        fill_w = int(w * (self._percent / 100.0))
        if fill_w > 0:
            self.create_rectangle(0, 0, fill_w, h, fill=self._fill, outline=self._fill)


# =========================
# Main App
# =========================
class MISTApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("MIST (GUI) - Practice / Stress")
        self.geometry(WINDOW_GEOMETRY)
        self.minsize(860, 640)
        self.resizable(True, True)

        # ---- Experiment state ----
        self.phase = "idle"
        self.phase_end_epoch = None

        self.current_question = ""
        self.current_answer = None
        self.trial_start_epoch = None

        self.difficulty = START_DIFFICULTY
        self.time_limit = TL_DEFAULT

        # Deceptive displayed performance (stress only), kept as fraction 0..1
        self.displayed_perf = 0.50

        # True scoring (internal)
        self.trials_total = 0
        self.trials_correct = 0
        self.streak_correct = 0
        self.streak_bad = 0

        # Timers
        self._phase_tick_id = None
        self._trial_bar_tick_id = None
        self._stress_timeout_id = None

        # Logging
        self.session_start = now_epoch()
        self.log_rows = []

        # Track whether bar block is visible
        self._bars_visible = True

        # performance bar is "armed" only after first submitted answer in stress
        self._perf_armed = False
        self._perf_first_set = False

        self._build_ui()
        self._log_event("session_start", {"note": "GUI opened"})
        self._set_idle_screen()

    # ---------------- UI ----------------
    def _build_ui(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        top = ttk.Frame(self, padding=12)
        top.grid(row=0, column=0, sticky="ew")
        top.columnconfigure(2, weight=1)

        self.title_label = ttk.Label(top, text="MIST (GUI) Adaptation", font=("Segoe UI", 18, "bold"))
        self.title_label.grid(row=0, column=0, sticky="w")

        self.start_btn = ttk.Button(top, text="Start Session", command=self.start_session)
        self.start_btn.grid(row=0, column=1, padx=(12, 0), sticky="e")

        self.quit_btn = ttk.Button(top, text="Quit", command=self.on_quit)
        self.quit_btn.grid(row=0, column=3, sticky="e")

        subtitle = ttk.Label(
            top,
            text="Practice (30s) -> Arithmetic (10min). Practice is only for familiarisation.",
            font=("Segoe UI", 10),
        )
        subtitle.grid(row=1, column=0, columnspan=4, sticky="w", pady=(6, 0))

        main = ttk.Frame(self, padding=12)
        main.grid(row=1, column=0, sticky="nsew")
        main.columnconfigure(0, weight=1)
        main.rowconfigure(9, weight=1)

        self.phase_label = ttk.Label(main, text="Phase: Idle", font=("Segoe UI", 12, "bold"))
        self.phase_label.grid(row=0, column=0, sticky="w")

        self.timer_label = ttk.Label(main, text="Time Remaining: --:--", font=("Segoe UI", 11))
        self.timer_label.grid(row=1, column=0, sticky="w", pady=(6, 10))

        self.display_label = ttk.Label(
            main, text="Ready", font=("Segoe UI", 40, "bold"), anchor="center", justify="center"
        )
        self.display_label.grid(row=2, column=0, sticky="ew", pady=(10, 8))

        ans_row = ttk.Frame(main)
        ans_row.grid(row=3, column=0, pady=(8, 0), sticky="ew")
        ans_row.columnconfigure(0, weight=1)

        self.answer_var = tk.StringVar()
        self.answer_entry = ttk.Entry(ans_row, textvariable=self.answer_var, font=("Segoe UI", 14))
        self.answer_entry.grid(row=0, column=0, sticky="ew")

        self.submit_btn = ttk.Button(ans_row, text="Submit", command=self.on_submit)
        self.submit_btn.grid(row=0, column=1, padx=(10, 0))

        # Enter / Numpad Enter / Space submit (same as clicking Submit)
        self.answer_entry.bind("<Return>", self._on_submit_key)
        self.answer_entry.bind("<KP_Enter>", self._on_submit_key)
        self.answer_entry.bind("<space>", self._on_submit_key)

        self.feedback_label = ttk.Label(main, text="", font=("Segoe UI", 12))
        self.feedback_label.grid(row=4, column=0, sticky="w", pady=(12, 6))

        # ---- Bars block ----
        self.trial_label = ttk.Label(main, text="Trial timer", font=("Segoe UI", 10, "bold"))
        self.trial_label.grid(row=5, column=0, sticky="w")

        self.trial_bar = CanvasBar(main, height=18, fill="#f5a623")  # orange
        self.trial_bar.grid(row=6, column=0, sticky="ew", pady=(4, 12))

        perf_frame = ttk.Frame(main)
        perf_frame.grid(row=7, column=0, sticky="ew")
        perf_frame.columnconfigure(0, weight=1)
        self.perf_frame = perf_frame

        self.self_label = ttk.Label(perf_frame, text="Your performance", font=("Segoe UI", 10, "bold"))
        self.self_label.grid(row=0, column=0, sticky="w")

        self.self_perf_bar = CanvasBar(perf_frame, height=18, fill="#cc2b2b")  # red
        self.self_perf_bar.grid(row=1, column=0, sticky="ew", pady=(4, 10))

        self.cohort_label = ttk.Label(perf_frame, text="Cohort average", font=("Segoe UI", 10, "bold"))
        self.cohort_label.grid(row=2, column=0, sticky="w")

        self.exp_perf_bar = CanvasBar(perf_frame, height=18, fill="#2aa84a")  # green
        self.exp_perf_bar.grid(row=3, column=0, sticky="ew", pady=(4, 0))

        self.monitor_label = ttk.Label(main, text="", font=("Segoe UI", 9))
        self.monitor_label.grid(row=8, column=0, sticky="w", pady=(10, 0))

        ttk.Frame(main).grid(row=9, column=0, sticky="nsew")

        self._set_interactive(False)
        self._set_bars_visible(False)

    def _on_submit_key(self, event):
        self.on_submit()
        return "break"

    def _set_interactive(self, enabled: bool):
        state = "normal" if enabled else "disabled"
        self.answer_entry.configure(state=state)
        self.submit_btn.configure(state=state)

    def _set_bars_visible(self, visible: bool):
        if visible and not self._bars_visible:
            self.trial_label.grid()
            self.trial_bar.grid()
            self.perf_frame.grid()
            self.monitor_label.configure(text="Performance is monitored.")
            self._bars_visible = True
        elif (not visible) and self._bars_visible:
            self.trial_label.grid_remove()
            self.trial_bar.grid_remove()
            self.perf_frame.grid_remove()
            self.monitor_label.configure(text="")
            self._bars_visible = False

        if not visible:
            self.trial_bar.set_percent(0)
            self.self_perf_bar.set_percent(0)
            self.exp_perf_bar.set_percent(EXPECTED_AVG * 100)

    # ---------------- Logging ----------------
    def _log_event(self, event: str, extra=None):
        row = {
            "t_epoch": now_epoch(),
            "t_from_start_s": now_epoch() - self.session_start,
            "event": event,
            "phase": self.phase,
            "difficulty": self.difficulty,
            "time_limit_s": self.time_limit,
            "trials_total": self.trials_total,
            "trials_correct": self.trials_correct,
        }
        if extra:
            row.update(extra)
        self.log_rows.append(row)

    # ---------------- Session flow ----------------
    def _set_idle_screen(self):
        self.phase = "idle"
        self.phase_end_epoch = None
        self.phase_label.configure(text="Phase: Idle")
        self.timer_label.configure(text="Time Remaining: --:--")
        self.display_label.configure(text="Ready")
        self.feedback_label.configure(text="Click 'Start Session' when participant is seated and ECG recording is running.")
        self._set_interactive(False)
        self._set_bars_visible(False)

    def start_session(self):
        self.start_btn.configure(state="disabled")
        self._cancel_all_timers()
        self._log_event("session_begin_clicked")
        self.start_practice()

    def start_practice(self):
        self.phase = "practice"
        self.phase_end_epoch = now_epoch() + PRACTICE_SEC
        self.difficulty = MIN_DIFFICULTY

        self.phase_label.configure(text="Phase: PRACTICE (30s)")
        self.feedback_label.configure(text="Familiarisation only. Answer normally (no time limit).")
        self._set_interactive(True)
        self._set_bars_visible(False)
        self._log_event("phase_start", {"phase_name": "practice"})
        self._next_trial_practice()
        self._tick_phase()

    def start_stress(self):
        self.phase = "stress"
        self.phase_end_epoch = now_epoch() + STRESS_SEC

        self.difficulty = START_DIFFICULTY

        self.phase_label.configure(text="Phase: ARITHMETIC (10min)")
        self.feedback_label.configure(text="Answer as quickly and accurately as possible.")
        self._set_interactive(True)
        self._set_bars_visible(True)

        self._perf_armed = False
        self._perf_first_set = False
        self.displayed_perf = 0.50
        self.self_perf_bar.set_percent(0)

        self.exp_perf_bar.set_percent(EXPECTED_AVG * 100)

        self.time_limit = TL_DEFAULT
        self._log_event("phase_start", {"phase_name": "stress", "initial_time_limit_s": self.time_limit})

        self._next_trial_stress()
        self._tick_phase()

    def _tick_phase(self):
        if self.phase_end_epoch is None:
            return

        remaining = self.phase_end_epoch - now_epoch()
        self.timer_label.configure(text=f"Time Remaining: {fmt_mmss(remaining)}")

        if remaining <= 0:
            self._log_event("phase_end", {"phase_name": self.phase})
            if self.phase == "practice":
                self.start_stress()
                return
            if self.phase == "stress":
                self.end_session()
                return

        self._phase_tick_id = self.after(200, self._tick_phase)

    def end_session(self):
        self._cancel_all_timers()
        self.phase = "ended"
        self.phase_end_epoch = None
        self.phase_label.configure(text="Phase: END")
        self.timer_label.configure(text="Time Remaining: --:--")
        self.display_label.configure(text="Done")
        self._set_interactive(False)
        self._set_bars_visible(False)

        path = self._save_logs()
        acc = (self.trials_correct / self.trials_total) if self.trials_total else 0.0
        self.feedback_label.configure(text=f"Session complete. Accuracy: {acc:.1%}\nLog saved: {path}")
        self._log_event("session_end", {"log_path": path, "true_accuracy": acc})

        messagebox.showinfo("MIST", f"Session complete.\n\nAccuracy: {acc:.1%}\n\nLog saved:\n{path}")

    # ---------------- Timers cancellation ----------------
    def _cancel_all_timers(self):
        if self._phase_tick_id is not None:
            try:
                self.after_cancel(self._phase_tick_id)
            except Exception:
                pass
            self._phase_tick_id = None

        if self._trial_bar_tick_id is not None:
            try:
                self.after_cancel(self._trial_bar_tick_id)
            except Exception:
                pass
            self._trial_bar_tick_id = None

        if self._stress_timeout_id is not None:
            try:
                self.after_cancel(self._stress_timeout_id)
            except Exception:
                pass
            self._stress_timeout_id = None

    # ---------------- PRACTICE trials ----------------
    def _next_trial_practice(self):
        if self.phase != "practice":
            return

        self.difficulty = MIN_DIFFICULTY
        self.answer_var.set("")
        self.answer_entry.focus_set()

        q, ans = generate_problem(self.difficulty)
        self.current_question = q
        self.current_answer = ans
        self.trial_start_epoch = now_epoch()

        self.display_label.configure(text=q)
        self._log_event("trial_start", {"condition": "practice", "question": q, "answer_true": ans})

    # ---------------- STRESS trials ----------------
    def _next_trial_stress(self):
        if self.phase != "stress":
            return
        self.after(int(ITI_SEC * 1000), self._start_stress_trial)

    def _start_stress_trial(self):
        if self.phase != "stress":
            return

        self.answer_var.set("")
        self.answer_entry.focus_set()

        q, ans = generate_problem(self.difficulty)
        self.current_question = q
        self.current_answer = ans
        self.trial_start_epoch = now_epoch()

        self.display_label.configure(text=q)
        self._log_event("trial_start", {"condition": "stress", "question": q, "answer_true": ans, "allowed_time_s": self.time_limit})
        self._start_enforced_timer(self.time_limit)

    def _start_enforced_timer(self, seconds_allowed: float):
        if self._trial_bar_tick_id is not None:
            try:
                self.after_cancel(self._trial_bar_tick_id)
            except Exception:
                pass
            self._trial_bar_tick_id = None

        if self._stress_timeout_id is not None:
            try:
                self.after_cancel(self._stress_timeout_id)
            except Exception:
                pass
            self._stress_timeout_id = None

        self._trial_timer_end = now_epoch() + seconds_allowed
        self.trial_bar.set_percent(100)
        self._update_trial_bar()
        self._stress_timeout_id = self.after(int(seconds_allowed * 1000), self.on_timeout)

    def _update_trial_bar(self):
        if self.phase != "stress":
            return

        remaining = self._trial_timer_end - now_epoch()
        total = max(0.001, self.time_limit)
        frac = clamp(remaining / total, 0.0, 1.0)
        self.trial_bar.set_percent(frac * 100)

        if remaining > 0:
            self._trial_bar_tick_id = self.after(50, self._update_trial_bar)
        else:
            self._trial_bar_tick_id = None

    # ---------------- Submit / Timeout ----------------
    def on_submit(self):
        if self.phase not in ("practice", "stress"):
            return

        raw = self.answer_var.get().strip()
        rt = now_epoch() - self.trial_start_epoch if self.trial_start_epoch else None

        if self.phase == "stress" and self._stress_timeout_id is not None:
            try:
                self.after_cancel(self._stress_timeout_id)
            except Exception:
                pass
            self._stress_timeout_id = None

        try:
            resp = int(raw)
            parse_ok = True
        except Exception:
            resp = None
            parse_ok = False

        correct = (parse_ok and resp == self.current_answer)
        outcome = "correct" if correct else "incorrect"

        if self.phase == "practice":
            self._log_trial_end("practice", rt, raw, correct, outcome)
            self.feedback_label.configure(text=f"(Practice) {outcome.capitalize()}")
            self.after(250, self._next_trial_practice)
            return

        # STRESS:
        self._apply_stress_scoring(correct=correct)
        self._log_trial_end("stress", rt, raw, correct, outcome)
        self.feedback_label.configure(text=outcome.capitalize())

        # arm after first real submitted answer
        if (not self._perf_armed) and (raw != ""):
            self._perf_armed = True

        # update red bar with directionality
        self._update_performance_bars(last_correct=correct, first_submit=(raw != ""))

        self._adapt_stress()
        self._next_trial_stress()

    def on_timeout(self):
        if self.phase != "stress":
            return
        self._stress_timeout_id = None

        rt = self.time_limit
        outcome = "timeout"
        correct = False

        self._apply_stress_scoring(correct=False)
        self._log_trial_end("stress", rt, "", correct, outcome)

        self.feedback_label.configure(text="Timeout")

        # timeout counts as wrong
        self._update_performance_bars(last_correct=False, first_submit=False)

        self._adapt_stress()
        self._next_trial_stress()

    # ---------------- Scoring / Adaptation ----------------
    def _apply_stress_scoring(self, correct: bool):
        self.trials_total += 1
        if correct:
            self.trials_correct += 1
            self.streak_correct += 1
            self.streak_bad = 0
        else:
            self.streak_correct = 0
            self.streak_bad += 1

    def _adapt_stress(self):
        if self.streak_correct >= STREAK_UP:
            if self.difficulty < MAX_DIFFICULTY:
                self.difficulty += 1
            else:
                self.time_limit = clamp(self.time_limit * 0.90, TL_MIN, TL_MAX)
            self.streak_correct = 0
            self._log_event("adapt_up", {"new_difficulty": self.difficulty, "new_time_limit_s": self.time_limit})

        # 3 incorrect/timeouts in a row: DO BOTH
        if self.streak_bad >= STREAK_DOWN:
            old_diff = self.difficulty
            old_tl = self.time_limit

            if self.difficulty > MIN_DIFFICULTY:
                self.difficulty -= 1

            if self.time_limit < TL_MAX:
                self.time_limit = clamp(self.time_limit * 1.10, TL_MIN, TL_MAX)

            self.streak_bad = 0
            self._log_event("adapt_down", {
                "old_difficulty": old_diff,
                "old_time_limit_s": old_tl,
                "new_difficulty": self.difficulty,
                "new_time_limit_s": self.time_limit,
                "note": "decrease_difficulty_and_increase_time",
            })

    # ---------------- Performance bars ----------------
    def _update_performance_bars(self, last_correct: bool, first_submit: bool):
        # Green bar fixed
        self.exp_perf_bar.set_percent(EXPECTED_AVG * 100)

        if self.phase != "stress":
            self.self_perf_bar.set_percent(0)
            return

        # Keep red bar empty until first submitted answer
        if not self._perf_armed:
            self.self_perf_bar.set_percent(0)
            return

        # First time we show it: set to believable fixed % (e.g., 53%)
        if not self._perf_first_set:
            self.displayed_perf = clamp(FIRST_PERF_PCT, DECEPTIVE_MIN, DECEPTIVE_MAX)
            self._perf_first_set = True
        else:
            jitter = random.uniform(-PERF_JITTER, PERF_JITTER)
            if last_correct:
                self.displayed_perf += (PERF_STEP_RIGHT + jitter)
            else:
                self.displayed_perf -= (PERF_STEP_WRONG - jitter)

            self.displayed_perf = clamp(self.displayed_perf, DECEPTIVE_MIN, DECEPTIVE_MAX)

        self.self_perf_bar.set_percent(self.displayed_perf * 100)

        true_perf = (self.trials_correct / self.trials_total) if self.trials_total else 0.0
        self._log_event("display_update", {"true_accuracy": true_perf, "displayed_perf": self.displayed_perf, "last_correct": int(last_correct)})

    # ---------------- Logging helpers ----------------
    def _log_trial_end(self, condition: str, rt, response: str, correct: bool, outcome: str):
        self._log_event("trial_end", {
            "condition": condition,
            "question": self.current_question,
            "answer_true": self.current_answer,
            "response_raw": response,
            "rt_s": rt,
            "outcome": outcome,
            "correct": int(correct),
            "allowed_time_s": self.time_limit if condition == "stress" else "",
        })

    def _save_logs(self) -> str:
        dt = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"mist_log_{dt}.csv"
        os.makedirs(LOG_DIR, exist_ok=True)
        out_path = os.path.abspath(os.path.join(LOG_DIR, filename))

        keys = set()
        for r in self.log_rows:
            keys.update(r.keys())
        fieldnames = sorted(keys)

        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in self.log_rows:
                writer.writerow(r)

        return out_path

    def on_quit(self):
        if self.phase not in ("idle", "ended"):
            if not messagebox.askyesno("Quit", "Session is running. Quit anyway?"):
                return
        self._log_event("session_quit")
        try:
            self._save_logs()
        except Exception:
            pass
        self.destroy()


def main():
    random.seed()
    app = MISTApp()
    app.mainloop()


if __name__ == "__main__":
    main()
