"""Ball impulse events = candidate touches/bounces, from the ball track."""
import numpy as np, pandas as pd
from scipy.signal import savgol_filter, find_peaks
def detect_events(track_csv, fps=59.94, min_dv=5.0, min_turn=25):
    d = pd.read_csv(track_csv)
    seg = np.r_[0, (np.diff(d.frame) > 3).cumsum()]
    ev = []
    for g, s in d.groupby(seg):
        if len(s) < 15: continue
        u = savgol_filter(s.u.values, 7, 2); v = savgol_filter(s.v.values, 7, 2)
        vu, vv = np.gradient(u), np.gradient(v)
        # velocity change across a +-4-frame window (impulse), gravity removed as the segment median
        L = 4
        dvu = np.r_[np.zeros(L), vu[2*L:] - vu[:-2*L], np.zeros(L)]
        dvv = np.r_[np.zeros(L), vv[2*L:] - vv[:-2*L], np.zeros(L)]
        dvv = dvv - np.median(dvv)
        mag = np.hypot(dvu, dvv)
        pk, _ = find_peaks(mag, height=min_dv, distance=12)
        for p in pk:
            a, b = max(0, p - L), min(len(s) - 1, p + L)
            v1 = np.array([vu[a], vv[a]]); v2 = np.array([vu[b], vv[b]])
            ang = np.degrees(np.arccos(np.clip(v1 @ v2 / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6), -1, 1)))
            sp = np.linalg.norm(v2) / (np.linalg.norm(v1) + 1e-6)
            if ang < min_turn and 0.6 < sp < 1.6: continue      # smooth flight, no contact
            ev.append(dict(frame=int(s.frame.iloc[p]), t_s=round(s.frame.iloc[p] / fps, 3),
                           u=float(u[p]), v=float(v[p]), size_px=float(s.size_px.iloc[p]),
                           dv=round(float(mag[p]), 1), turn_deg=round(float(ang), 0), speed_ratio=round(float(sp), 2)))
        # first sighting of a segment that starts mid-air is also a (serve / unseen) contact
    return pd.DataFrame(ev)
if __name__ == "__main__":
    e = detect_events("ball_track.csv"); print(e.to_string()); e.to_csv("ball_events.csv", index=False)
