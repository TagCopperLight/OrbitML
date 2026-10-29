#!/usr/bin/env python3
"""
Scenes illustrating the steps of orbitml.ipynb, streamed to the Aurora viewer (viz/viewer).
The data comes from the notebook's last cell, viz/data/orbitml.npz. See viewer/src/main.cpp for the messages.

Usage: scenes.py <scene> [--tle N] [--at DAYS]
  1    the truth: the orbit in TEME and its ground track, the Earth turning under it
  3    the R, S, W frame on the satellite
  8    every test TLE seen from its true position at the same Δt: the cloud stretches along S
  9    the position at each TLE date: always at the ascending node
  13   the plane of the orbit over 60 days, seen with the Sun kept to the right
  14   the error left after step 13 in R, averaged by place: it depends on the longitude
  fin  like 8, with the corrected positions
  replay  the first viewer of OrbitML: the truth played back, with a fading ground track
Keys: SPACE pause, RIGHT step, UP/DOWN speed, R restart, N/P next/previous TLE (or week, for 9)
"""

import argparse
import time
from pathlib import Path

import numpy as np
from aurora_client import AuroraClient, AuroraDisconnectedError, AuroraUnavailableError, Custom
from sgp4.api import Satrec

import coastlines

DATA = Path(__file__).parents[1] / "data" / "orbitml.npz"
FPS = 60
EARTH_RADIUS = 6378.137  # km
DAY = 86400.0  # s


def gmst(jd: np.ndarray | float) -> np.ndarray:
    """Greenwich mean sidereal time in rad, the same formula as step 14."""
    return np.radians(280.46061837 + 360.98564736629 * (np.asarray(jd) - 2451545.0))


def earth_rotation(jd: float) -> list[float]:
    """Earth fixed to TEME, row by row: a rotation of the sidereal time around z."""
    c, s = np.cos(gmst(jd)), np.sin(gmst(jd))
    return [c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0]


def latitude_longitude(r: np.ndarray, jd: np.ndarray) -> np.ndarray:
    """(N, 2) geocentric latitude and longitude in degrees of TEME positions (N, 3) at the dates jd."""
    longitude = np.degrees(np.arctan2(r[:, 1], r[:, 0]) - gmst(jd))
    latitude = np.degrees(np.arcsin(r[:, 2] / np.linalg.norm(r, axis=1)))
    return np.column_stack([latitude, (longitude + 180) % 360 - 180])


def rsw(r: np.ndarray, v: np.ndarray) -> np.ndarray:
    """(..., 3, 3) rows R, S, W of the frame of each position and velocity."""
    R = r / np.linalg.norm(r, axis=-1, keepdims=True)
    W = np.cross(r, v)
    W /= np.linalg.norm(W, axis=-1, keepdims=True)
    return np.stack([R, np.cross(W, R), W], axis=-2)


def camera_facing(normal: np.ndarray, tilt: float) -> list[float]:
    """Looks at the plane of `normal` from `tilt` degrees toward the north pole, with the north pole up."""
    normal = normal / np.linalg.norm(normal)
    north = np.array([0.0, 0.0, 1.0])
    toward_north = north - north @ normal * normal
    toward_north /= np.linalg.norm(toward_north)
    back = np.cos(np.radians(tilt)) * normal + np.sin(np.radians(tilt)) * toward_north
    up = north - north @ back * back
    up /= np.linalg.norm(up)
    return [*np.cross(up, back), *up, *back]


def rounded(points: np.ndarray, decimals: int = 1) -> list[float]:
    return np.round(np.asarray(points, dtype=float), decimals).ravel().tolist()


class Data:
    def __init__(self) -> None:
        if not DATA.exists():
            raise SystemExit(f"{DATA} is missing, run the last cell of orbitml.ipynb")
        data = np.load(DATA)
        self.truth_jd, self.truth_r = data["truth_jd"], data["truth_r"]
        self.tle = data["tle"]
        self.test = data["test"]
        self.dt = data["dt"]
        self.error = data["error"]
        self.sun = data["sun"]
        self.ground = data["ground"]
        self.satellites = [Satrec.twoline2rv(line1, line2) for line1, line2 in self.tle]
        self.jd = np.array([sat.jdsatepoch + sat.jdsatepochF for sat in self.satellites])

    def propagate(self, n: int, jd: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        sat = self.satellites[n]
        jd = np.atleast_1d(jd)
        _, r, v = sat.sgp4_array(np.full(len(jd), sat.jdsatepoch), jd - sat.jdsatepoch)
        return r, v

    def period(self, n: int) -> float:
        """Orbital period in days, from the mean motion in rad/min."""
        return 2 * np.pi / self.satellites[n].no_kozai / 1440


class Scene:
    """Plays `duration` days of simulated time, looping. Subclasses draw in setup() and frame()."""

    duration = 1.0  # days
    loop = True  # otherwise the scene stops on its last image
    speeds = [60, 300, 1200, 4800]  # simulated seconds per real second
    speed = 1  # index in speeds

    def __init__(self, data: Data, aurora: AuroraClient, tle: int) -> None:
        self.data, self.aurora, self.tle = data, aurora, tle
        self.t = 0.0  # days since the start of the scene
        self.paused = False

    def send(self, **payload) -> None:
        self.aurora.send(Custom({key: value.item() if isinstance(value, np.generic) else value for key, value in payload.items()}))

    def start(self) -> None:
        self.send(clear=1)
        self.aurora.send(*(Custom({"coastline": rounded(line, 2)}) for line in coastlines.load()))
        self.setup()
        self.frame()

    def setup(self) -> None:
        pass

    def frame(self) -> None:
        pass

    def watches(self, values: dict, section: str) -> None:
        self.aurora.watches(values, section=section)

    def command(self, name: str) -> None:
        match name:
            case "pause":
                self.paused = not self.paused
            case "step":
                self.paused = True
                self.t = (self.t + self.speeds[self.speed] / FPS / DAY) % self.duration
            case "faster":
                self.speed = min(self.speed + 1, len(self.speeds) - 1)
            case "slower":
                self.speed = max(self.speed - 1, 0)
            case "restart":
                self.t = 0.0
                self.start()
            case "next" | "previous":
                self.tle = self.next_tle(1 if name == "next" else -1)
                self.t = 0.0
                self.start()

    def next_tle(self, step: int) -> int:
        """Moves through the test TLEs whose prediction has truth up to 14 days."""
        full = np.flatnonzero(self.data.test & np.isfinite(self.data.error[:, 0, -1]))
        position = np.searchsorted(full, self.tle) + step
        return int(full[position % len(full)])

    def advance(self, seconds: float) -> None:
        if not self.paused:
            self.t += self.speeds[self.speed] * seconds / DAY
            self.t = self.t % self.duration if self.loop else min(self.t, self.duration)
        self.watches({"ETAT": "PAUSE" if self.paused else "LECTURE", "VITESSE": f"{self.speeds[self.speed]}x"}, "LECTURE")


class Truth(Scene):
    """Step 1: the orbit is fixed in TEME, the Earth turns under it, so each revolution passes further west."""

    duration = 2.0
    speeds = [60, 300, 1200, 4800]

    def setup(self) -> None:
        d = self.data
        self.send(layout="both")
        self.send(titles=1, space="ORBITE, REPERE TEME", map="TRACE AU SOL")
        v = np.gradient(d.truth_r[:3], axis=0)
        self.send(camera=camera_facing(np.cross(d.truth_r[1], v[1]), -45), center=[0, 0, 0], extent=8000)
        self.period = 2 * np.pi * np.sqrt(np.mean(np.linalg.norm(d.truth_r, axis=1)) ** 3 / 398600.4418) / DAY

    def at(self, jd: np.ndarray) -> np.ndarray:
        d = self.data
        return np.stack([np.interp(jd, d.truth_jd, column) for column in d.truth_r.T], axis=-1)

    def frame(self) -> None:
        d = self.data
        jd = d.truth_jd[0] + self.t
        self.send(earth=earth_rotation(jd), surface=1)
        orbit = self.at(np.linspace(jd - self.period, jd, 120))
        self.send(id="orbit", line=rounded(orbit), color="blue", size=2)
        self.send(id="satellite", point=rounded(orbit[-1]), color="blue", size=5)

        # Three revolutions of ground track, so that the shift to the west shows
        times = np.linspace(max(jd - 3 * self.period, d.truth_jd[0]), jd, 600)
        track = latitude_longitude(self.at(times), times)
        self.send(id="track", track=rounded(track, 2), color="blue", size=2)

        latitude, longitude = track[-1]
        revolutions = int(self.t / self.period) + 1
        self.watches({
            "DATE": julian_iso(jd),
            "REVOLUTION": revolutions,
            "ALTITUDE": f"{np.linalg.norm(orbit[-1]) - EARTH_RADIUS:.1f} km",
            "LATITUDE": f"{latitude:+.2f} deg",
            "LONGITUDE": f"{longitude:+.2f} deg",
        }, "SENTINEL-6A")


class Frame(Truth):
    """Step 3: R points up, S along the motion, W out of the plane of the orbit."""

    duration = 2.0
    speeds = [30, 60, 300, 1200]
    ARROW = 3500.0  # km

    def setup(self) -> None:
        super().setup()
        self.send(layout="space")
        self.send(titles=1, space="REPERE R, S, W", map="")
        self.watches({"R": "RADIALE", "S": "LE LONG DE LA TRAJECTOIRE", "W": "PERPENDICULAIRE AU PLAN"}, "AXES")

    def frame(self) -> None:
        d = self.data
        jd = d.truth_jd[0] + self.t
        self.send(earth=earth_rotation(jd), surface=1)
        orbit = self.at(np.linspace(jd - self.period, jd + 0.01 * self.period, 121))
        self.send(id="orbit", line=rounded(orbit), color="gray", size=2)
        r = self.at(np.array([jd]))[0]
        v = (self.at(np.array([jd + 1e-4]))[0] - r) / 1e-4
        R, S, W = rsw(r, v)
        # Seen along the diagonal of the three axes, so that they are all shortened the same way
        back = R - S + W
        back /= np.linalg.norm(back)
        up = W - W @ back * back
        up /= np.linalg.norm(up)
        self.send(camera=[*np.cross(up, back), *up, *back], center=[0, 0, 0], extent=8500)
        self.send(id="satellite", point=rounded(r), color="white", size=6)
        for axis, name, color in zip((R, S, W), "RSW", ("red", "orange", "green")):
            tip = r + self.ARROW * axis
            side = np.cross(axis, rsw(r, v)[2] if name != "W" else rsw(r, v)[0])
            head = [tip - 250 * axis + 120 * side, tip, tip - 250 * axis - 120 * side]
            self.send(id=f"{name}", line=rounded([r, tip]), color=color, size=3)
            self.send(id=f"{name} head", line=rounded(head), color=color, size=3)
            self.send(id=f"{name} label", point=rounded(tip), color=color, size=1, label=name)


class Drift(Scene):
    """Step 8: every test TLE seen from its true position at the same Δt, the cloud stretches along S."""

    loop = False
    duration = 14.0
    speeds = [4800, 20000, 80000]
    speed = 1
    correction = False

    def setup(self) -> None:
        d = self.data
        full = d.test & np.isfinite(d.error[:, 0, -1])  # test TLEs with truth up to 14 days
        error = d.error[full].astype(float) * 1e3  # (TLE, 3, Δt) m
        # Where each point is, seen from the truth: SGP4 at -e, the corrected position at -(e - ê)
        self.offsets = {"SGP4": -error}
        if self.correction:
            self.offsets["CORRIGE"] = -(error - d.ground[full].astype(float) * 1e3)

        self.send(layout="space")
        self.send(titles=1, space="CHAQUE POINT : UN TLE DE TEST VU DEPUIS SA VRAIE POSITION, TOUS AU MEME DT", map="")
        extent = 1.05 * np.nanmax(np.abs(error[:, 1]))
        self.send(camera=[1, 0, 0, 0, 1, 0, 0, 0, 1], center=[0, 0, 0], extent=extent)
        axis = 0.95 * extent
        self.send(id="S axis", line=[-axis, 0, 0, axis, 0, 0], color="gray", size=1, label="S")
        self.send(id="R axis", line=[0, -axis, 0, 0, axis, 0], color="gray", size=1, label="R")
        scale = 10 ** np.floor(np.log10(extent))  # m
        self.send(id="scale", line=[-axis, -axis, 0, -axis + scale, -axis, 0], color="white", size=2,
                  label=f"{scale / 1e3:g} KM")
        self.send(id="truth", point=[0, 0, 0], color="blue", size=7)
        self.send(id="truth label", point=rounded([-0.13 * extent, 0.06 * extent, 0]), color="blue", size=1, label="VERITE")
        self.watches({"TLE DE TEST": int(full.sum())}, "NUAGE")

    def frame(self) -> None:
        d = self.data
        days = min(max(self.t, d.dt[0]), d.dt[-1])
        values = {"DT": f"{days:.2f} j"}
        for (name, offset), color in zip(self.offsets.items(), ("orange", "green")):
            # (TLE, 3) at this Δt, every TLE at the same place of its orbit
            here = np.stack([smooth(d.dt, offset[:, k].T, days) for k in range(3)], axis=-1)
            dots = np.column_stack([here[:, 1], here[:, 0], np.zeros(len(here))])
            self.send(id=name, dots=rounded(dots), color=color, size=4)
            rms = np.sqrt(np.mean(here**2, axis=0))
            values |= {f"RMS {axis} {name}": f"{rms[k]:.0f} m" for k, axis in enumerate("RSW")}
        self.watches(values, "ECART A LA VERITE")


class Corrected(Drift):
    """The end of the notebook: the same as step 8, with the network of step 14."""

    correction = True


class Node(Scene):
    """Step 9: every TLE is dated at the ascending node, where the satellite crosses the equator northward."""

    loop = False
    duration = 7.0
    speeds = [600, 2400, 9600, 38400]
    speed = 2

    def setup(self) -> None:
        d, n = self.data, self.tle
        week = (d.jd >= d.jd[n]) & (d.jd < d.jd[n] + 7)
        self.week = np.flatnonzero(week)
        self.send(layout="both")
        self.send(titles=1, space="POSITION A LA DATE DE CHAQUE TLE, EN GRIS LE PLAN DE L'EQUATEUR", map="LES MEMES, SUR LA CARTE")
        r, v = d.propagate(n, d.jd[n])
        self.send(camera=camera_facing(np.cross(r[0], v[0]), -30), center=[0, 0, 0], extent=8000)
        # The plane of the equator at the height of the orbit, drawn from the node so that its label ends up opposite
        node = np.arctan2(r[0, 1], r[0, 0])
        angles = node + np.pi + np.linspace(0.05, 2 * np.pi - 0.05, 121)
        equator = np.linalg.norm(r[0]) * np.column_stack([np.cos(angles), np.sin(angles), np.zeros(121)])
        self.send(id="equator", line=rounded(equator), color="gray", size=1)
        self.send(id="node label", point=rounded(r[0] * 1.08), color="orange", size=1, label="NOEUD ASCENDANT")
        self.watches({"DU": julian_iso(d.jd[n]), "AU": julian_iso(d.jd[n] + 7), "TLE": len(self.week)}, "SEMAINE")

    def frame(self) -> None:
        d, n = self.data, self.tle
        jd = d.jd[n] + self.t
        self.send(earth=earth_rotation(jd), surface=1)
        last = self.week[np.searchsorted(d.jd[self.week], jd, side="right") - 1]
        period = d.period(last)
        r, _ = d.propagate(last, np.linspace(jd - period, jd, 120))
        self.send(id="orbit", line=rounded(r), color="blue", size=2)
        self.send(id="satellite", point=rounded(r[-1]), color="blue", size=5)
        times = np.linspace(max(jd - period, d.jd[n]), jd, 200)
        self.send(id="track", track=rounded(latitude_longitude(d.propagate(last, times)[0], times), 2), color="blue", size=2)

        # Each TLE published so far, at its own date
        published = self.week[d.jd[self.week] <= jd]
        for m in published:
            at, _ = d.propagate(m, d.jd[m])
            self.send(id=f"tle {m}", point=rounded(at[0]), color="orange", size=5)
        epochs = np.array([d.propagate(m, d.jd[m])[0][0] for m in published])
        where = latitude_longitude(epochs, d.jd[published])
        # Spots are sent again only when a TLE is added
        if len(published) != getattr(self, "count", -1):
            self.count = len(published)
            self.send(id="tle spots", spots=rounded(where, 2), size=5, colors=[0.96, 0.73, 0.38] * len(published))
        self.watches({"DATE": julian_iso(jd), "TLE PUBLIES": len(published), "LATITUDE DU DERNIER": f"{where[-1, 0]:+.3f} deg"}, "SATELLITE")

    def next_tle(self, step: int) -> int:
        """A week later or earlier."""
        return int(np.clip(np.searchsorted(self.data.jd, self.data.jd[self.tle] + 7 * step), 0, len(self.data.jd) - 1))


class Sun(Scene):
    """Step 13: seen from above the north pole, with the Sun kept to the right, the plane of the orbit turns
    half a turn in 60 days, and is then back in the same position with respect to the Sun."""

    loop = False
    duration = 60.0
    speeds = [86400, 4 * 86400, 16 * 86400]
    speed = 1
    RADIUS = 9500.0  # km, half length of the line of the orbit plane
    STEP = 10  # days between the lines left behind

    def setup(self) -> None:
        from astropy.coordinates import get_body
        from astropy.time import Time

        d, n = self.data, self.tle
        self.days = np.arange(0, self.duration + 1, 0.25)
        sun_ra = np.unwrap(get_body("sun", Time(d.jd[n] + self.days, format="jd")).ra.rad)
        node = np.interp(d.jd[n] + self.days, d.jd, np.unwrap([sat.nodeo for sat in d.satellites]))
        self.angle = node - sun_ra  # Ω - α, the direction of the node when the Sun is along +x
        self.send(layout="space")
        self.send(titles=1, space="VU DU POLE NORD, LE SOLEIL TOUJOURS A DROITE : LE PLAN DE L'ORBITE VU PAR LA TRANCHE", map="")
        self.send(camera=[1, 0, 0, 0, 1, 0, 0, 0, 1], center=[0, 0, 0], extent=11000)
        self.send(earth=[1, 0, 0, 0, 1, 0, 0, 0, 1], surface=0)
        self.send(id="sun ray", line=[EARTH_RADIUS, 0, 0, 10300, 0, 0], color="orange", size=3)
        self.send(id="sun", point=[10300, 0, 0], color="orange", size=12, label="SOLEIL")
        self.watches({"DEBUT": julian_iso(d.jd[n])}, "ANGLES")

    def plane(self, angle: float) -> list[float]:
        """The orbit plane seen edge on: a line through the Earth, from the descending to the ascending node."""
        u = np.array([np.cos(angle), np.sin(angle), 0.0])
        return rounded([-self.RADIUS * u, self.RADIUS * u])

    def frame(self) -> None:
        d, n = self.data, self.tle
        angle = np.interp(self.t, self.days, self.angle)
        for day in range(0, int(self.t) + 1, self.STEP):
            if day + self.STEP > self.t and day != 0:
                continue  # the current line is drawn on top
            past = np.interp(day, self.days, self.angle)
            self.send(id=f"day {day}", line=self.plane(past), color="gray", size=1, label=f"J{day}")
        self.send(id="plane", line=self.plane(angle), color="blue", size=3, label=f"J{self.t:.0f}")
        self.watches({
            "DATE": julian_iso(d.jd[n] + self.t),
            "JOURS": f"{self.t:.0f}",
            "NOEUD - SOLEIL": f"{np.degrees(angle) % 360:.0f} deg",
            "SIN 2(NOEUD - SOLEIL)": f"{np.sin(2 * angle):+.2f}",
        }, "ANGLES")

    def next_tle(self, step: int) -> int:
        """A month later or earlier."""
        return int(np.clip(np.searchsorted(self.data.jd, self.data.jd[self.tle] + 30 * step), 0, len(self.data.jd) - 1))


class Ground(Scene):
    """Step 14: the error left after step 13, averaged by place over every test TLE, depends on the longitude."""

    loop = False
    duration = 14.0
    speeds = [4800, 20000, 80000]
    speed = 1
    CELL = 3.0  # degrees
    TRAIL = 0.1  # days of ground track behind the satellite

    def setup(self) -> None:
        d, n = self.data, self.tle
        self.send(layout="map")
        self.send(titles=1, space="", map="ERREUR RADIALE QUI RESTE APRES L'ETAPE 13, MOYENNE PAR CASE DE 3 DEG")

        # Every test TLE, every 10 min: where the satellite is, and the error left in R
        where, left = [], []
        for m in np.flatnonzero(d.test):
            jd = d.jd[m] + d.dt
            known = np.isfinite(d.error[m, 0])
            where.append(latitude_longitude(d.propagate(m, jd[known])[0], jd[known]))
            left.append((d.error[m, 0, known] - d.sun[m, 0, known]) * 1e3)  # m
        where, left = np.concatenate(where), np.concatenate(left)

        rows = np.floor((where[:, 0] + 90) / self.CELL).astype(int)
        columns = np.floor((where[:, 1] + 180) / self.CELL).astype(int)
        cell = rows * 1000 + columns
        cells, inverse = np.unique(cell, return_inverse=True)
        mean = np.bincount(inverse, left) / np.bincount(inverse)
        centers = np.column_stack([(cells // 1000 + 0.5) * self.CELL - 90, (cells % 1000 + 0.5) * self.CELL - 180])
        limit = np.percentile(np.abs(mean), 95)
        self.send(id="cells", spots=rounded(centers, 2), colors=np.round(diverging(mean / limit), 3).ravel().tolist(), size=6.5)

        self.watches({"TLE DE TEST": int(d.test.sum()), "POINTS": len(left)}, "MOYENNE")
        self.watches({"BLEU": f"-{limit:.0f} m", "GRIS": "0 m", "ORANGE": f"+{limit:.0f} m"}, "ECHELLE")
        self.watches({"DATE": julian_iso(d.jd[n]), "NUMERO": f"{n}"}, "TLE")

    def frame(self) -> None:
        d, n = self.data, self.tle
        times = d.jd[n] + np.linspace(max(self.t - self.TRAIL, 0), self.t, 120)
        track = latitude_longitude(d.propagate(n, times)[0], times)
        self.send(id="track", track=rounded(track, 2), color="white", size=2)
        self.watches({"DT": f"{self.t:.2f} j", "LONGITUDE": f"{track[-1, 1]:+.1f} deg"}, "TLE")


def diverging(x: np.ndarray) -> np.ndarray:
    """(N, 3) colors from the theme's blue (x = -1) through gray (0) to orange (+1)."""
    blue, gray, orange = np.array([0.45, 0.75, 0.98]), np.array([0.25, 0.26, 0.32]), np.array([0.96, 0.73, 0.38])
    x = np.clip(np.nan_to_num(x), -1, 1)[:, None]
    return np.where(x < 0, gray + (blue - gray) * -x, gray + (orange - gray) * x)


def smooth(x: np.ndarray, y: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Catmull-Rom interpolation of y at t, from the values at the evenly spaced x: a curve, not a polygon."""
    i = np.clip(np.searchsorted(x, t) - 1, 0, len(x) - 2)
    u = np.clip((t - x[i]) / (x[i + 1] - x[i]), 0, 1)
    y = np.concatenate([y[:1], y, y[-1:]])  # the first and last values repeated, for the two end intervals
    p0, p1, p2, p3 = y[i], y[i + 1], y[i + 2], y[i + 3]
    return 0.5 * (2 * p1 + (p2 - p0) * u + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u**2 + (3 * p1 - p0 - 3 * p2 + p3) * u**3)


def julian_iso(jd: float) -> str:
    seconds = (jd - 2440587.5) * DAY
    return str(np.datetime64(int(round(seconds)), "s")).replace("T", " ")


class Replay(Scene):
    """The first viewer of OrbitML: the truth played back on the globe and the ground track, with a fading trail."""

    speeds = [1, 10, 60, 300, 1200, 2400]
    speed = 3
    ORBIT_POINTS = 180  # points of the revolution drawn on the globe

    def setup(self) -> None:
        d = self.data
        self.seconds = (d.truth_jd - d.truth_jd[0]) * DAY
        self.duration = self.seconds[-1] / DAY
        self.r = d.truth_r
        self.v = np.gradient(self.r, self.seconds, axis=0)  # km/s
        self.speed_norm = np.linalg.norm(self.v, axis=1)

        # Revolution of each state, incremented at each ascending node (z going from south to north)
        z = self.r[:, 2]
        self.orbit = np.concatenate([[0], np.cumsum((z[:-1] < 0) & (z[1:] >= 0))])
        semi_major_axis = 1 / (2 / np.linalg.norm(self.r, axis=1) - self.speed_norm**2 / 398600.4418)  # vis-viva
        self.period = float(np.mean(2 * np.pi * np.sqrt(semi_major_axis**3 / 398600.4418)))  # s
        self.sent_orbit = None

        self.send(layout="both")
        self.send(titles=1, space="GLOBE (TEME)", map="TRACE AU SOL")
        self.send(camera=camera_facing(np.cross(self.r[0], self.v[0]), -45), center=[0, 0, 0], extent=7600)
        self.watches({"MISSION": "SENTINEL-6A", "DU": julian_iso(d.truth_jd[0]), "AU": julian_iso(d.truth_jd[-1])}, "SOURCE")

    def at(self, values: np.ndarray, seconds: np.ndarray | float) -> np.ndarray:
        values = np.stack([smooth(self.seconds, column, np.atleast_1d(seconds)) for column in np.atleast_2d(values.T)], axis=-1)
        return values

    def frame(self) -> None:
        d = self.data
        seconds = self.t * DAY
        jd = d.truth_jd[0] + self.t
        index = np.searchsorted(self.seconds, seconds, side="right") - 1
        orbit = int(self.orbit[index])
        if orbit != self.sent_orbit:
            # One revolution from the first state of this one, kept inside the data
            start = self.seconds[np.argmax(self.orbit == orbit)]
            start = max(0.0, min(start, self.seconds[-1] - self.period))
            ring = self.at(self.r, start + np.linspace(0.0, self.period, self.ORBIT_POINTS + 1))
            self.send(id="orbit", line=rounded(ring), color="blue", size=2)
            self.sent_orbit = orbit

        r = self.at(self.r, seconds)[0]
        latitude, longitude = latitude_longitude(r[None], np.array([jd]))[0]
        self.send(earth=earth_rotation(jd), surface=1)
        self.send(id="satellite", point=rounded(r), color="blue", size=5)
        self.send(id="track", trail=[round(float(latitude), 2), round(float(longitude), 2)], color="blue", size=2, length=900)
        self.send(id="track satellite", spots=[round(float(latitude), 2), round(float(longitude), 2)], colors=[0.45, 0.75, 0.98], size=5)
        self.watches({
            "UTC": julian_iso(jd),
            "REVOLUTION": orbit,
            "ALTITUDE": f"{np.linalg.norm(r) - EARTH_RADIUS:.1f} km",
            "LATITUDE": f"{latitude:+.2f} deg",
            "LONGITUDE": f"{longitude:+.2f} deg",
            "VITESSE": f"{np.interp(seconds, self.seconds, self.speed_norm):.3f} km/s",
        }, "SATELLITE")

    def command(self, name: str) -> None:
        if name == "restart":
            self.sent_orbit = None
        super().command(name)


SCENES = {"1": Truth, "3": Frame, "8": Drift, "9": Node, "13": Sun, "14": Ground, "fin": Corrected, "replay": Replay}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scene", choices=SCENES)
    parser.add_argument("--at", type=float, default=0.0, help="start this many days into the scene")
    parser.add_argument("--tle", type=int, help="index of the TLE in the notebook's dataset (default: the first test TLE with 14 days of truth)")
    args = parser.parse_args()

    data = Data()
    first = int(np.flatnonzero(data.test & np.isfinite(data.error[:, 0, -1]))[0])
    tle = first if args.tle is None else args.tle
    if args.scene == "13" and args.tle is None:
        tle = 0  # the 60 days start with the year, inside the training TLEs

    with AuroraClient() as aurora:
        print(f"Connected to {aurora.address}  (Ctrl+C to stop)")
        scene = SCENES[args.scene](data, aurora, tle)
        scene.t = args.at
        scene.start()
        last = time.perf_counter()
        while True:
            for command in aurora.commands():
                scene.command(command.name)
            now = time.perf_counter()
            scene.advance(now - last)
            last = now
            scene.frame()
            time.sleep(max(0.0, 1 / FPS - (time.perf_counter() - now)))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")
    except AuroraUnavailableError as error:
        print(f"{error}, is orbit_viewer running?")
    except AuroraDisconnectedError:
        print("Viewer closed the connection.")
