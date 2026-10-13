# -*- coding: utf-8 -*-
"""
Maç Analiz Paneli · Matrix V3 — TEK DOSYA (harici modül gerekmez)

Çalıştırma:   python -m pip install streamlit pandas numpy requests
              python -m streamlit run app.py
Veri:         Aynı klasörde  data/europe-master.zip  ve  data/football.json-master.zip
              (yoksa kenar çubuğundan yol verilebilir). API anahtarı dosyanın içine gömülüdür (EMBEDDED_KEYS).
Bölümler:     1) Veri yükleme  2) İstatistik modeli  3) Matrix kuralları  4) V2 karakter/patlama/radar
              5) Canlı API  6) V3 senelik/aylık örüntüler  7) Streamlit arayüzü
"""
from __future__ import annotations

import io
import json
import math
import os
import re
import time
import unicodedata
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st



# ======================================================================
# 1) VERİ YÜKLEME
# ======================================================================
"""
data_loader.py
--------------
ZIP dosyalarındaki (openfootball formatı: football.json + Football.TXT) maç verilerini
dinamik olarak okur, tek bir Pandas DataFrame'de birleştirir, temizler ve standartlaştırır.

Çıktı sütunları:
    date, season, league, league_name, week,
    home, away, HTHG, HTAG, FTHG, FTAG, played (bool)
Not: Veri setlerinde kart / şut / xG bilgisi YOKTUR; yalnızca ilk yarı ve maç sonu skoru vardır.
"""



# --------------------------------------------------------------------------- #
# Lig kodları -> okunur isimler (tanımsız kodlar "ÜLKE kod" olarak gösterilir)
# --------------------------------------------------------------------------- #
LEAGUE_NAMES = {
    "en1": "İngiltere – Premier League", "en2": "İngiltere – Championship",
    "en3": "İngiltere – League One", "en4": "İngiltere – League Two",
    "de1": "Almanya – Bundesliga", "de2": "Almanya – 2. Bundesliga", "de3": "Almanya – 3. Liga",
    "es1": "İspanya – La Liga", "es2": "İspanya – Segunda",
    "it1": "İtalya – Serie A", "it2": "İtalya – Serie B",
    "fr1": "Fransa – Ligue 1", "fr2": "Fransa – Ligue 2",
    "nl1": "Hollanda – Eredivisie", "nl2": "Hollanda – Eerste Divisie",
    "pt1": "Portekiz – Primeira Liga", "pt2": "Portekiz – Liga Portugal 2",
    "tr1": "Türkiye – Süper Lig", "tr2": "Türkiye – 1. Lig",
    "at1": "Avusturya – Bundesliga", "at2": "Avusturya – 2. Liga",
    "be1": "Belçika – Pro League", "gr1": "Yunanistan – Super League",
    "sco1": "İskoçya – Premiership", "br1": "Brezilya – Série A", "br2": "Brezilya – Série B",
    "mx1": "Meksika – Liga MX", "jp1": "Japonya – J1", "cn1": "Çin – Süper Lig",
    "ar1": "Arjantin – Liga Profesional", "ch1": "İsviçre – Super League",
    "ru1": "Rusya – Premier Liga", "cz1": "Çekya – 1. Liga", "hu1": "Macaristan – NB I",
    "ucl": "UEFA – Şampiyonlar Ligi", "mls": "ABD – MLS", "au1": "Avustralya – A-League", "dk1": "Danimarka – Superliga",
    "se1": "İsveç – Allsvenskan", "no1": "Norveç – Eliteserien", "pl1": "Polonya – Ekstraklasa",
    "ro1": "Romanya – Liga I", "hr1": "Hırvatistan – HNL", "rs1": "Sırbistan – Super Liga",
    "ua1": "Ukrayna – Premier Liga", "bg1": "Bulgaristan – First League",
}

# Bilinen takım ismi varyasyonları (normalize edilmiş anahtar -> ortak anahtar)
ALIASES = {
    "man city": "manchester city", "man united": "manchester united", "man utd": "manchester united",
    "spurs": "tottenham hotspur", "tottenham": "tottenham hotspur", "wolves": "wolverhampton wanderers",
    "wolverhampton": "wolverhampton wanderers", "newcastle": "newcastle united",
    "west ham": "west ham united", "brighton": "brighton hove albion", "leeds": "leeds united",
    "leicester": "leicester city", "norwich": "norwich city", "inter": "internazionale",
    "inter milano": "internazionale", "internazionale milano": "internazionale",
    "atletico madrid": "atletico", "bayern munchen": "bayern munich", "bayern": "bayern munich",
    "borussia monchengladbach": "monchengladbach", "paris saint germain": "paris saint germain",
    "psg": "paris saint germain", "besiktas jk": "besiktas", "istanbul basaksehir": "basaksehir",
    "medipol basaksehir": "basaksehir", "aytemiz alanyaspor": "alanyaspor", "istanbul bb": "basaksehir",
    "gazisehir gaziantep": "gaziantep", "atiker konyaspor": "konyaspor",
}
# İsmin başında/sonunda yer alıp kimliği değiştirmeyen kısaltmalar
NOISE_TOKENS = {
    "fc", "sk", "as", "a", "s", "cf", "afc", "ac", "sc", "fk", "nk", "sv", "vfb", "vfl", "tsv", "bk",
    "if", "ik", "cd", "ud", "sd", "ca", "fsv", "rcd", "rc", "ssc", "us", "ssd", "hsc", "ogc", "sco", "kv",
    "krc", "ksc", "rsc", "cs", "jk", "kf", "gnk", "hnk", "fcm", "spor", "kulubu", "as", "aş", "as",
}

_MD_RE = re.compile(r"(\d+)")


def _strip_accents(s: str) -> str:
    """Aksanları/Türkçe karakterleri ASCII'ye indirger (ı->i, ş->s, ö->o ...)."""
    s = s.replace("ı", "i").replace("İ", "I").replace("ß", "ss")
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def normalize_team_key(name: str) -> str:
    """Takım adı için karşılaştırma anahtarı üretir (yazım/ek farklarını yok sayar)."""
    s = re.sub(r"\s*\([A-Za-z]{3}\)\s*$", "", str(name))     # UCL verisindeki "(ENG)" ülke eki
    s = _strip_accents(s).lower()
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    toks = [t for t in s.split() if not t.isdigit()]
    while len(toks) > 1 and toks[0] in NOISE_TOKENS:
        toks.pop(0)
    while len(toks) > 1 and toks[-1] in NOISE_TOKENS:
        toks.pop()
    key = " ".join(toks)
    return ALIASES.get(key, key)


# --------------------------------------------------------------------------- #
# Ayrıştırıcılar
# --------------------------------------------------------------------------- #
def _is_league_code(code: str) -> bool:
    """Kupa/play-off/'full' dosyalarını eler; yalnızca lig kodlarını (en1, tr1, mls...) kabul eder."""
    return bool(re.fullmatch(r"[a-z]{2,3}\d", code)) or code in ("mls", "ucl")


def _season_start_year(season: str) -> int:
    return int(season[:4])


def _parse_json(raw: bytes, season: str, code: str) -> list[dict]:
    """football.json formatındaki bir lig-sezon dosyasını satır listesine çevirir."""
    data = json.loads(raw.decode("utf-8"))
    rows = []
    for m in data.get("matches", []):
        sc = m.get("score") or {}
        if isinstance(sc, list):  # bazı dosyalarda skor doğrudan [ev, dep] listesi (yalnızca maç sonu)
            sc = {"ft": sc}
        ft, ht = sc.get("ft"), sc.get("ht")
        ft = ft if ft and len(ft) == 2 else None
        ht = ht if ht and len(ht) == 2 else None
        rnd = _MD_RE.search(str(m.get("round", "")))
        rtxt = str(m.get("round", ""))
        # Şampiyonlar Ligi: Grup/Lig aşaması ile Eleme turlarını ayır
        stage = ("Grup" if rtxt.startswith(("Group", "League")) else "Eleme") if code == "ucl" else ""
        if code == "ucl" and stage == "Eleme":
            rnd = None
        rows.append({
            "date": m.get("date"), "season": season, "league": code, "stage": stage,
            "week": int(rnd.group(1)) if rnd else np.nan,
            "home": m.get("team1"), "away": m.get("team2"),
            "HTHG": ht[0] if ht else np.nan, "HTAG": ht[1] if ht else np.nan,
            "FTHG": ft[0] if ft else np.nan, "FTAG": ft[1] if ft else np.nan,
        })
    return rows


_DATE_FULL = re.compile(r"^\s*(?:[A-Z][a-z]{2}\s+)?([A-Z][a-z]{2})\s+(\d{1,2})(?:\s+(\d{4}))?\s*$")
_MATCH = re.compile(
    r"^\s+(?:\d{1,2}[:.]\d{2}\s+)?(?P<h>.+?)\s{2,}v\s+(?P<a>.+?)"
    r"(?:\s{2,}(?P<fh>\d+)-(?P<fa>\d+)(?:\s+\((?P<hh>\d+)-(?P<ha>\d+)\))?)?\s*$"
)
# Yeni format (2026/27 vb.): "Ev Sahibi  2-2 (0-0)  Deplasman" + altında golcü satırları
_MATCH_B = re.compile(
    r"^\s+(?:\d{1,2}[:.]\d{2}\s+)?(?P<h>[^()]+?)\s{2,}(?P<fh>\d+)-(?P<fa>\d+)"
    r"(?:\s+\((?P<hh>\d+)-(?P<ha>\d+)\))?\s{2,}(?P<a>[^()]+?)\s*$"
)
_MONTHS = {m: i for i, m in enumerate(
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], 1)}


def _parse_txt(raw: bytes, season: str, code: str) -> list[dict]:
    """Football.TXT formatındaki bir lig-sezon dosyasını satır listesine çevirir."""
    rows, week, cur_date = [], np.nan, None
    start_year = _season_start_year(season)
    for line in raw.decode("utf-8", errors="ignore").splitlines():
        if line.startswith(("▪", "»")):
            n = _MD_RE.search(line)
            week = int(n.group(1)) if n else np.nan
            continue
        m = _MATCH.match(line) or _MATCH_B.match(line)
        if m and "pen." not in line and "a.e.t." not in line:
            if cur_date is None:
                continue
            g = m.groupdict()
            rows.append({
                "date": cur_date, "season": season, "league": code, "week": week,
                "home": g["h"].strip(), "away": g["a"].strip(),
                "HTHG": int(g["hh"]) if g["hh"] else np.nan, "HTAG": int(g["ha"]) if g["ha"] else np.nan,
                "FTHG": int(g["fh"]) if g["fh"] else np.nan, "FTAG": int(g["fa"]) if g["fa"] else np.nan,
            })
            continue
        d = _DATE_FULL.match(line)
        if d and d.group(1) in _MONTHS:
            mon, day, yr = _MONTHS[d.group(1)], int(d.group(2)), d.group(3)
            if yr:
                year = int(yr)
            else:  # yıl yazılmamışsa sezondan çıkar (Tem–Ara => başlangıç yılı, diğerleri => bitiş yılı)
                year = start_year if (mon >= 7 or len(season) == 4) else start_year + 1
            cur_date = f"{year:04d}-{mon:02d}-{day:02d}"
    return rows


def _iter_sources(src):
    """ZIP dosyası ya da klasör içindeki (isim, bytes) çiftlerini üretir."""
    p = Path(src)
    if p.is_dir():
        for f in p.rglob("*"):
            if f.is_file() and f.suffix in (".json", ".txt"):
                yield f.as_posix(), f.read_bytes()
    else:
        with zipfile.ZipFile(p) as z:
            for n in z.namelist():
                if n.endswith((".json", ".txt")):
                    yield n, z.read(n)


def _read_source(src) -> list[dict]:
    """Bir kaynağı (ZIP/klasör) okuyup ham maç satırlarını döndürür."""
    rows = []
    for name, raw in _iter_sources(src):
        parts = name.split("/")
        base, folder = parts[-1], (parts[-2] if len(parts) > 1 else "")
        try:
            if base.endswith(".json") and re.fullmatch(r"\d{4}(-\d{2})?", folder):
                code = "ucl" if base == "uefa.cl.json" else base[:-5].replace(".", "")
                if _is_league_code(code):
                    rows += _parse_json(raw, folder, code)
            elif base.endswith(".txt"):
                m = re.fullmatch(r"(\d{4}(?:-\d{2})?)_([a-z0-9-]+)\.txt", base)
                if m and _is_league_code(m.group(2)):
                    rows += _parse_txt(raw, m.group(1), m.group(2))
        except (ValueError, KeyError):
            continue  # bozuk dosyayı atla
    return rows


# --------------------------------------------------------------------------- #
# Ana yükleyici
# --------------------------------------------------------------------------- #
def load_matches(sources: list[str], extra: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Verilen ZIP/klasör listesindeki tüm verileri okur, birleştirir, temizler.
    Kaynaklar listedeki sıraya göre önceliklidir (çift kayıtlarda ilki korunur).
    extra: canlı API'den gelen maçlar (live_data.py). Aynı maç hem ZIP'te hem API'de varsa API kaydı
           (daha güncel) tercih edilir; tarih farkı <= 2 gün olan kayıtlar aynı maç sayılır.
    """
    rows = []
    for s in sources:
        rows += _read_source(s)
    df = pd.DataFrame(rows)
    if extra is not None and len(extra):
        ex = extra.copy()
        ex["_live"] = True
        df = pd.concat([ex, df], ignore_index=True)       # canlı veri başa: önceliği yüksek
    if df.empty:
        return df
    if "_live" not in df:
        df["_live"] = False
    df["_live"] = df["_live"].fillna(False).astype(bool)
    if "stage" not in df:
        df["stage"] = ""
    df["stage"] = df["stage"].fillna("")

    # --- Temizlik / standardizasyon ---
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "home", "away"])
    for c in ["HTHG", "HTAG", "FTHG", "FTAG"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["played"] = df["FTHG"].notna() & df["FTAG"].notna()
    # İlk yarı skoru tutarsızsa (HT > FT) güvenilmez say -> NaN
    bad_ht = (df["HTHG"] > df["FTHG"]) | (df["HTAG"] > df["FTAG"])
    df.loc[bad_ht, ["HTHG", "HTAG"]] = np.nan

    # --- Takım isimlerini normalize et: ortak anahtar -> en sık kullanılan yazım ---
    df["_hk"] = df["home"].map(normalize_team_key)
    df["_ak"] = df["away"].map(normalize_team_key)
    cnt = Counter()
    for col, kcol in (("home", "_hk"), ("away", "_ak")):
        cnt.update(zip(df[kcol], df[col]))
    best = {}
    for (k, n), c in cnt.items():
        if k not in best or c > best[k][1]:
            best[k] = (n, c)
    df["home"] = df["_hk"].map(lambda k: best[k][0])
    df["away"] = df["_ak"].map(lambda k: best[k][0])

    # --- Çift kayıtları sil: aynı lig+takımlar ve tarih farkı <= 2 gün; oynanmış ve canlı kayıt öncelikli ---
    df["_src_order"] = np.arange(len(df))
    df = df.sort_values(["played", "_live", "_src_order"], ascending=[False, False, True])
    seen: dict = {}
    keep = []
    for lg, hk, ak, dt in zip(df["league"], df["_hk"], df["_ak"], df["date"].map(pd.Timestamp.toordinal)):
        lst = seen.setdefault((lg, hk, ak), [])
        dup = any(abs(dt - o) <= 2 for o in lst)
        keep.append(not dup)
        if not dup:
            lst.append(dt)
    df = df[keep].drop(columns=["_hk", "_ak", "_src_order", "_live"])

    df["league_name"] = df["league"].map(lambda c: LEAGUE_NAMES.get(c, c.upper()))
    df = df.sort_values("date").reset_index(drop=True)

    # Hafta bilgisi eksikse: takımın o sezondaki kaçıncı maçı olduğundan türet
    if df["week"].isna().any():
        long = pd.concat([
            df[["league", "season", "date", "home"]].rename(columns={"home": "t"}).assign(i=df.index),
            df[["league", "season", "date", "away"]].rename(columns={"away": "t"}).assign(i=df.index),
        ]).sort_values("date")
        long["n"] = long.groupby(["league", "season", "t"]).cumcount() + 1
        wk = long.groupby("i")["n"].max()
        df["week"] = df["week"].fillna(wk)
    df["week"] = df["week"].astype("float")
    return df


def default_sources(base: str | Path | None = None) -> list[str]:
    """data/ klasöründeki ZIP dosyalarını döndürür (football.json önce gelir: daha temiz veri)."""
    base = Path(base or Path(__file__).parent / "data")
    zips = sorted(base.glob("*.zip"), key=lambda p: (0 if "json" in p.name else 1, p.name))
    return [str(p) for p in zips]


# ======================================================================
# 2) İSTATİSTİK MODELİ
# ======================================================================
"""
analysis.py
-----------
İstatistiksel motor (arayüzden bağımsız, saf pandas/numpy):

  * Zaman ağırlıklı (üstel sönümlü) hücum/savunma gücü  -> beklenen gol (xG, xGA)
  * Poisson skor matrisi -> 1X2, en olası skorlar, Alt/Üst, KG Var
  * 6+ gol olasılığı  (Poisson kuyruğu + ampirik gözlem karması)
  * 1/2 ve 2/1 (geri dönüş) olasılığı  (takım bazlı HT->FT geçiş oranları + H2H)
  * Favori belirleme  (model + form + saha içi/dışı form + H2H)
  * Örüntü (pattern) tespiti: H2H serileri, aynı hafta/ay tekrarları, güncel seriler.
    Her örüntü için "tesadüf olasılığı" (taban oranın k. kuvveti) hesaplanır.

Not: Veri setinde kart/şut bilgisi olmadığı için örüntüler gol/skor/sonuç temellidir.
"""



HALF_LIFE_DAYS = 540      # maç ağırlığı her ~1.5 yılda yarıya düşer
HISTORY_YEARS = 6         # güç hesabında en fazla bu kadar geriye bakılır
SHRINK_K = 6.0            # az maçlı takımları lig ortalamasına çeken "sanal maç" sayısı
H2H_PSEUDO = 8.0          # H2H gollerinin model beklentisine karışma gücü
MAX_GOALS = 10

# ----- Eşik değerleri (threshold) — gerekirse buradan ayarlayın -----
REV_MIN = 0.15            # 1/2 veya 2/1 yalnızca modelde >= %15 ise (veya net örüntü varsa) gösterilir
P6_MIN = 0.15             # 6+ gol yalnızca >= %15 ise (veya net örüntü varsa) gösterilir
HIGH_GOAL_P35 = 0.40      # P(3.5 Üst) bu değerin üstündeyse skorlar 4+ gollü senaryodan seçilir
HIGH_GOAL_P25 = 0.50      # P(2.5 Üst) bu değerin üstündeyse skorlar 3+ gollü senaryodan seçilir
LOW_GOAL_P25 = 0.38       # P(2.5 Üst) bu değerin altındaysa skorlar 2 ve altı gollü senaryodan seçilir
HT_SIDE_MIN, HT_SIDE_GAP = 0.38, 0.12   # İY'de taraf seçmek için gereken olasılık / fark


# --------------------------------------------------------------------------- #
# Yardımcı fonksiyonlar
# --------------------------------------------------------------------------- #
def teams_in_league(d: pd.DataFrame, last_n_seasons: int = 2) -> list[str]:
    """Dropdown için: ligin son N sezonunda yer alan takımların alfabetik listesi."""
    seasons = sorted(d["season"].unique(), key=lambda s: d.loc[d["season"] == s, "date"].max())
    recent = d[d["season"].isin(seasons[-last_n_seasons:])]
    return sorted(set(recent["home"]) | set(recent["away"]))


def poisson_pmf(lam: float, n: int = MAX_GOALS) -> np.ndarray:
    """0..n gol için Poisson olasılıkları (log-gamma ile sayısal olarak kararlı)."""
    k = np.arange(n + 1)
    logp = -lam + k * math.log(lam) - np.array([math.lgamma(i + 1) for i in k])
    return np.exp(logp)


def shrunk_rate(sum_w_x: float, sum_w: float, prior: float, k: float) -> float:
    """Bayes büzülmesi: (Σw·x + k·prior) / (Σw + k)."""
    return (sum_w_x + k * prior) / (sum_w + k)


def _long_view(d: pd.DataFrame, ref: pd.Timestamp) -> pd.DataFrame:
    """Oynanmış maçları takım bazlı (her maç iki satır) uzun forma çevirir + zaman ağırlığı."""
    d = d[d["played"]]
    base = dict(date=d["date"], season=d["season"], week=d["week"])
    h = pd.DataFrame({**base, "team": d["home"], "opp": d["away"], "venue": "H",
                      "gf": d["FTHG"], "ga": d["FTAG"], "htgf": d["HTHG"], "htga": d["HTAG"]})
    a = pd.DataFrame({**base, "team": d["away"], "opp": d["home"], "venue": "A",
                      "gf": d["FTAG"], "ga": d["FTHG"], "htgf": d["HTAG"], "htga": d["HTHG"]})
    lv = pd.concat([h, a]).sort_values("date").reset_index(drop=True)
    lv["pts"] = np.where(lv.gf > lv.ga, 3, np.where(lv.gf == lv.ga, 1, 0))
    age = (ref - lv["date"]).dt.days.clip(lower=0)
    lv["w"] = 0.5 ** (age / HALF_LIFE_DAYS)
    lv["tg"] = lv.gf + lv.ga
    lv["btts"] = (lv.gf > 0) & (lv.ga > 0)
    lv["over25"] = lv.tg >= 3
    lv["ht_has"] = lv.htgf.notna()
    return lv


def _wmean(x: pd.Series, w: pd.Series) -> tuple[float, float]:
    """Ağırlıklı toplam ve ağırlık toplamı (NaN'ler hariç)."""
    m = x.notna()
    return float((x[m] * w[m]).sum()), float(w[m].sum())


# --------------------------------------------------------------------------- #
# Güç modeli (hücum / savunma) ve Poisson
# --------------------------------------------------------------------------- #
@dataclass
class Strength:
    attack: float      # lig ortalamasına oranlı hücum gücü (1.0 = ortalama)
    defense: float     # lig ortalamasına oranlı yenilen gol (1.0 = ortalama, <1 iyi savunma)
    n: int             # kullanılan maç sayısı
    gf_pg: float       # maç başı atılan gol (ham, ağırlıklı)
    ga_pg: float       # maç başı yenilen gol (ham, ağırlıklı)


def team_strength(lv: pd.DataFrame, team: str, avg_goals: float) -> Strength:
    """Takımın zaman ağırlıklı, büzülmeli hücum ve savunma gücü."""
    t = lv[lv.team == team]
    sw = t["w"].sum()
    gf_s, ga_s = (t.gf * t.w).sum(), (t.ga * t.w).sum()
    att = shrunk_rate(gf_s, sw, avg_goals, SHRINK_K) / avg_goals
    dfn = shrunk_rate(ga_s, sw, avg_goals, SHRINK_K) / avg_goals
    return Strength(att, dfn, len(t), gf_s / sw if sw else np.nan, ga_s / sw if sw else np.nan)


def score_matrix(lh: float, la: float) -> np.ndarray:
    """Bağımsız Poisson varsayımıyla (ev x deplasman) skor olasılık matrisi."""
    m = np.outer(poisson_pmf(lh), poisson_pmf(la))
    return m / m.sum()


def matrix_summary(m: np.ndarray) -> dict:
    """Skor matrisinden 1X2, KG Var, Alt/Üst ve en olası skorları çıkarır."""
    n = m.shape[0]
    i, j = np.indices(m.shape)
    tot = i + j
    res = {
        "p_home": float(m[i > j].sum()), "p_draw": float(m[i == j].sum()), "p_away": float(m[i < j].sum()),
        "btts": float(m[1:, 1:].sum()),
        "over": {L: float(m[tot > L].sum()) for L in (0.5, 1.5, 2.5, 3.5, 4.5)},
        "p6": float(m[tot >= 6].sum()),
    }
    flat = np.dstack(np.unravel_index(np.argsort(-m, axis=None), m.shape))[0][:3]
    res["top_scores"] = [(int(a), int(b), float(m[a, b])) for a, b in flat]
    return res


# --------------------------------------------------------------------------- #
# 1/2 - 2/1 (geri dönüş) analizi
# --------------------------------------------------------------------------- #
def _ht_stats(lv: pd.DataFrame, team: str | None, venue: str | None) -> dict:
    """
    HT durumuna göre koşullu oranlar (zaman ağırlıklı):
      lead   = P(HT'de önde)          trail = P(HT'de geride)
      blow   = P(MS yenilgi | HT önde) comeback = P(MS galibiyet | HT geride)
    team=None -> tüm lig tabanı. Her biri (pay_w, payda_w) olarak döner.
    """
    t = lv[lv.ht_has]
    if team is not None:
        t = t[t.team == team]
    if venue is not None:
        t = t[t.venue == venue]
    lead, trail = t.htgf > t.htga, t.htgf < t.htga
    w = t.w
    return {
        "lead": (w[lead].sum(), w.sum()), "trail": (w[trail].sum(), w.sum()),
        "blow": (w[lead & (t.gf < t.ga)].sum(), w[lead].sum()),
        "comeback": (w[trail & (t.gf > t.ga)].sum(), w[trail].sum()),
        "n": len(t), "n_lead": int(lead.sum()), "n_trail": int(trail.sum()),
        "n_blow": int((lead & (t.gf < t.ga)).sum()), "n_comeback": int((trail & (t.gf > t.ga)).sum()),
    }


def reversal_analysis(lv: pd.DataFrame, home: str, away: str, h2h: pd.DataFrame) -> dict:
    """
    P(1/2): ev sahibi HT önde, MS deplasman kazanır.   P(2/1): tersi.
    Her bileşen takımın kendi sahası/deplasmanındaki oranı olup lig tabanına büzülür;
    sonuç H2H'deki gözlemlenen sıklıkla (n/(n+10) ağırlıkla) harmanlanır.
    """
    K = 12.0
    base_h, base_a = _ht_stats(lv, None, "H"), _ht_stats(lv, None, "A")

    def rate(st, key, base):
        num, den = st[key]; bnum, bden = base[key]
        prior = bnum / bden if bden else 0.0
        return shrunk_rate(num, den, prior, K) if den or bden else prior

    sh, sa = _ht_stats(lv, home, "H"), _ht_stats(lv, away, "A")
    # 1/2 : ev HT önde  &  deplasman kazanır
    p_home_lead = 0.5 * (rate(sh, "lead", base_h) + rate(sa, "trail", base_a))
    p_away_win_given = 0.5 * (rate(sh, "blow", base_h) + rate(sa, "comeback", base_a))
    # 2/1 : deplasman HT önde  &  ev kazanır
    p_away_lead = 0.5 * (rate(sh, "trail", base_h) + rate(sa, "lead", base_a))
    p_home_win_given = 0.5 * (rate(sh, "comeback", base_h) + rate(sa, "blow", base_a))
    m12, m21 = p_home_lead * p_away_win_given, p_away_lead * p_home_win_given

    # H2H gözlemi (ev sahibi takım perspektifinden)
    hh = h2h[h2h.hth.notna()]
    n = len(hh)
    c12 = int(((hh.hth > hh.hta) & (hh.fh < hh.fa)).sum())
    c21 = int(((hh.hth < hh.hta) & (hh.fh > hh.fa)).sum())
    KH = 10.0
    p12 = (m12 * KH + c12) / (KH + n)
    p21 = (m21 * KH + c21) / (KH + n)

    # Genel (tüm saha) lig performansı özeti
    gh, ga = _ht_stats(lv, home, None), _ht_stats(lv, away, None)
    return {
        "p12": p12, "p21": p21, "p_any": p12 + p21, "model12": m12, "model21": m21,
        "h2h_n": n, "h2h_12": c12, "h2h_21": c21,
        "home_overall": gh, "away_overall": ga, "base_overall": _ht_stats(lv, None, None),
    }


# --------------------------------------------------------------------------- #
# H2H, form
# --------------------------------------------------------------------------- #
def head_to_head(d: pd.DataFrame, home: str, away: str) -> pd.DataFrame:
    """İki takımın oynanmış tüm maçları; sütunlar ev(home) takımı perspektifinden normalize edilir."""
    d = d[d["played"]]
    a = d[(d.home == home) & (d.away == away)]
    b = d[(d.home == away) & (d.away == home)]
    out = pd.concat([
        pd.DataFrame({"date": a.date, "season": a.season, "week": a.week, "fh": a.FTHG, "fa": a.FTAG,
                      "hth": a.HTHG, "hta": a.HTAG, "at_home": True}),
        pd.DataFrame({"date": b.date, "season": b.season, "week": b.week, "fh": b.FTAG, "fa": b.FTHG,
                      "hth": b.HTAG, "hta": b.HTHG, "at_home": False}),
    ]).sort_values("date").reset_index(drop=True)
    return out


def _ppg(t: pd.DataFrame, n: int) -> tuple[float, int]:
    """Son n maçın maç başı puanı ve gerçek maç sayısı."""
    t = t.tail(n)
    return (float(t.pts.mean()) if len(t) else np.nan), len(t)


def form_string(lv: pd.DataFrame, team: str, n: int = 5) -> str:
    t = lv[lv.team == team].tail(n)
    return "".join("G" if p == 3 else "B" if p == 1 else "M" for p in t.pts)


# --------------------------------------------------------------------------- #
# Örüntü (pattern) tespiti
# --------------------------------------------------------------------------- #
def _streak_len(flags: list[bool]) -> int:
    """Listenin SONUNDAN geriye doğru ardışık True sayısı (en yeniden eskiye)."""
    k = 0
    for f in reversed(flags):
        if f:
            k += 1
        else:
            break
    return k


def _team_conditions(name: str) -> dict:
    """Takım perspektifinden satır koşulları (lv satırı -> bool seri)."""
    return {
        "yenilgi aldı": lambda t: t.gf < t.ga,
        "galibiyet aldı": lambda t: t.gf > t.ga,
        "yenilmedi": lambda t: t.gf >= t.ga,
        "kazanamadı": lambda t: t.gf <= t.ga,
        "karşılıklı gol oldu (KG Var)": lambda t: t.btts,
        "2.5 üstü bitti": lambda t: t.over25,
        "2.5 altı bitti": lambda t: ~t.over25,
        "gol yemedi (clean sheet)": lambda t: t.ga == 0,
        "ilk yarıda gol yedi": lambda t: t.htga.notna() & (t.htga > 0),
        "ilk yarıda gol attı": lambda t: t.htgf.notna() & (t.htgf > 0),
    }


def _chance(base: float, k: int) -> float:
    return float(min(max(base, 1e-3), 0.999) ** k)


def find_patterns(d: pd.DataFrame, lv: pd.DataFrame, home: str, away: str,
                  h2h: pd.DataFrame, match_date: pd.Timestamp, week: float | None) -> list[dict]:
    """
    Tekrar eden örüntüleri bulur. Her biri: {text, k, chance, kind}.
    chance = bu serinin salt şansla oluşma olasılığı (taban oran ^ seri uzunluğu). Küçük = anlamlı.
    """
    out: list[dict] = []
    month_tr = ["Ocak", "Şubat", "Mart", "Nisan", "Mayıs", "Haziran", "Temmuz", "Ağustos", "Eylül",
                "Ekim", "Kasım", "Aralık"][match_date.month - 1]

    # --- 1) H2H serileri (son 3+ karşılaşma) ---
    if len(h2h) >= 3:
        lg = lv[lv.venue == "H"]                       # lig geneli (maç başına tek satır)
        r_btts, r_o25 = float(lg.btts.mean()), float(lg.over25.mean())
        lh = lg[lg.ht_has]
        r_htg = float(((lh.htgf + lh.htga) > 0).mean())
        r_draw = float((lg.gf == lg.ga).mean())
        th, ta = lv[lv.team == home], lv[lv.team == away]  # taban oranlar takımın kendi genel oranı
        hc = {
            "karşılıklı gol oldu (KG Var)": (lambda x: (x.fh > 0) & (x.fa > 0), r_btts),
            "karşılıklı gol olmadı": (lambda x: (x.fh == 0) | (x.fa == 0), 1 - r_btts),
            "2.5 üstü bitti": (lambda x: x.fh + x.fa >= 3, r_o25),
            "2.5 altı bitti": (lambda x: x.fh + x.fa < 3, 1 - r_o25),
            "ilk yarıda gol görüldü": (lambda x: (x.hth + x.hta) > 0, r_htg),
            "ilk yarı golsüz bitti": (lambda x: (x.hth + x.hta) == 0, 1 - r_htg),
            f"{home} yenilmedi": (lambda x: x.fh >= x.fa, float((th.gf >= th.ga).mean())),
            f"{away} yenilmedi": (lambda x: x.fa >= x.fh, float((ta.gf >= ta.ga).mean())),
            f"{home} kazandı": (lambda x: x.fh > x.fa, float((th.gf > th.ga).mean())),
            f"{away} kazandı": (lambda x: x.fa > x.fh, float((ta.gf > ta.ga).mean())),
            "berabere bitti": (lambda x: x.fh == x.fa, r_draw),
        }
        for name, (f, b) in hc.items():
            sub = h2h[h2h.hth.notna()] if "ilk yarı" in name else h2h
            k = _streak_len(list(f(sub)))
            if k >= 3:
                out.append({"kind": "H2H", "k": k, "chance": _chance(b, k),
                            "text": f"Bu iki takımın son {k} karşılaşmasının tamamında {name}."})

    # --- 2) Aynı hafta örüntüsü (her takım, geçmiş sezonlarda aynı hafta numarası) ---
    if week is not None and not np.isnan(week):
        for team in (home, away):
            t = lv[(lv.team == team)]
            same = t[t.week == week].sort_values("date")
            tb = t
            if len(same) >= 3:
                for name, f in _team_conditions(team).items():
                    flags = list(f(same))
                    k = _streak_len(flags)
                    if k >= 3:
                        base = float(f(tb).mean())
                        out.append({"kind": "Hafta", "k": k, "chance": _chance(base, k),
                                    "text": f"{team}, son {k} sezondur ligin {int(week)}. haftasında "
                                            f"{name}."})

    # --- 3) Aynı ay örüntüsü (puan ortalaması, z-testi) ---
    for team in (home, away):
        t = lv[lv.team == team]
        tm = t[t.date.dt.month == match_date.month]
        n_seasons = tm.season.nunique()
        if len(tm) >= 8 and n_seasons >= 3 and t.pts.std() > 0:
            z = (tm.pts.mean() - t.pts.mean()) / (t.pts.std() / math.sqrt(len(tm)))
            if abs(z) >= 1.75:
                lost_m = 100 * (1 - tm.pts.sum() / (3 * len(tm)))
                lost_all = 100 * (1 - t.pts.sum() / (3 * len(t)))
                p = math.erfc(abs(z) / math.sqrt(2))     # iki yönlü normal yaklaşım
                out.append({"kind": "Ay", "k": len(tm), "chance": p,
                            "text": f"{team}, {month_tr} aylarında oynadığı {len(tm)} maçta ({n_seasons} sezon) "
                                    f"puanların %{lost_m:.0f}'ini kaybetmiş (genel ortalama %{lost_all:.0f}). "
                                    f"{'Olumsuz' if z < 0 else 'Olumlu'} sapma, z={z:+.1f}."})

    # --- 4) Güncel seriler (saha bazlı) ---
    for team, venue, label in ((home, "H", "evinde"), (away, "A", "deplasmanda")):
        t_all = lv[lv.team == team]
        t = t_all[t_all.venue == venue]
        for tt, tag in ((t_all, "son"), (t, f"{label} oynadığı son")):
            for name, f in _team_conditions(team).items():
                if len(tt) < 10:
                    continue
                flags = list(f(tt))
                k = _streak_len(flags)
                if k >= 4:
                    base = float(f(tt).mean())
                    out.append({"kind": "Seri", "k": k, "chance": _chance(base, k),
                                "text": f"{team}, {tag} {k} maçın tamamında {name}."})

    # Aynı bilgiyi tekrar eden (aynı metin) kayıtları ele, en anlamlı olanlar öne
    seen, uniq = set(), []
    for p in sorted(out, key=lambda p: p["chance"]):
        if p["text"] not in seen and p["chance"] < 0.10:
            seen.add(p["text"]); uniq.append(p)
    return uniq[:10]


# --------------------------------------------------------------------------- #
# Ana fonksiyon
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
# Gol potansiyeli, dinamik skor seçimi
# --------------------------------------------------------------------------- #
def empirical_over(lv: pd.DataFrame, home: str, away: str, lines=(1.5, 2.5, 3.5, 4.5)) -> dict:
    """İki takımın son maçlarındaki (zaman ağırlıklı) gerçek Üst oranları, lig tabanına büzülmüş."""
    rows = lv[lv.team.isin([home, away])]
    base = lv[lv.venue == "H"]
    return {L: shrunk_rate(((rows.tg > L) * rows.w).sum(), rows.w.sum(), float((base.tg > L).mean()), 20.0)
            for L in lines}


def dynamic_scores(m: np.ndarray, over: dict) -> tuple[list, str]:
    """
    Skor tahminini maçın gol potansiyeline göre seçer.
    Saf Poisson'da en olası tek skor hep 1-0/1-1 olur (kütle dağılır). Bu yüzden önce gol senaryosu
    belirlenir (harmanlanmış 2.5/3.5 Üst olasılıklarına göre), sonra skorlar o senaryoya KOŞULLANMIŞ
    matristen seçilir:
        P(3.5 Üst) >= %40  -> toplam gol >= 4   (örn. 3-1, 2-2, 3-2, 4-1)
        P(2.5 Üst) >= %50  -> toplam gol >= 3   (örn. 2-1, 1-2, 3-0)
        P(2.5 Üst) <= %38  -> toplam gol <= 2   (örn. 1-0, 1-1, 0-1)
    Üç skor aynı sonuca (1/X/2) ait en fazla 2 tane olabilir (çeşitlilik).
    Dönen olasılıklar maçın GERÇEK (koşulsuz) olasılıklarıdır.
    """
    i, j = np.indices(m.shape)
    tot = i + j
    if over[3.5] >= HIGH_GOAL_P35:
        mask, label = tot >= 4, "Çok gollü senaryo (4+ gol)"
    elif over[2.5] >= HIGH_GOAL_P25:
        mask, label = tot >= 3, "Gollü senaryo (3+ gol)"
    elif over[2.5] <= LOW_GOAL_P25:
        mask, label = tot <= 2, "Düşük gollü senaryo (2 veya daha az gol)"
    else:
        mask, label = tot >= 2, "Dengeli gol senaryosu (2+ gol)"
    cond = np.where(mask, m, 0.0)
    picks, per_outcome = [], {}
    for idx in np.argsort(-cond, axis=None):
        a, b = divmod(int(idx), m.shape[1])
        if cond[a, b] <= 0:
            break
        o = 1 if a > b else 2 if a < b else 0
        if per_outcome.get(o, 0) >= 2:
            continue
        per_outcome[o] = per_outcome.get(o, 0) + 1
        picks.append((a, b, float(m[a, b])))
        if len(picks) == 3:
            break
    return picks, label


# --------------------------------------------------------------------------- #
# İlk yarı / ikinci yarı modeli ve momentum
# --------------------------------------------------------------------------- #
def half_model(lv: pd.DataFrame, home: str, away: str, lam_h: float, lam_a: float) -> dict:
    """
    Her takımın 1. ve 2. yarı hücum/savunma gücünü ayrı hesaplar. Maç sonu beklenen golü (lam) bu iki yarının
    göreli ağırlığına göre böler -> İY beklenen golü. Momentum = (2Y gol farkı/maç) - (1Y gol farkı/maç).
    """
    t = lv[lv.ht_has].copy()
    t["g1f"], t["g1a"] = t.htgf, t.htga
    t["g2f"], t["g2a"] = t.gf - t.htgf, t.ga - t.htga
    H = t[t.venue == "H"]
    sw = H.w.sum()
    mu = {k: float((H[c] * H.w).sum() / sw) for k, c in
          (("1h", "g1f"), ("1a", "g1a"), ("2h", "g2f"), ("2a", "g2a"))}
    avg1, avg2 = (mu["1h"] + mu["1a"]) / 2, (mu["2h"] + mu["2a"]) / 2

    def rates(team):
        r = t[t.team == team]; w = r.w.sum()
        f = lambda c, avg: shrunk_rate((r[c] * r.w).sum(), w, avg, SHRINK_K) / avg
        gd1 = float(((r.g1f - r.g1a) * r.w).sum() / w) if w else 0.0
        gd2 = float(((r.g2f - r.g2a) * r.w).sum() / w) if w else 0.0
        return {"a1": f("g1f", avg1), "d1": f("g1a", avg1), "a2": f("g2f", avg2), "d2": f("g2a", avg2),
                "gd1": gd1, "gd2": gd2, "momentum": gd2 - gd1,
                "p1_share": float(((r.g1f.sum()) / max(r.gf.sum(), 1)))}
    rh, ra = rates(home), rates(away)
    l1h, l2h = rh["a1"] * ra["d1"] * mu["1h"], rh["a2"] * ra["d2"] * mu["2h"]
    l1a, l2a = ra["a1"] * rh["d1"] * mu["1a"], ra["a2"] * rh["d2"] * mu["2a"]
    lh1 = float(np.clip(lam_h * l1h / (l1h + l2h), 0.05, 3.0))
    la1 = float(np.clip(lam_a * l1a / (l1a + l2a), 0.05, 3.0))
    ms = matrix_summary(score_matrix(lh1, la1))
    p1, px, p2 = ms["p_home"], ms["p_draw"], ms["p_away"]
    side = max(p1, p2)
    if side >= HT_SIDE_MIN and abs(p1 - p2) >= HT_SIDE_GAP and side >= px - 0.05:
        pick = "1" if p1 > p2 else "2"
    else:
        pick = "X"
    return {"p1": p1, "px": px, "p2": p2, "pick": pick, "lam1": (lh1, la1),
            "home": rh, "away": ra}


# --------------------------------------------------------------------------- #
# Sürpriz (1/2, 2/1, 6+) filtresi
# --------------------------------------------------------------------------- #
def surprise_signals(lv: pd.DataFrame, h2h: pd.DataFrame, home: str, away: str,
                     week: float | None, rev: dict, p6_final: float) -> list[dict]:
    """
    Sürpriz ihtimalleri yalnızca ciddi bir sinyal varsa döner:
      * model olasılığı eşik üstünde (1/2, 2/1: REV_MIN; 6+: P6_MIN), veya
      * H2H'de son 8 maçta en az 2 geri dönüş (HT önde olan kaybetti) / son 10 maçta en az 2 adet 6+ gol, veya
      * aynı hafta numarasında (son 6 sezon) takımlardan birinin maçlarının en az %30'unda (en az 2 kez) geri dönüş / 6+ gol.
    Hiçbiri yoksa boş liste döner (arayüz sade bir mesaj yazar).
    """
    out: list[dict] = []
    if rev["p12"] >= REV_MIN:
        out.append({"kind": "1/2", "prob": rev["p12"], "reason": "Model olasılığı eşiğin üzerinde."})
    if rev["p21"] >= REV_MIN:
        out.append({"kind": "2/1", "prob": rev["p21"], "reason": "Model olasılığı eşiğin üzerinde."})
    if p6_final >= P6_MIN:
        out.append({"kind": "6+ Gol", "prob": p6_final, "reason": "Model olasılığı eşiğin üzerinde."})

    def is_rev(t):
        return t.ht_has & (((t.htgf > t.htga) & (t.gf < t.ga)) | ((t.htgf < t.htga) & (t.gf > t.ga)))

    hh = h2h[h2h.hth.notna()].tail(8)
    n_rev = int((((hh.hth > hh.hta) & (hh.fh < hh.fa)) | ((hh.hth < hh.hta) & (hh.fh > hh.fa))).sum())
    if n_rev >= 2:
        out.append({"kind": "1/2 veya 2/1", "prob": None,
                    "reason": f"Bu iki takımın son {len(hh)} H2H maçının {n_rev}'inde ilk yarıyı önde kapatan "
                              f"taraf maçı kaybetti."})
    n6 = int(((h2h.tail(10).fh + h2h.tail(10).fa) >= 6).sum())
    if n6 >= 2:
        out.append({"kind": "6+ Gol", "prob": None,
                    "reason": f"Bu iki takımın son {min(len(h2h), 10)} H2H maçının {n6}'inde 6 ve üzeri gol atıldı."})

    if week is not None and not np.isnan(week):
        for team in (home, away):
            t = lv[(lv.team == team) & (lv.week == week)].tail(6)   # son 6 sezon
            if len(t) >= 3:
                k, k6 = int(is_rev(t).sum()), int((t.tg >= 6).sum())
                if k >= 2 and k / len(t) >= 0.30:
                    out.append({"kind": "1/2 veya 2/1", "prob": None,
                                "reason": f"{team}, geçmiş sezonlarda ligin {int(week)}. haftasında {len(t)} maçın "
                                          f"{k}'inde ilk yarı/maç sonu sonucunu çevirdi veya çevrildi."})
                if k6 >= 2 and k6 / len(t) >= 0.30:
                    out.append({"kind": "6+ Gol", "prob": None,
                                "reason": f"{team}, geçmiş sezonlarda ligin {int(week)}. haftasında {len(t)} maçın "
                                          f"{k6}'ında 6+ gol gördü."})
    return out


def find_next_fixture(d: pd.DataFrame, home: str, away: str, today: pd.Timestamp):
    """Ev sahibi-deplasman eşleşmesi için henüz oynanmamış en yakın fikstür (varsa)."""
    f = d[(~d.played) & (d.home == home) & (d.away == away) & (d.date >= today - pd.Timedelta(days=3))]
    return None if f.empty else f.sort_values("date").iloc[0]


def analyze(d: pd.DataFrame, home: str, away: str,
            match_date: pd.Timestamp | None = None, week: float | None = None) -> dict:
    """
    d: tek bir lige ait tüm maçlar (oynanmış + fikstür). Tüm analizleri tek sözlükte döndürür.
    """
    played = d[d.played]
    ref = played["date"].max()
    today = pd.Timestamp.today().normalize()
    fx = find_next_fixture(d, home, away, today)
    if match_date is None:
        match_date = fx["date"] if fx is not None else today
    if week is None and fx is not None:
        week = fx["week"]
    if week is None:
        mine = played[(played.home == home) | (played.away == home)].sort_values("date")
        if mine.empty:
            week = np.nan
        else:
            wk = mine[mine.season == mine.season.iloc[-1]].week.max()
            week = wk + 1 if pd.notna(wk) else np.nan

    # Güç hesabı için son HISTORY_YEARS yıl; örüntüler için tüm geçmiş
    recent = played[played.date >= ref - pd.Timedelta(days=365 * HISTORY_YEARS)]
    lv_recent = _long_view(recent, ref)
    lv_all = _long_view(d, ref)

    # --- Lig tabanları (ağırlıklı ev/deplasman gol ortalaması) ---
    w_m = lv_recent[lv_recent.venue == "H"]
    mu_h = (w_m.gf * w_m.w).sum() / w_m.w.sum()
    mu_a = (w_m.ga * w_m.w).sum() / w_m.w.sum()
    avg = (mu_h + mu_a) / 2

    sh, sa = team_strength(lv_recent, home, avg), team_strength(lv_recent, away, avg)
    lam_h = sh.attack * sa.defense * avg * (mu_h / avg)
    lam_a = sa.attack * sh.defense * avg * (mu_a / avg)

    # H2H: modelin beklentisini gözlemlenen H2H gol ortalamasına doğru çek (Bayes)
    h2h = head_to_head(d, home, away)
    h2h_last = h2h.tail(10)
    if len(h2h_last):
        n = len(h2h_last)
        lam_h = (h2h_last.fh.sum() + H2H_PSEUDO * lam_h) / (n + H2H_PSEUDO)
        lam_a = (h2h_last.fa.sum() + H2H_PSEUDO * lam_a) / (n + H2H_PSEUDO)
    lam_h, lam_a = float(np.clip(lam_h, 0.15, 4.5)), float(np.clip(lam_a, 0.15, 4.5))

    m = score_matrix(lam_h, lam_a)
    ms = matrix_summary(m)

    # --- Gol potansiyeli: Poisson Üst olasılıkları ile iki takımın GERÇEK Üst sıklığının ortalaması ---
    emp_over = empirical_over(lv_recent, home, away)
    over = {L: 0.5 * ms["over"][L] + 0.5 * emp_over[L] for L in emp_over}
    top_scores, score_scenario = dynamic_scores(m, over)

    # --- 6+ gol: Poisson kuyruğu ile ampirik sıklığın karması ---
    league_p6 = float((lv_recent[lv_recent.venue == "H"].tg >= 6).mean())
    emp_rows = pd.concat([lv_recent[(lv_recent.team == home) & (lv_recent.venue == "H")],
                          lv_recent[(lv_recent.team == away) & (lv_recent.venue == "A")]])
    emp_p6 = shrunk_rate(((emp_rows.tg >= 6) * emp_rows.w).sum(), emp_rows.w.sum(), league_p6, 40.0)
    p6 = 0.5 * ms["p6"] + 0.5 * emp_p6
    h2h_p6 = float(((h2h.fh + h2h.fa) >= 6).mean()) if len(h2h) else np.nan

    # --- Form bileşenleri ---
    lv_h = lv_all[lv_all.team == home]; lv_a = lv_all[lv_all.team == away]
    ppg_h5, _ = _ppg(lv_h, 5); ppg_a5, _ = _ppg(lv_a, 5)
    ppg_h_home, _ = _ppg(lv_h[lv_h.venue == "H"], 8)
    ppg_a_away, _ = _ppg(lv_a[lv_a.venue == "A"], 8)
    nh = len(h2h_last)
    if nh:
        share = ((h2h_last.fh > h2h_last.fa).sum() + 0.5 * (h2h_last.fh == h2h_last.fa).sum()) / nh
        h2h_score = (2 * share - 1) * nh / (nh + 4)
    else:
        share, h2h_score = np.nan, 0.0
    f5 = 0.0 if np.isnan(ppg_h5 - ppg_a5) else (ppg_h5 - ppg_a5) / 3
    fv = 0.0 if np.isnan(ppg_h_home - ppg_a_away) else (ppg_h_home - ppg_a_away) / 3
    # İlk yarı / ikinci yarı modeli ve momentum (2. yarıda yükselen takım FT'de avantaj kazanır)
    hm = half_model(lv_recent, home, away, lam_h, lam_a)
    mom_diff = float(np.clip(hm["home"]["momentum"] - hm["away"]["momentum"], -1, 1))
    tilt_score = 0.35 * f5 + 0.30 * fv + 0.20 * h2h_score + 0.15 * mom_diff   # [-1, 1] aralığında

    delta = 0.5 * tilt_score
    ph, pd_, pa = ms["p_home"] * math.exp(delta), ms["p_draw"], ms["p_away"] * math.exp(-delta)
    s = ph + pd_ + pa
    ph, pd_, pa = ph / s, pd_ / s, pa / s
    margin = abs(ph - pa)
    if margin < 0.10 and max(ph, pa) < 0.45:
        favorite, fav_note = "Beraberliğe Yakın", "Güçler dengede"
        ft_pick = "X"
    else:
        ft_pick = "1" if ph > pa else "2"
        fav_team = home if ph > pa else away
        favorite = fav_team
        fav_note = "Net favori" if margin >= 0.30 else "Hafif favori"

    patterns = find_patterns(d, lv_all, home, away, h2h, match_date, week)
    rev = reversal_analysis(lv_recent, home, away, h2h)
    surprises = surprise_signals(lv_all, h2h, home, away, week, rev, p6)

    return {
        "league_avg": {"home": mu_h, "away": mu_a}, "match_date": match_date, "week": week,
        "fixture_found": fx is not None, "ref_date": ref,
        "n_matches": {"home": sh.n, "away": sa.n, "h2h": len(h2h)},
        "strength": {"home": sh, "away": sa},
        "xg": {"home": lam_h, "away": lam_a, "total": lam_h + lam_a},
        "probs": {"home": ph, "draw": pd_, "away": pa},
        "model_probs": {"home": ms["p_home"], "draw": ms["p_draw"], "away": ms["p_away"]},
        "favorite": favorite, "favorite_note": fav_note,
        "components": {"model": ms["p_home"] - ms["p_away"], "form5": f5, "venue_form": fv,
                       "h2h": h2h_score, "tilt": tilt_score},
        "form": {"home": form_string(lv_all, home), "away": form_string(lv_all, away),
                 "ppg5": (ppg_h5, ppg_a5), "ppg_venue": (ppg_h_home, ppg_a_away)},
        "top_scores": top_scores, "score_scenario": score_scenario,
        "over": over, "over_poisson": ms["over"], "over_empirical": emp_over, "btts": ms["btts"],
        "ht": hm, "ft_pick": ft_pick, "surprises": surprises,
        "p6": {"final": p6, "poisson": ms["p6"], "empirical": emp_p6, "league": league_p6, "h2h": h2h_p6},
        "reversal": rev, "patterns": patterns, "h2h": h2h,
        "h2h_summary": {
            "n": len(h2h), "home_w": int((h2h.fh > h2h.fa).sum()), "draw": int((h2h.fh == h2h.fa).sum()),
            "away_w": int((h2h.fh < h2h.fa).sum()),
            "avg_goals": float((h2h.fh + h2h.fa).mean()) if len(h2h) else np.nan,
        },
    }


# ======================================================================
# 3) MATRIX KURALLARI
# ======================================================================
"""
matrix.py  —  "Matrix Analisti": fikstür içindeki ardışık örüntüleri tarar.

"Sürpriz" = 1/2 (ev İY önde, MS deplasman kazanır), 2/1 (tersi) veya 6+ gol.
Yedi kural (hepsi yalnızca hedef maçtan ÖNCEKİ verilerle çalışır):
  1  Temas Takımı (Köprü)   : iki takım aynı 3. takımla önce/sonra oynuyor
  2  Sandviç                : X'in önceki rakibi = Y'nin sonraki rakibi (veya tersi)
  3  Hafta/Takvim Uyumu     : başka bir takımın geçmiş sezonda aynı rakip-sonuç döngüsünü aynı haftalarda
                              yaşayıp sürpriz yapması; ayrıca aynı gün/ara dönüşü takvim tekrarı
  4  Tetikleyici (Kelebek)  : bir takımın ekstrem skoru sonrası, aynı skor geçmişte hedef takımda sürpriz doğurmuş
  5  H2H Döngüsü            : iki takımın maç sonuçlarının A-B-A sırası
  6  N-1 Birebirlik         : takımın önceki maçı (İY+MS skoru), geçmişte aynı skorun ardından sürpriz gelmiş
  7  Skor Döngüsü           : takımın son iki maçının skor dizisi, geçmişte aynı dizinin ardından sürpriz gelmiş
Her sinyalde örnek sayısı (n), sürpriz sayısı (k) ve "tesadüf olasılığı" (binom kuyruğu) verilir.
"""



MIN_N = 2          # oran temelli kurallarda asgari geçmiş örnek
MIN_RATE = 0.60    # ... ve asgari isabet oranı


def binom_sf(k: int, n: int, p: float) -> float:
    """P(X >= k), X ~ Binom(n, p): bu kadar isabetin salt şansla gelme olasılığı."""
    p = min(max(p, 1e-6), 1 - 1e-6)
    return float(sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k, n + 1)))


def _res(a, b) -> str:
    """Takım perspektifinden G/B/M (a: atılan, b: yenilen); bilinmiyorsa '?'."""
    if pd.isna(a) or pd.isna(b):
        return "?"
    return "G" if a > b else "B" if a == b else "M"


class Matrix:
    """Bir lig için maç tablosunu hazırlar ve verilen maç için 7 kuralı tarar."""

    def __init__(self, d: pd.DataFrame):
        d = d.sort_values("date").reset_index(drop=True).copy()
        d["mid"] = d.index
        tot = d.FTHG + d.FTAG
        htk = d.HTHG.notna()
        rev12 = htk & (d.HTHG > d.HTAG) & (d.FTHG < d.FTAG)
        rev21 = htk & (d.HTHG < d.HTAG) & (d.FTHG > d.FTAG)
        d["ev"] = np.where(d.played & (tot >= 6), "6+", np.where(rev12, "1/2", np.where(rev21, "2/1", "")))
        d["surprise"] = d.ev != ""
        self.d = d
        self.base = float(d[d.played].surprise.mean())

        # --- takım bazlı uzun tablo (fikstürler dahil: "sonraki rakip" için) ---
        def side(c_team, c_opp, venue, gf, ga, hf, ha):
            return pd.DataFrame({"mid": d.mid, "date": d.date, "season": d.season, "week": d.week,
                                 "team": d[c_team], "opp": d[c_opp], "venue": venue, "played": d.played,
                                 "gf": d[gf], "ga": d[ga], "hf": d[hf], "ha": d[ha], "ev": d.ev,
                                 "surprise": d.surprise})
        L = pd.concat([side("home", "away", "H", "FTHG", "FTAG", "HTHG", "HTAG"),
                       side("away", "home", "A", "FTAG", "FTHG", "HTAG", "HTHG")])
        L = L.sort_values(["team", "date", "mid"]).reset_index(drop=True)
        g = L.groupby("team")
        L["prev_opp"], L["next_opp"] = g.opp.shift(1), g.opp.shift(-1)
        L["next_mid"], L["prev_mid"] = g.mid.shift(-1), g.mid.shift(1)
        L["htres"] = [_res(a, b) for a, b in zip(L.hf, L.ha)]
        L["ftres"] = [_res(a, b) if p else "?" for a, b, p in zip(L.gf, L.ga, L.played)]
        # takım perspektifinden olay etiketi: geriden dönüp kazandı / önde iken kaybetti / 6+ gol
        hk = L.hf.notna()
        comeback = hk & (L.hf < L.ha) & (L.gf > L.ga)
        lostlead = hk & (L.hf > L.ha) & (L.gf < L.ga)
        six = L.played & ((L.gf + L.ga) >= 6)
        L["tpev"] = np.where(comeback, "COMEBACK", np.where(lostlead, "LOST_LEAD", np.where(six, "SIX", "")))
        self.L = L
        self.by_team = {t: x.reset_index(drop=True) for t, x in L.groupby("team")}
        Lh, La = L[L.venue == "H"].set_index("mid"), L[L.venue == "A"].set_index("mid")
        for c, src in (("pH", Lh.prev_opp), ("nH", Lh.next_opp), ("pA", La.prev_opp), ("nA", La.next_opp)):
            d[c] = d.mid.map(src)

        # --- köprü / sandviç indeksi: (tür, Z takımı) -> o yapıdaki geçmiş maçlar ---
        recs = []
        for r in d[d.played].itertuples():
            z1, z2 = self._z(r.pH, r.nH, r.pA, r.nA, r.home, r.away)
            recs += [("Temas", z, r.mid) for z in z1] + [("Sandviç", z, r.mid) for z in z2]
        zx = pd.DataFrame(recs, columns=["kind", "z", "mid"])
        zx = zx.join(d[["date", "surprise", "ev"]], on="mid") if len(zx) else zx
        self.zg = {k: v.sort_values("date") for k, v in zx.groupby(["kind", "z"])} if len(zx) else {}

        # --- Zaman çizelgesi indeksi: (Aşama1 rakibi, Aşama1 olayı, Aşama4 rakibi) -> [(takım, konum i)] ---
        # i = "patlama" maçının konumu; i-1 tetikleyici, i+1 köprü (araya alınan maç), i+2 hedef rakip
        self.tl_idx: dict = {}
        self.tl_n_pos = 0
        self.tl_ev_count: dict = {}
        for t, x in self.by_team.items():
            opp, pl, tp = x.opp.values, x.played.values, x.tpev.values
            for i in range(1, len(x) - 2):
                if pl[i - 1] and pl[i] and pl[i + 1] and pl[i + 2]:
                    self.tl_n_pos += 1
                    self.tl_ev_count[tp[i - 1]] = self.tl_ev_count.get(tp[i - 1], 0) + 1
                    if tp[i - 1]:
                        self.tl_idx.setdefault((opp[i - 1], tp[i - 1], opp[i + 2]), []).append((t, i))
        self._notes: list[str] = []

        # (takım, sezon, hafta) -> satır ; (rakip, hafta) -> o rakiple o haftada oynayanlar
        pl = L[L.played & L.week.notna()]
        self.wk = {(t, s, w): row for (t, s, w), row in
                   ((k, v) for k, v in zip(zip(pl.team, pl.season, pl.week), pl.itertuples()))}
        self.wkopp: dict = {}
        for row in pl.itertuples():
            self.wkopp.setdefault((row.opp, row.week), []).append(row)

    # ------------------------------------------------------------------ yardımcılar
    @staticmethod
    def _z(pH, nH, pA, nA, h, a):
        """Temas (ikisi de önce / ikisi de sonra aynı rakip) ve sandviç (önce-sonra) takım kümeleri."""
        ex = {None, h, a, np.nan}
        z1 = ({pH} & {pA}) | ({nH} & {nA})
        z2 = ({pH} & {nA}) | ({pA} & {nH})
        return {z for z in z1 if isinstance(z, str)} - ex, {z for z in z2 if isinstance(z, str)} - ex

    def _fmt(self, mid: int) -> str:
        r = self.d.loc[mid]
        ht = f" (İY {int(r.HTHG)}-{int(r.HTAG)})" if pd.notna(r.HTHG) else ""
        return f"{r.date:%d.%m.%Y} {r.home} {int(r.FTHG)}-{int(r.FTAG)} {r.away}{ht}"

    @staticmethod
    def _pick(evs) -> str:
        """Analog olayların çoğunluğundan net beklenti metni."""
        evs = [e for e in evs if e]
        if not evs:
            return "1/2 veya 2/1"
        top = max(set(evs), key=evs.count)
        return "+6 Gol" if top == "6+" else top

    def _nb(self, team: str, date):
        rows = self.by_team.get(team)
        if rows is None:
            return None, None
        prev = rows[rows.date < date]
        nxt = rows[rows.date > date]
        return (prev.iloc[-1] if len(prev) else None), (nxt.iloc[0] if len(nxt) else None)

    # ------------------------------------------------------------------ ana tarama
    def scan(self, home: str, away: str, date, season: str, week: float | None) -> list[dict]:
        """Hedef maç için tüm kuralları tarar; sinyalleri 'tesadüf olasılığı'na göre sıralı döndürür."""
        date = pd.Timestamp(date)
        ctx = {"home": home, "away": away, "date": date, "season": season, "week": week}
        sig: list[dict] = []
        self._notes = []
        for fn in (self.r_bridge, self.r_copy_week, self.r_calendar, self.r_butterfly,
                   self.r_h2h_cycle, self.r_n1, self.r_score_cycle, self.r_timeline):
            try:
                sig += fn(ctx)
            except (KeyError, IndexError, ValueError):
                continue
        return sorted(sig, key=lambda s: s["chance"])

    # ------------------------------------------------------------------ 1 & 2
    def r_bridge(self, c) -> list[dict]:
        pH, nH = self._nb(c["home"], c["date"]); pA, nA = self._nb(c["away"], c["date"])
        g = lambda r, k: (r[k] if r is not None else None)
        z1, z2 = self._z(g(pH, "opp"), g(nH, "opp"), g(pA, "opp"), g(nA, "opp"), c["home"], c["away"])
        out = []
        for kind, zs in (("Temas", z1), ("Sandviç", z2)):
            for z in zs:
                grp = self.zg.get((kind, z))
                if grp is None:
                    continue
                grp = grp[grp.date < c["date"]]
                n, k = len(grp), int(grp.surprise.sum())
                recent = grp[grp.surprise & (grp.date >= c["date"] - pd.Timedelta(days=365 * 3))]
                if k < 1 or (len(recent) < 1 and k < 2):
                    continue
                last = grp[grp.surprise].tail(2)
                title = ("TEMAS TAKIMI" if kind == "Temas" else "SANDVİÇ SİMETRİSİ") + f": {z.upper()}"
                out.append({
                    "rule": 1 if kind == "Temas" else 2, "title": title,
                    "past": f"Daha önce {z} ile aynı {kind.lower()} yapısına sahip {n} maçtan {k}'inde sürpriz çıktı. "
                            "Son örnekler: " + " | ".join(f"{self._fmt(m)} → {e}" for m, e in zip(last.mid, last.ev)),
                    "now": f"{c['home']} ve {c['away']} de {z} üzerinden aynı {kind.lower()} yapısını kuruyor "
                           f"(önceki rakipler: {g(pH,'opp')} / {g(pA,'opp')}; sonraki rakipler: {g(nH,'opp')} / {g(nA,'opp')}).",
                    "extra": f"Bu yapıdaki genel sürpriz oranı %{100*k/n:.0f} (lig tabanı %{100*self.base:.0f}).",
                    "pick": self._pick(list(last.ev)), "n": n, "k": k,
                    "chance": binom_sf(k, n, self.base)})
        return out

    # ------------------------------------------------------------------ 3a: kusursuz hafta kopyası
    def r_copy_week(self, c) -> list[dict]:
        w = c["week"]
        if w is None or pd.isna(w):
            return []
        out = []
        for T, O in ((c["home"], c["away"]), (c["away"], c["home"])):
            chain = []
            for k in range(1, 5):
                r = self.wk.get((T, c["season"], w - k))
                if r is None:
                    break
                chain.append(r)
            if not chain:
                continue
            for cand in self.wkopp.get((O, w), []):
                if cand.season == c["season"] or cand.team == T or cand.date >= c["date"] or not cand.surprise:
                    continue
                m = 0
                for k, r in enumerate(chain, start=1):
                    s = self.wk.get((cand.team, cand.season, w - k))
                    if s is None or s.opp != r.opp or (s.htres, s.ftres) != (r.htres, r.ftres):
                        break
                    m += 1
                if m >= 1:
                    steps = [f"{w-k:.0f}. hf: {chain[k-1].opp} ({chain[k-1].htres}/{chain[k-1].ftres})" for k in range(m, 0, -1)]
                    out.append({
                        "rule": 3, "title": f"MÜKEMMEL HAFTA UYUMU: {T.upper()} ≈ {cand.team.upper()} ({cand.season})",
                        "past": f"{cand.season} sezonunda {cand.team}, " + "; ".join(steps) + f" döngüsünün ardından "
                                f"{w:.0f}. haftada {O} ile oynadı: {self._fmt(cand.mid)} → {cand.ev}.",
                        "now": f"{T} bu sezon aynı rakipleri ve aynı sonuç döngüsünü ({m} hafta) birebir kopyaladı; "
                               f"şimdi {w:.0f}. haftada {O} ile oynuyor.",
                        "extra": f"Eşleşen ardışık hafta sayısı: {m}.",
                        "pick": self._pick([cand.ev]), "n": m, "k": m,
                        "chance": self.base ** m})   # m ardışık haftanın tesadüfen eşleşme olasılığı (yaklaşık)
        return out

    # ------------------------------------------------------------------ 3b: takvim örüntüsü (gün + ara dönüşü)
    def r_calendar(self, c) -> list[dict]:
        out = []
        for T, venue in ((c["home"], "H"), (c["away"], "A")):
            rows = self.by_team.get(T)
            if rows is None:
                continue
            gap = rows.date.diff().dt.days
            rows = rows.assign(gap=gap, dow=rows.date.dt.dayofweek)
            prev = rows[rows.date < c["date"]]
            if prev.empty:
                continue
            tgt_gap = (c["date"] - prev.date.iloc[-1]).days
            if tgt_gap < 12:
                continue                      # ara dönüşü değilse bu kural geçerli değil
            h = rows[(rows.played) & (rows.date < c["date"]) & (rows.venue == venue)
                     & (rows.dow == c["date"].dayofweek) & (rows.gap >= 12)]
            n, k = len(h), int(h.surprise.sum())
            if n >= 3 and k >= 3 and k / n >= MIN_RATE:
                gun = ["Pazartesi", "Salı", "Çarşamba", "Perşembe", "Cuma", "Cumartesi", "Pazar"][c["date"].dayofweek]
                out.append({
                    "rule": 3, "title": f"TAKVİM ÖRÜNTÜSÜ: {T.upper()} — ARADAN SONRA {gun.upper()}",
                    "past": f"{T}, {n} kez uzun aradan (12+ gün) sonra {gun} günü {'evinde' if venue=='H' else 'deplasmanda'} "
                            f"oynadı ve {k}'inde sürpriz çıktı: " + " | ".join(f"{self._fmt(m)} → {e}" for m, e in zip(h[h.surprise].mid.tail(3), h[h.surprise].ev.tail(3))),
                    "now": f"Bu maç da {tgt_gap} gün aranın ardından {gun} günü oynanıyor.",
                    "extra": f"Takvim koşulundaki isabet %{100*k/n:.0f} (lig tabanı %{100*self.base:.0f}).",
                    "pick": self._pick(list(h[h.surprise].ev)), "n": n, "k": k, "chance": binom_sf(k, n, self.base)})
        return out

    # ------------------------------------------------------------------ 4: kelebek etkisi
    def r_butterfly(self, c) -> list[dict]:
        d = self.d
        recent = d[d.played & (d.date < c["date"]) & (d.date >= c["date"] - pd.Timedelta(days=10))]
        gdv = (recent.FTHG - recent.FTAG).abs()
        recent = recent[(gdv >= 4) | ((recent.FTHG + recent.FTAG) >= 7)]
        out, base2 = [], 1 - (1 - self.base) ** 2
        for r in recent.itertuples():
            loser, lg, gg = (r.home, r.FTHG, r.FTAG) if r.FTHG < r.FTAG else (r.away, r.FTAG, r.FTHG)
            if r.FTHG == r.FTAG:
                continue
            if loser in (c["home"], c["away"]):
                continue
            rows = self.by_team.get(loser)
            occ = rows[rows.played & (rows.date < r.date) & (rows.gf == lg) & (rows.ga == gg)]
            for Y in (c["home"], c["away"]):
                hits, n, ex = 0, 0, []
                for o in occ.itertuples():
                    if pd.isna(o.week):
                        continue
                    ys = [self.wk.get((Y, o.season, o.week + i)) for i in (0, 1)]
                    ys = [y for y in ys if y is not None]
                    if not ys:
                        continue
                    n += 1
                    hit = [y for y in ys if y.surprise]
                    if hit:
                        hits += 1; ex.append((o, hit[0]))
                if n >= MIN_N and hits >= 2 and hits / n >= MIN_RATE:
                    o, y = ex[-1]
                    out.append({
                        "rule": 4, "title": f"TETİKLEYİCİ SKOR: {loser.upper()} {int(lg)}-{int(gg)} → {Y.upper()}",
                        "past": f"{loser} her {int(lg)}-{int(gg)}'lik skor aldığında, o hafta veya sonraki hafta {Y} "
                                f"{n} kezin {hits}'inde sürpriz çıkardı. Son örnek: {o.season} {o.week:.0f}. hf {loser}, "
                                f"ardından {Y}: {self._fmt(y.mid)} → {y.ev}.",
                        "now": f"{loser}, {r.date:%d.%m.%Y} tarihinde {self._fmt(r.mid)} maçında bu tetikleyici skoru aldı; "
                               f"sıradaki maç {c['home']} - {c['away']}.",
                        "extra": f"Tetikleyici sonrası iki maçlık pencerede isabet %{100*hits/n:.0f} (tabana göre %{100*base2:.0f}).",
                        "pick": self._pick([y.ev for _, y in ex]), "n": n, "k": hits,
                        "chance": binom_sf(hits, n, base2)})
        return out

    # ------------------------------------------------------------------ 5: H2H döngüsü
    def r_h2h_cycle(self, c) -> list[dict]:
        d = self.d
        m = d[d.played & (d.date < c["date"]) & (((d.home == c["home"]) & (d.away == c["away"]))
                                                  | ((d.home == c["away"]) & (d.away == c["home"])))].tail(6)
        if len(m) < 3:
            return []
        code = lambda e: "N" if e == "" else ("S" if e == "6+" else "R")
        cs = [code(e) for e in m.ev]
        a, b, a2 = cs[-3], cs[-2], cs[-1]
        if a == a2 and a != b and "N" not in (a, b):
            seq = " → ".join(f"{e or 'normal'} ({int(x)}-{int(y)})" for e, x, y in zip(m.ev, m.FTHG, m.FTAG))
            pickev = list(m.ev)[-2]
            return [{"rule": 5, "title": f"H2H DÖNGÜSÜ: {c['home'].upper()} – {c['away'].upper()}",
                     "past": f"Son maçların sırası: {seq}. Sürprizler {'/'.join(list(m.ev)[-3:])} biçiminde dönüşümlü.",
                     "now": "Döngü sırasına göre bir sonraki halka bu maç.",
                     "extra": "Sırayı bozan bir normal sonuç yok; son üç maçın üçü de sürpriz.",
                     "pick": self._pick([pickev]), "n": 3, "k": 3, "chance": self.base ** 3}]
        return []

    # ------------------------------------------------------------------ 6: N-1 birebirliği
    def r_n1(self, c) -> list[dict]:
        out = []
        for T in (c["home"], c["away"]):
            rows = self.by_team.get(T)
            prev = rows[rows.date < c["date"]]
            if prev.empty:
                continue
            last = prev.iloc[-1]
            if not last.played or pd.isna(last.hf):
                continue
            key = (last.hf, last.ha, last.gf, last.ga)
            occ = rows[rows.played & (rows.date < last.date) & (rows.hf == key[0]) & (rows.ha == key[1])
                       & (rows.gf == key[2]) & (rows.ga == key[3])]
            nxt = [self.d.loc[int(m)] for m in occ.next_mid.dropna() if self.d.loc[int(m)].played]
            n, k = len(nxt), sum(bool(x.surprise) for x in nxt)
            if n >= MIN_N and k >= 2 and k / n >= MIN_RATE:
                hits = [x for x in nxt if x.surprise]
                out.append({
                    "rule": 6, "title": f"N-1 BİREBİRLİK: {T.upper()}",
                    "past": f"{T} geçmişte aynı N-1 senaryosunu (İY {int(key[0])}-{int(key[1])}, MS {int(key[2])}-{int(key[3])}; "
                            f"kendi perspektifinden) {n} kez yaşadı; bir sonraki maçta {k} kez sürpriz çıktı: "
                            + " | ".join(f"{self._fmt(int(x.mid))} → {x.ev}" for x in hits[-2:]),
                    "now": f"{T}'in son maçı yine İY {int(key[0])}-{int(key[1])}, MS {int(key[2])}-{int(key[3])} bitti "
                           f"({last.date:%d.%m.%Y}, rakip {last.opp}). Sıradaki maç bu.",
                    "extra": f"N-1 sonrası sürpriz oranı %{100*k/n:.0f} (tabana göre %{100*self.base:.0f}).",
                    "pick": self._pick([x.ev for x in hits]), "n": n, "k": k, "chance": binom_sf(k, n, self.base)})
        return out

    # ------------------------------------------------------------------ 7: skor döngüsü (ön koşul)
    def r_score_cycle(self, c) -> list[dict]:
        out = []
        for T in (c["home"], c["away"]):
            rows = self.by_team.get(T)
            pl = rows[rows.played & (rows.date < c["date"])].reset_index(drop=True)
            if len(pl) < 6:
                continue
            seq = list(zip(pl.gf, pl.ga))
            key = (seq[-2], seq[-1])
            hits, n, ex = 0, 0, []
            for i in range(1, len(pl) - 1):
                if (seq[i - 1], seq[i]) == key:
                    nxt_mid = pl.next_mid.iloc[i]
                    if pd.isna(nxt_mid) or not self.d.loc[int(nxt_mid)].played:
                        continue
                    n += 1
                    if self.d.loc[int(nxt_mid)].surprise:
                        hits += 1; ex.append(self.d.loc[int(nxt_mid)])
            if n >= MIN_N and hits >= 2 and hits / n >= MIN_RATE:
                out.append({
                    "rule": 7, "title": f"SKOR DÖNGÜSÜ: {T.upper()} {int(key[0][0])}-{int(key[0][1])} → {int(key[1][0])}-{int(key[1][1])}",
                    "past": f"{T} art arda {int(key[0][0])}-{int(key[0][1])} ve {int(key[1][0])}-{int(key[1][1])} "
                            f"(kendi perspektifinden) skorlarını {n} kez gördü; sonraki maçta {hits} kez sürpriz çıktı: "
                            + " | ".join(f"{self._fmt(int(x.mid))} → {x.ev}" for x in ex[-2:]),
                    "now": f"{T}'in son iki maçı tam bu skor dizisiyle bitti; sıradaki maç bu.",
                    "extra": f"Skor dizisi sonrası isabet %{100*hits/n:.0f} (tabana göre %{100*self.base:.0f}).",
                    "pick": self._pick([x.ev for x in ex]), "n": n, "k": hits, "chance": binom_sf(hits, n, self.base)})
        return out

    # ------------------------------------------------------------------ 8: zaman çizelgesi kopyası
    _TP_TXT = {"COMEBACK": "geriden gelip kazandı", "LOST_LEAD": "ilk yarıyı önde kapatıp kaybetti", "SIX": "6+ gollü maç oynadı"}

    @staticmethod
    def _tp_label(row) -> str:
        """Takım perspektifinden kısa etiket: '1/2 yaptı', '5-1 kazandı' gibi."""
        ev = row.ev if isinstance(row.ev, str) else ""
        sc = f"{int(row.gf)}-{int(row.ga)}"
        res = "kazandı" if row.gf > row.ga else "berabere kaldı" if row.gf == row.ga else "kaybetti"
        return f"{ev + ' yaptı, ' if ev else ''}{sc} {res}"

    def r_timeline(self, c) -> list[dict]:
        """
        4 aşamalı döngü: (1) rakip O1 ile sürpriz  -> (2) patlama maçı  -> (3) araya 1 maç  -> (4) Z ile maç.
        Hedef maç, B takımının (2). aşamasıdır; geçmişte A takımı aynı O1 sürprizini ve aynı Z'yi yaşadıysa,
        A'nın (2). aşama sonucu B'nin bu maçı için tahmin edilir. Gelecek fikstür (j+1, j+2) veride olmalıdır.
        """
        out = []
        for T, O, venue in ((c["home"], c["away"], "H"), (c["away"], c["home"], "A")):
            rows = self.by_team.get(T)
            if rows is None:
                continue
            j = int(rows.date.values.searchsorted(np.datetime64(c["date"])))
            if j < 1 or j >= len(rows) or rows.opp.iloc[j] != O:
                self._notes.append(f"{T}: hedef maç fikstür listesinde yok; zaman çizelgesi kontrol edilemedi.")
                continue
            if j + 2 >= len(rows):
                self._notes.append(f"{T}: sonraki 2 hafta fikstürü veride yok (API ile güncelleyin); çizelge doğrulanamadı.")
                continue
            prev = rows.iloc[j - 1]
            if not prev.played or not prev.tpev:
                continue
            key = (prev.opp, prev.tpev, rows.opp.iloc[j + 2])
            cands = []
            for t2, i in self.tl_idx.get(key, []):
                x = self.by_team[t2]
                if x.date.iloc[i + 2] >= prev.date or (t2 == T and x.date.iloc[i] >= prev.date):
                    continue                      # analog tamamen geçmişte olmalı
                cands.append((t2, i))
            if not cands:
                continue
            # tesadüf olasılığı: beklenen rastlantısal eşleşme sayısı (Poisson)
            n_opp = max(len(set(rows.opp)) - 1, 2)
            p_ev = self.tl_ev_count.get(prev.tpev, 0) / max(self.tl_n_pos, 1)
            lam = self.tl_n_pos * p_ev * (1 / n_opp) ** 2
            chance = 1 - math.exp(-lam)
            cands.sort(key=lambda ti: self.by_team[ti[0]].date.iloc[ti[1]], reverse=True)
            for t2, i in cands[:2]:
                x = self.by_team[t2]
                a1, a2, a3, a4 = x.iloc[i - 1], x.iloc[i], x.iloc[i + 1], x.iloc[i + 2]
                tot = int(a2.gf + a2.ga); margin = int(abs(a2.gf - a2.ga))
                notable = tot >= 4 or margin >= 3 or a2.tpev != ""
                if not notable:
                    continue
                # B'nin perspektifinden hedef maç skor tahmini
                gf, ga = int(a2.gf), int(a2.ga)
                score_txt = f"{c['home']} {gf if venue == 'H' else ga}-{ga if venue == 'H' else gf} {c['away']}"
                if tot >= 6: pick, pick_txt = "+6 Gol", "+6 Gol / +4.5 Üst"
                elif tot >= 5: pick, pick_txt = "+4.5 Üst", "+4.5 Üst (5+ gol)"
                elif tot >= 4: pick, pick_txt = "+3.5 Üst", "+3.5 Üst (4+ gol)"
                elif margin >= 3: pick, pick_txt = "Büyük Fark", "Büyük fark"
                elif a2.tpev == "COMEBACK": pick, pick_txt = ("2/1" if venue == "H" else "1/2"), None
                elif a2.tpev == "LOST_LEAD": pick, pick_txt = ("1/2" if venue == "H" else "2/1"), None
                else: pick, pick_txt = "Skor", "Skor kopyası"
                pick_txt = pick_txt or pick
                extra_match = sum([x.opp.iloc[i - 2] == rows.opp.iloc[j - 2] if i >= 2 and j >= 2 else False,
                                   x.opp.iloc[i - 3] == rows.opp.iloc[j - 3] if i >= 3 and j >= 3 else False])
                fmt_row = lambda r: f"{r.opp} ({self._tp_label(r)})"
                flow_orig = (f"{fmt_row(a1)} ➔ {fmt_row(a2)} ➔ [1 Maç Boşluk: {a3.opp}] ➔ {a4.opp} ile oynadı.")
                flow_now = (f"{fmt_row(prev)} ➔ [ŞİMDİKİ MAÇ: {O}] ➔ [1 Maç Boşluk: {rows.opp.iloc[j + 1]}] ➔ "
                            f"{rows.opp.iloc[j + 2]} ile oynayacak.")
                out.append({
                    "rule": 8, "kind": "timeline", "title": "ZAMAN ÇİZELGESİ KOPYALAMA ALARMI",
                    "orig_team": t2, "orig_season": a2.season, "new_team": T,
                    "flow_orig": f"Geçmişteki Orijinal Döngü ({t2}, {a2.season}): {flow_orig}",
                    "flow_now": f"Bu Yılın Kopyalanan Döngüsü ({T}): {flow_now}",
                    "past": f"{t2} ({a2.season}) bu döngüde {a1.opp} sürprizinden sonra {self._fmt(int(a2.mid))} oynadı.",
                    "now": f"{T} aynı {prev.opp} sürprizini yaşadı ve {rows.opp.iloc[j + 2]} ile oynayacak.",
                    "extra": f"Ek eşleşen önceki rakip sayısı: {extra_match}. Rastlantısal eşleşme beklentisi: {lam:.2f}.",
                    "pick": pick, "pick_text": pick_txt, "score_pred": score_txt, "orig_score": f"{gf}-{ga}",
                    "n": len(cands), "k": len(cands), "chance": chance, "venue": venue,
                    "pred_total": tot, "pred_margin": margin})
        return out


# ======================================================================
# 4) V2: KARAKTER / PATLAMA / RADAR
# ======================================================================
"""
v2.py — Matrix V2 eklentileri
  * character()    : Geri dönüş & kırılganlık faktörü (1/2 ve 2/1 için takım karakteri)
  * burst()        : Gol sıkışması / patlama noktası (+6 gol için)
  * burst_history(): patlama koşulunun geçmişteki gerçekleşme oranı (tüm liglerden)
  * ucl_cross()    : Şampiyonlar Ligi çapraz fikstür (hafta sonu lig maçı <-> hafta içi ŞL maçı) örüntüleri
  * build_radar()  : tüm sinyalleri tek bir "Matrix Sürpriz Radarı"nda birleştirir (seviye, potansiyel, kanıt gücü)

Not: veri setinde şut/xG yoktur; "xG sapması" burada GOL temelli beklenen-gol (uzun dönem toplam gol ortalaması)
ile son iki maçın gerçekleşen golü arasındaki fark olarak hesaplanır.
"""



# ----- Eşikler (backtest ile kalibre edilir; gerekirse değiştirin) -----
CHAR_LIFT = 1.25          # karakter uyumu: her iki oran lig tabanının en az 1.25 katı
CHAR_STRONG = 1.50        # güçlü uyum
CHAR_MIN_N = 8            # karakter oranı için asgari örnek (HT'de önde/geride kalınan maç)
BURST_TEND = 0.20         # her iki takımın uzun dönem toplam gol ort. >= lig ort. + 0.20
BURST_LAST_MAX = 1        # son iki maçın her birinde toplam gol <= 1 (0-0, 1-0, 0-1)
BURST_MIN_XG = 2.6        # modelin beklenen toplam golü en az bu kadar olmalı


# --------------------------------------------------------------------------- #
# 1) Karakter: kırılganlık x geri dönüş
# --------------------------------------------------------------------------- #
def _cond(lv: pd.DataFrame, team: str, venue: str | None, side: str) -> tuple[int, int]:
    """
    side='lead' : HT'de önde olup maçı kazanamama (puan kaybı) sayısı / HT'de önde olunan maç sayısı
    side='trail': HT'de geride olup maçı kazanma sayısı / HT'de geride olunan maç sayısı
    """
    t = lv[(lv.team == team) & lv.ht_has]
    if venue:
        t = t[t.venue == venue]
    if side == "lead":
        sub = t[t.htgf > t.htga]
        return int((sub.gf <= sub.ga).sum()), len(sub)
    sub = t[t.htgf < t.htga]
    return int((sub.gf > sub.ga).sum()), len(sub)


def _rate_block(lv: pd.DataFrame, team: str, venue: str, side: str) -> dict:
    """Takımın oranı (saha bazlı; örnek azsa tüm sahalar) + lig tabanı + büzülmüş oran, lift ve tesadüf olasılığı."""
    k, n = _cond(lv, team, venue, side)
    scope = venue
    if n < CHAR_MIN_N:
        k, n = _cond(lv, team, None, side); scope = "tümü"
    # lig tabanı (aynı saha oryantasyonu)
    allv = lv[lv.ht_has & (lv.venue == venue)]
    if side == "lead":
        sub = allv[allv.htgf > allv.htga]; base = float((sub.gf <= sub.ga).mean())
    else:
        sub = allv[allv.htgf < allv.htga]; base = float((sub.gf > sub.ga).mean())
    shr = (k + 8 * base) / (n + 8)
    return {"k": k, "n": n, "rate": k / n if n else np.nan, "shrunk": shr, "base": base,
            "lift": shr / base if base else 1.0, "chance": binom_sf(k, n, base) if n else 1.0, "scope": scope}


def character(lv: pd.DataFrame, home: str, away: str) -> dict:
    """
    1/2: ev sahibi 'ilk yarıyı önde kapatıp puan kaybeder' (kırılgan)  VE  deplasman 'geriye düştüğü maçı çevirir'.
    2/1: deplasman kırılgan  VE  ev sahibi geri dönüşçü.
    """
    out = {}
    for d, fb, cb in (("1/2", (home, "H"), (away, "A")), ("2/1", (away, "A"), (home, "H"))):
        frag = _rate_block(lv, fb[0], fb[1], "lead")
        comeb = _rate_block(lv, cb[0], cb[1], "trail")
        ok = min(frag["lift"], comeb["lift"]) >= CHAR_LIFT and frag["n"] >= CHAR_MIN_N and comeb["n"] >= CHAR_MIN_N
        out[d] = {"frag": frag, "comeback": comeb, "frag_team": fb[0], "comeback_team": cb[0], "match": ok,
                  "strong": ok and min(frag["lift"], comeb["lift"]) >= CHAR_STRONG}
    return out


# --------------------------------------------------------------------------- #
# 2) Gol sıkışması / patlama noktası (+6)
# --------------------------------------------------------------------------- #
def burst(lv: pd.DataFrame, home: str, away: str, xg_total: float, league_mean_tg: float) -> dict:
    """
    Her takım için: uzun dönem toplam gol ortalaması (son 2 maç hariç, zaman ağırlıklı) ve son iki maçın toplam golü.
    Sıkışma: iki takım da yüksek-gollü eğilimde (ort >= lig + BURST_TEND) ama ikisinin de son 2 maçı kısır (tg <= 1).
    xG sapması z: (son 2 maç ortalaması - uzun dönem ortalama) / (std / sqrt(2)); negatif ve büyük = sıkışma.
    """
    info = {}
    for team in (home, away):
        t = lv[lv.team == team].sort_values("date")
        last2 = t.tail(2)
        hist = t.iloc[:-2]
        if len(hist) < 10 or len(last2) < 2:
            info[team] = None
            continue
        mu = float((hist.tg * hist.w).sum() / hist.w.sum())
        sd = float(hist.tg.std()) or 1.0
        l2 = [int(x) for x in last2.tg]
        info[team] = {"mu": mu, "last2": l2, "z": (np.mean(l2) - mu) / (sd / np.sqrt(2)),
                      "tend": mu >= league_mean_tg + BURST_TEND, "compress": max(l2) <= BURST_LAST_MAX,
                      "scores": [f"{int(a)}-{int(b)}" for a, b in zip(last2.gf, last2.ga)]}
    ok = all(v is not None for v in info.values())
    flag = bool(ok and all(v["tend"] and v["compress"] for v in info.values()) and xg_total >= BURST_MIN_XG)
    return {"flag": flag, "teams": info, "league_mean": league_mean_tg}


def burst_history(df: pd.DataFrame, cutoff: pd.Timestamp | None = None) -> dict:
    """
    Patlama koşulunun (tüm liglerde, yalnızca maç öncesi bilgiyle) geçmişte kaç maçta oluştuğu ve kaçında 6+ gol çıktığı.
    cutoff verilirse yalnızca bu tarihten önceki maçlar kullanılır (backtest sızıntısını önlemek için).
    """
    d = df[df.played & (df.league != "ucl")].copy()
    if cutoff is not None:
        d = d[d.date < cutoff]
    d = d.sort_values("date").reset_index(drop=True)
    d["tg"] = d.FTHG + d.FTAG
    lm = d.groupby("league").tg.transform("mean")
    rows = []
    for side in ("home", "away"):
        x = pd.DataFrame({"i": d.index, "team": d[side], "league": d.league, "tg": d.tg, "date": d.date, "side": side})
        rows.append(x)
    L = pd.concat(rows).sort_values(["league", "team", "date", "i"])
    g = L.groupby(["league", "team"])
    cs, cn = g.tg.cumsum() - L.tg, g.cumcount()                 # bu maça kadar toplam / sayı
    prev1, prev2 = g.tg.shift(1), g.tg.shift(2)
    n_hist = cn - 2
    mu = (cs - prev1 - prev2) / n_hist.where(n_hist >= 10)       # son 2 maç hariç uzun dönem ortalama
    L["mu"], L["l2max"] = mu, np.maximum(prev1, prev2)
    H = L[L.side == "home"].set_index("i").reindex(d.index)
    A = L[L.side == "away"].set_index("i").reindex(d.index)
    lmean = lm
    cond = ((H.mu >= lmean + BURST_TEND) & (A.mu >= lmean + BURST_TEND)
            & (H.l2max <= BURST_LAST_MAX) & (A.l2max <= BURST_LAST_MAX))
    six = d.tg >= 6
    n = int(cond.sum()); k = int((cond & six).sum())
    return {"n": n, "k": k, "rate": k / n if n else np.nan, "base": float(six.mean())}


# --------------------------------------------------------------------------- #
# 3) Şampiyonlar Ligi çapraz fikstür
# --------------------------------------------------------------------------- #
_PROPS = {
    "puan kaybetti": lambda r: r.pts < 3,
    "yenildi": lambda r: r.pts == 0,
    "sürpriz sonuç (1/2, 2/1 veya 6+) çıktı": lambda r: r.surprise,
}


def _dom_frame(dom: pd.DataFrame, team: str) -> pd.DataFrame:
    """Takımın yerel lig maçları (oynanmış), takım perspektifinden."""
    h = dom[dom.home == team]; a = dom[dom.away == team]
    f = pd.concat([
        pd.DataFrame({"date": h.date, "opp": h.away, "gf": h.FTHG, "ga": h.FTAG, "hf": h.HTHG, "ha": h.HTAG,
                      "ev": h.ev, "home": True}),
        pd.DataFrame({"date": a.date, "opp": a.home, "gf": a.FTAG, "ga": a.FTHG, "hf": a.HTAG, "ha": a.HTHG,
                      "ev": a.ev, "home": False}),
    ]).sort_values("date").reset_index(drop=True)
    f["pts"] = np.where(f.gf > f.ga, 3, np.where(f.gf == f.ga, 1, 0))
    f["surprise"] = f.ev != ""
    return f


def _tag_events(d: pd.DataFrame) -> pd.DataFrame:
    """Maç düzeyinde sürpriz etiketi (1/2, 2/1, 6+)."""
    d = d.copy()
    tot = d.FTHG + d.FTAG
    htk = d.HTHG.notna()
    r12 = htk & (d.HTHG > d.HTAG) & (d.FTHG < d.FTAG)
    r21 = htk & (d.HTHG < d.HTAG) & (d.FTHG > d.FTAG)
    d["ev"] = np.where(tot >= 6, "6+", np.where(r12, "1/2", np.where(r21, "2/1", "")))
    return d


def ucl_cross(df: pd.DataFrame, home: str, away: str, date: pd.Timestamp, stage: str | None) -> dict:
    """
    Hafta içi ŞL maçı ile hafta sonu lig maçları arasındaki çapraz örüntüler.
      (a/b) Takımın ŞL maçından hemen ÖNCEKİ / SONRAKİ (6 gün içindeki) lig maçlarında tekrar eden olumsuzluk/sürpriz
      (c)   Bu ŞL maçından önceki lig maçı geçmişte hangi sonuçla bitmişse, o durumdan sonra ŞL'de sürpriz sıklığı
    Aşama (Grup/Eleme) biliniyorsa önce o aşamanın ŞL maçlarına bakılır (örnek yetersizse tümüne).
    """
    played = df[df.played]
    ucl = _tag_events(played[played.league == "ucl"])
    dom = _tag_events(played[played.league != "ucl"])
    base_ucl = float((ucl.ev != "").mean()) if len(ucl) else 0.1
    sigs, notes = [], []
    for T in (home, away):
        du = _dom_frame(dom, T)
        if du.empty:
            continue
        mine = ucl[(ucl.home == T) | (ucl.away == T)]
        mine = mine[mine.date < date]
        if stage and len(mine[mine.stage == stage]) >= 6:
            mine = mine[mine.stage == stage]
            st_txt = f"{stage} aşaması "
        else:
            st_txt = ""
        dates = du.date.values
        def around(u, lo, hi):
            m = du[(du.date >= u - pd.Timedelta(days=lo)) & (du.date <= u + pd.Timedelta(days=hi))]
            return m
        prev_rows, next_rows, pairs = [], [], []
        for u, ev in zip(mine.date, mine.ev):
            p = around(u, 6, -1)            # u-6 .. u-1
            n_ = around(u, -1, 6)           # u+1 .. u+6
            p = p.iloc[-1] if len(p) else None
            n_ = n_.iloc[0] if len(n_) else None
            if p is not None: prev_rows.append(p)
            if n_ is not None: next_rows.append(n_)
            pairs.append((p, ev, u))
        base = {name: float(f(du).mean()) for name, f in _PROPS.items()}
        for label, rows in (("ŞL maçından hemen önceki hafta sonu lig maçında", prev_rows),
                            ("ŞL maçından hemen sonraki lig maçında", next_rows)):
            if len(rows) < 4:
                continue
            R = pd.DataFrame(rows)
            for name, f in _PROPS.items():
                k, n = int(f(R).sum()), len(R)
                ch = binom_sf(k, n, base[name])
                if k >= 3 and k / n >= max(0.5, 1.8 * base[name]) and ch <= 0.08:
                    sigs.append({"rule": "ŞL", "kind": "cross_info", "team": T,
                                 "title": f"ÇAPRAZ FİKSTÜR: {T.upper()}",
                                 "past": f"{T}, {st_txt}Şampiyonlar Ligi maçlarının {n}'inde {label} {k} kez {name} "
                                         f"(genel lig oranı %{100*base[name]:.0f}, burada %{100*k/n:.0f}).",
                                 "now": f"{T}'in bu ŞL maçı {date:%d.%m.%Y}; aynı kalıbın o haftaki lig maçında da görülmesi beklenebilir.",
                                 "extra": f"Örneklem: {n} ŞL maçı.", "pick": "1/2 veya 2/1" if "sürpriz" in name else "—",
                                 "n": n, "k": k, "chance": ch})
        # (c) bu maçtan önceki lig maçı bilinen duruma sahipse
        cur_prev = du[(du.date >= date - pd.Timedelta(days=6)) & (du.date < date)]
        if len(cur_prev) and len(pairs) >= 6:
            cp = cur_prev.iloc[-1]
            for name, f in _PROPS.items():
                if not bool(f(pd.DataFrame([cp])).iloc[0]):
                    continue
                sel = [(ev != "", ev) for p, ev, u in pairs if p is not None and bool(f(pd.DataFrame([p])).iloc[0])]
                n, k = len(sel), sum(x for x, _ in sel)
                ch = binom_sf(k, n, base_ucl) if n else 1.0
                if n >= 5 and k >= 3 and k / n >= 1.8 * base_ucl and ch <= 0.10:
                    evs = [e for x, e in sel if x]
                    top = max(set(evs), key=evs.count)
                    sigs.append({"rule": "ŞL", "kind": "cross", "team": T,
                                 "title": f"ÇAPRAZ FİKSTÜR (ŞL SÜRPRİZİ): {T.upper()}",
                                 "past": f"{T}, ŞL öncesindeki lig maçında {name} {n} durumun {k}'inde ŞL maçında sürpriz çıktı "
                                         f"(ŞL taban oranı %{100*base_ucl:.0f}, burada %{100*k/n:.0f}).",
                                 "now": f"{T}'in son lig maçı ({cp.date:%d.%m.%Y}, rakip {cp.opp}, {int(cp.gf)}-{int(cp.ga)}) tam bu durumla bitti: {name}.",
                                 "extra": f"En sık sürpriz türü: {top}.",
                                 "pick": "+6 Gol" if top == "6+" else top, "n": n, "k": k, "chance": ch})
        # güncel not: hafta sonu maçı sonucu
        if len(cur_prev):
            cp = cur_prev.iloc[-1]
            notes.append(f"{T}: ŞL maçından önceki lig maçı {cp.date:%d.%m.%Y} {cp.opp} {int(cp.gf)}-{int(cp.ga)}"
                         + (f" ({cp.ev})" if cp.ev else ""))
    return {"signals": sorted(sigs, key=lambda s: s["chance"]), "notes": notes, "ucl_base": base_ucl}


# --------------------------------------------------------------------------- #
# 4) Matrix Sürpriz Radarı
# --------------------------------------------------------------------------- #
TIER_LABEL = {2: "🔥🔥 YÜKSEK POTANSİYEL", 1: "🔥 POTANSİYEL", 0: "👀 İZLEMEDE"}
# Geriye dönük testten (backtest) ölçülen gerçekleşme oranları: (seviye, tür) -> (gerçekleşen, toplam uyarı)
TIER_STATS: dict = {
    ("1/2", 0): (3, 161), ("1/2", 1): (0, 63), ("1/2", 2): (0, 4),
    ("2/1", 0): (1, 147), ("2/1", 1): (0, 26), ("2/1", 2): (0, 1),
    ("6+", 0): (4, 41), ("6+", 1): (10, 57),
}
SEASONAL_NOTE = (
    "Geriye dönük test (18 lig, 28.544 maç, her maç yalnızca önceki sezonların verisiyle): aylık 6+ alışkanlığı uyarısı çıkan "
    "1.068 maçın %8,1'inde 6+ çıktı (taban %6,0) — hafif ama anlamlı bir artış. Aylık 1/2-2/1 uyarısı 468 maçta %5,6 "
    "(taban %4,4; fark istatistiksel olarak anlamsız). Sezon kapanışı 6+ uyarısı 78 maçta %12,8, 1/2-2/1 uyarısı 40 maçta %10 "
    "(örnek küçük). Ayın ilk/son maçı kuralı (6 uyarı) ve domino etkisi çok nadir tetiklendiği için ölçülemedi."
)
TIMELINE_NOTE = (
    "Geriye dönük test (18 lig, 47.827 maç, her maç yalnızca kendinden önceki veriyle): zaman çizelgesi alarmı 992 maçta "
    "çıktı. Bu maçlarda 4+ gol oranı %34 (genel taban %29), 6+ gol %6,9 (taban %5,7), tam skor tutturma %3,7. "
    "Yani döngü gollü maç ihtimalini hafif artırıyor, ama orijinal döngüdeki 5-1 gibi patlamalar ortalama 4,5 gollük "
    "orijinalden 2,9 gole geri çekiliyor: skor kopyası beklentisi tutmuyor."
)
BACKTEST_NOTE = (
    "Geriye dönük test (10 lig, 2.200 maç, her maç yalnızca kendinden önceki veriyle): "
    "+6 gol model olasılığı ≥%15 olan 54 maçın 10'unda (%18,5) 6+ çıktı (genel taban %6,4) — bu sinyal işe yarıyor. "
    "Buna karşılık 1/2 ve 2/1 için karakter uyumu ve Matrix örüntüleri ayırt edici olmadı: 1/2 karakter uyumlu 206 maçta "
    "3 isabet (%1,5; taban %2,1), 2/1 için 149 maçta 1. Sıkışma/patlama koşulu da 6+ ihtimalini artırmadı "
    "(tüm liglerde 29 maçta 1). Bu yüzden 1/2 ve 2/1 uyarılarını kesin bilgi değil, dikkat notu olarak okuyun."
)


def _combine_evidence(chances: list[float]) -> float:
    """Kanıt gücü = 1 - Π(tesadüf olasılıkları), %95 ile sınırlı (çoklu test nedeniyle iyimser olabilir)."""
    p = 1.0
    for c in chances:
        p *= min(max(c, 1e-4), 1.0)
    return min(0.95, 1 - p)


def build_radar(r: dict, home: str, away: str, msig: list[dict], ch: dict, bu: dict,
                burst_hist: dict, cross_sigs: list[dict]) -> list[dict]:
    """
    r: analysis.analyze() çıktısı. Her yön (1/2, 2/1, 6+) için seviye (0-2), model olasılığı (potansiyel),
    kanıt gücü ve gerekçeler döner. Seviye mantığı:
      🔥🔥 : (1/2, 2/1) karakter uyumu + aynı yönde ≥1 örüntü  |  (6+) patlama koşulu + aynı yönde ≥1 örüntü
      🔥  : güçlü karakter uyumu, ya da ≥2 örüntü, ya da model olasılığı eşik üstü / patlama koşulu tek başına
      👀  : tek örüntü ya da orta karakter uyumu
    """
    pool = [s for s in (msig + cross_sigs) if s.get("chance", 1) <= 0.10 and s.get("kind") != "cross_info"]
    items = []
    for d, key in (("1/2", "p12"), ("2/1", "p21")):
        c = ch[d]
        aligned = [s for s in pool if s["pick"] in (d, "1/2 veya 2/1")]
        reasons, chances = [], []
        if c["match"]:
            f, b = c["frag"], c["comeback"]
            reasons.append(
                f"Karakter: {c['frag_team']} ilk yarıyı önde kapatıp puan kaybetme %{100*f['rate']:.0f} "
                f"({f['k']}/{f['n']}, lig %{100*f['base']:.0f}) · {c['comeback_team']} geriye düştüğü maçı çevirme "
                f"%{100*b['rate']:.0f} ({b['k']}/{b['n']}, lig %{100*b['base']:.0f})")
            chances += [f["chance"], b["chance"]]
        for s in aligned:
            reasons.append(f"Örüntü ({s['rule']}): {s['title']}")
            chances.append(s["chance"])
        p = r["reversal"][key]
        if c["match"] and aligned:
            tier = 2
        elif c["strong"] or len(aligned) >= 2 or p >= REV_MIN:
            tier = 1
        elif c["match"] or aligned:
            tier = 0
        else:
            continue
        items.append({"dir": d, "tier": tier, "pot": float(p), "evid": _combine_evidence(chances),
                      "reasons": reasons, "signals": aligned, "char": c})
    # +6 gol
    aligned6 = [s for s in pool if s["pick"] == "+6 Gol"]
    reasons, chances = [], []
    p6 = r["p6"]["final"]
    pot = p6
    # Patlama koşulu yalnızca tarihsel veri onu destekliyorsa seviye yükseltir (aksi halde sadece bilgi notu)
    supported = bool(bu["flag"] and (burst_hist["n"] < 5 or burst_hist["rate"] >= 1.2 * burst_hist["base"]))
    if bu["flag"]:
        parts = []
        for tm, v in bu["teams"].items():
            parts.append(f"{tm}: uzun dönem {v['mu']:.2f} gol/maç, son 2 maç {v['scores'][0]} ve {v['scores'][1]}")
        reasons.append("Sıkışma/Patlama noktası — " + " · ".join(parts) + f" (lig ort. {bu['league_mean']:.2f}).")
        if burst_hist["n"]:
            reasons.append(f"Bu koşul geçmişte {burst_hist['n']} maçta oluştu, {burst_hist['k']}'inde 6+ gol çıktı "
                           f"(%{100*burst_hist['rate']:.1f}; tüm maçlarda taban %{100*burst_hist['base']:.1f}).")
            if supported:
                pot = max(p6, burst_hist["rate"])
                if burst_hist["n"] >= 5:
                    chances.append(binom_sf(burst_hist["k"], burst_hist["n"], burst_hist["base"]))
            else:
                reasons.append("⚠️ Tarihsel veri bu koşulun 6+ gol ihtimalini artırmadığını gösteriyor; seviye yükseltilmedi.")
    for s in aligned6:
        reasons.append(f"Örüntü ({s['rule']}): {s['title']}")
        chances.append(s["chance"])
    if supported and aligned6:
        tier6 = 2
    elif supported or len(aligned6) >= 2 or p6 >= P6_MIN:
        tier6 = 1
    elif aligned6:
        tier6 = 0
    elif bu["flag"]:
        tier6 = 0                       # bilgi amaçlı: koşul var ama tarihsel destek yok
    else:
        tier6 = None
    if tier6 is not None:
        items.append({"dir": "6+", "tier": tier6, "pot": float(pot), "evid": _combine_evidence(chances),
                      "reasons": reasons, "signals": aligned6, "char": None})
    return sorted(items, key=lambda x: (-x["tier"], -x["pot"]))


# ======================================================================
# 5) CANLI API
# ======================================================================
"""
live_data.py — football-data.org (v4) üzerinden güncel fikstür/sonuç çekme.

Ücretsiz plan (kayıt: https://www.football-data.org/client/register) şu ligleri kapsar:
PL, ELC, BL1, PD, SA, FL1, DED, PPL, BSA, CL (+ WC, EC). Dakikada 10 istek sınırı vardır.
Çekilen maçlar data/live_cache.csv dosyasına kaydedilir ve uygulama açılışında otomatik yüklenir.

Not: Bu modül bu geliştirme ortamında canlı API'ye bağlanılamadığı için yalnızca örnek yanıtla test edilmiştir.
"""



API = "https://api.football-data.org/v4"
CACHE = Path(__file__).parent / "data" / "live_cache.csv"

# football-data.org yarışma kodu -> bu projedeki lig kodu / görünen ad
COMPETITIONS = {
    "CL": ("ucl", "Şampiyonlar Ligi"), "PL": ("en1", "Premier League"), "ELC": ("en2", "Championship"),
    "BL1": ("de1", "Bundesliga"), "PD": ("es1", "La Liga"), "SA": ("it1", "Serie A"),
    "FL1": ("fr1", "Ligue 1"), "DED": ("nl1", "Eredivisie"), "PPL": ("pt1", "Primeira Liga"),
    "BSA": ("br1", "Brezilya Série A"),
}
CALENDAR_YEAR = {"BSA"}                      # takvim yılı sezonu kullanan ligler
FINISHED = {"FINISHED", "AWARDED"}
KNOCKOUT = {"PLAYOFFS", "LAST_16", "QUARTER_FINALS", "SEMI_FINALS", "FINAL", "LAST_32", "ROUND_OF_16"}


def current_season_year(today: pd.Timestamp | None = None) -> int:
    """Avrupa sezonu için başlangıç yılı (Temmuz'dan itibaren yeni sezon)."""
    t = today or pd.Timestamp.today()
    return t.year if t.month >= 7 else t.year - 1


def _season_label(code: str, start_year: int) -> str:
    return str(start_year) if code in CALENDAR_YEAR else f"{start_year}-{(start_year + 1) % 100:02d}"


def parse_matches(payload: dict, comp: str) -> list[dict]:
    """football-data.org /competitions/{code}/matches yanıtını proje satır formatına çevirir."""
    league, _ = COMPETITIONS[comp]
    rows = []
    for m in payload.get("matches", []):
        sc = m.get("score") or {}
        finished = m.get("status") in FINISHED
        ft_h = ft_a = ht_h = ht_a = np.nan
        if finished:
            ft = sc.get("fullTime") or {}
            reg, et = sc.get("regularTime"), sc.get("extraTime")
            if reg and reg.get("home") is not None:             # uzatma/penaltı olan maç: 90 dk skoru
                ft = reg
            elif sc.get("duration") not in (None, "REGULAR") and et:   # fullTime uzatmayı içeriyorsa düş
                ft = {"home": (ft.get("home") or 0) - (et.get("home") or 0),
                      "away": (ft.get("away") or 0) - (et.get("away") or 0)}
            elif sc.get("duration") not in (None, "REGULAR"):
                ft = {}                                           # güvenilir 90 dk skoru yok
            ht = sc.get("halfTime") or {}
            ft_h, ft_a = ft.get("home", np.nan), ft.get("away", np.nan)
            ht_h, ht_a = ht.get("home", np.nan), ht.get("away", np.nan)
        # UTC -> İstanbul yerel tarih (gün kaymasını azaltır)
        dt = pd.Timestamp(m["utcDate"]).tz_convert("Europe/Istanbul").tz_localize(None)
        start_year = int(((m.get("season") or {}).get("startDate") or str(dt.year))[:4])
        stage = ""
        if league == "ucl":
            stage = "Eleme" if m.get("stage") in KNOCKOUT else "Grup"
        rows.append({
            "date": dt.normalize(), "season": _season_label(comp, start_year), "league": league, "stage": stage,
            "week": m.get("matchday") if stage != "Eleme" else np.nan,
            "home": (m.get("homeTeam") or {}).get("name"), "away": (m.get("awayTeam") or {}).get("name"),
            "HTHG": ht_h, "HTAG": ht_a, "FTHG": ft_h, "FTAG": ft_a,
        })
    return [r for r in rows if r["home"] and r["away"]]


def fetch_competition(token: str, comp: str, season_year: int, session=None) -> tuple[list[dict], str]:
    """Tek yarışma + sezon çeker. (satırlar, durum mesajı) döndürür; hata durumunda boş liste ve açıklama."""
    http = session or requests
    try:
        resp = http.get(f"{API}/competitions/{comp}/matches", params={"season": season_year},
                        headers={"X-Auth-Token": token}, timeout=20)
    except requests.RequestException as e:
        return [], f"{comp}: bağlantı hatası ({type(e).__name__})"
    if resp.status_code == 429:
        return [], f"{comp}: istek sınırı aşıldı (dakikada 10); biraz bekleyip tekrar deneyin"
    if resp.status_code in (401, 403):
        return [], f"{comp}: erişim yok ({resp.status_code}) — API anahtarı hatalı ya da ücretsiz planda bu veri kapalı"
    if resp.status_code != 200:
        return [], f"{comp}: HTTP {resp.status_code}"
    rows = parse_matches(resp.json(), comp)
    return rows, f"{comp} {season_year}: {len(rows)} maç"


def fetch_all(token: str, comps: list[str], seasons: list[int], progress=None) -> tuple[pd.DataFrame, list[str]]:
    """Seçilen yarışma ve sezonları çeker (rate-limit için aralara bekleme koyar), önbelleğe yazar."""
    rows, log = [], []
    jobs = [(c, s) for c in comps for s in seasons]
    for i, (c, s) in enumerate(jobs):
        r, msg = fetch_competition(token, c, s)
        rows += r; log.append(msg)
        if progress:
            progress((i + 1) / len(jobs), msg)
        if i < len(jobs) - 1:
            time.sleep(6.5)                                       # 10 istek/dk sınırı
    df = pd.DataFrame(rows)
    if len(df):
        merge_into_cache(df)
    return df, log


def merge_into_cache(new: pd.DataFrame) -> None:
    """Yeni çekilen maçları önbellekle birleştirir (aynı lig/sezon/takım/tarih -> yenisi kazanır)."""
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    old = load_cache()
    allr = pd.concat([new, old], ignore_index=True) if len(old) else new
    allr["date"] = pd.to_datetime(allr["date"])
    allr = allr.drop_duplicates(subset=["league", "season", "home", "away", "date", "stage"], keep="first")
    allr.to_csv(CACHE, index=False)


def load_cache() -> pd.DataFrame:
    """Daha önce çekilmiş canlı veriyi okur (yoksa boş DataFrame)."""
    if not CACHE.exists():
        return pd.DataFrame()
    df = pd.read_csv(CACHE, parse_dates=["date"])
    df["stage"] = df["stage"].fillna("")
    return df


def cache_stamp() -> float:
    """Önbellek dosyasının değişim zamanı (Streamlit önbelleğini yenilemek için anahtar)."""
    return CACHE.stat().st_mtime if CACHE.exists() else 0.0


# =========================================================================== #
# API-Football (api-sports.io, v3)  —  header: x-apisports-key
# =========================================================================== #
APIF = "https://v3.football.api-sports.io"
# API-Football lig kimliği -> (proje lig kodu, görünen ad)
APIF_LEAGUES = {
    2: ("ucl", "Şampiyonlar Ligi"), 39: ("en1", "Premier League"), 40: ("en2", "Championship"),
    78: ("de1", "Bundesliga"), 140: ("es1", "La Liga"), 135: ("it1", "Serie A"), 61: ("fr1", "Ligue 1"),
    88: ("nl1", "Eredivisie"), 94: ("pt1", "Primeira Liga"), 203: ("tr1", "Süper Lig"),
    144: ("be1", "Belçika Pro League"), 179: ("sco1", "İskoçya Premiership"), 197: ("gr1", "Yunanistan Super League"),
    218: ("at1", "Avusturya Bundesliga"), 207: ("ch1", "İsviçre Super League"), 71: ("br1", "Brezilya Série A"),
}
APIF_FINISHED = {"FT", "AET", "PEN"}
_CAL_YEAR_LEAGUES = {"br1"}


def parse_apifootball(payload: dict) -> list[dict]:
    """/fixtures yanıtını proje satır formatına çevirir. Skor: 90 dakikalık (score.fulltime) ve ilk yarı (score.halftime)."""
    import re
    rows = []
    for it in payload.get("response", []):
        lg = it.get("league") or {}
        if lg.get("id") not in APIF_LEAGUES:
            continue
        code = APIF_LEAGUES[lg["id"]][0]
        fx, tm, sc = it.get("fixture") or {}, it.get("teams") or {}, it.get("score") or {}
        finished = (fx.get("status") or {}).get("short") in APIF_FINISHED
        ft, ht = sc.get("fulltime") or {}, sc.get("halftime") or {}
        ok = finished and ft.get("home") is not None and ft.get("away") is not None
        rnd = str(lg.get("round") or "")
        n = re.search(r"(\d+)\s*$", rnd)
        stage = ""
        if code == "ucl":
            stage = "Grup" if ("Group" in rnd or "League Stage" in rnd) else "Eleme"
        year = int(lg.get("season") or pd.Timestamp(fx["date"]).year)
        dt = pd.Timestamp(fx["date"])
        dt = (dt.tz_convert("Europe/Istanbul") if dt.tzinfo else dt).tz_localize(None) if dt.tzinfo else dt
        rows.append({
            "date": dt.normalize(),
            "season": str(year) if code in _CAL_YEAR_LEAGUES else f"{year}-{(year + 1) % 100:02d}",
            "league": code, "stage": stage,
            "week": int(n.group(1)) if (n and stage != "Eleme") else np.nan,
            "home": (tm.get("home") or {}).get("name"), "away": (tm.get("away") or {}).get("name"),
            "HTHG": ht.get("home") if ok else np.nan, "HTAG": ht.get("away") if ok else np.nan,
            "FTHG": ft.get("home") if ok else np.nan, "FTAG": ft.get("away") if ok else np.nan,
        })
    return [r for r in rows if r["home"] and r["away"]]


def _apif_get(key: str, params: dict, session=None) -> tuple[list[dict], str]:
    """API-Football'a tek istek atar. API hata mesajlarını (200 dönse bile 'errors' alanında) kullanıcıya aktarır."""
    http = session or requests
    try:
        resp = http.get(f"{APIF}/fixtures", params=params, headers={"x-apisports-key": key}, timeout=25)
    except requests.RequestException as e:
        return [], f"bağlantı hatası ({type(e).__name__})"
    if resp.status_code != 200:
        return [], f"HTTP {resp.status_code}"
    js = resp.json()
    errs = js.get("errors")
    if errs:                                    # liste ya da sözlük gelebilir
        msg = "; ".join(errs.values()) if isinstance(errs, dict) else "; ".join(map(str, errs))
        return [], f"API: {msg}"
    left = resp.headers.get("x-ratelimit-requests-remaining")
    rows = parse_apifootball(js)
    return rows, f"{len(rows)} maç" + (f" (günlük kalan istek: {left})" if left else "")


def fetch_apif(key: str, league_ids: list[int], season: int, progress=None) -> tuple[pd.DataFrame, list[str]]:
    """Seçili ligleri sezon bazında çeker (lig başına 1 istek)."""
    rows, log = [], []
    for i, lid in enumerate(league_ids):
        r, msg = _apif_get(key, {"league": lid, "season": season})
        rows += r; log.append(f"{APIF_LEAGUES[lid][1]} {season}: {msg}")
        if progress:
            progress((i + 1) / len(league_ids), log[-1])
        if i < len(league_ids) - 1:
            time.sleep(6.5)                      # dakikalık limit için
    df = pd.DataFrame(rows)
    if len(df):
        merge_into_cache(df)
    return df, log


def fetch_apif_dates(key: str, days: list[pd.Timestamp], progress=None) -> tuple[pd.DataFrame, list[str]]:
    """Günün (ve istenen diğer günlerin) tüm maçlarını gün başına TEK istekle çeker; desteklenen liglere süzer."""
    rows, log = [], []
    for i, day in enumerate(days):
        r, msg = _apif_get(key, {"date": f"{day:%Y-%m-%d}", "timezone": "Europe/Istanbul"})
        rows += r; log.append(f"{day:%d.%m.%Y}: {msg}")
        if progress:
            progress((i + 1) / len(days), log[-1])
        if i < len(days) - 1:
            time.sleep(6.5)
    df = pd.DataFrame(rows)
    if len(df):
        merge_into_cache(df)
    return df, log


# =========================================================================== #
# Gelecek fikstür çekme (Matrix V3, kural 4): seçili liglerin önümüzdeki N günü
# =========================================================================== #
def fetch_apif_upcoming(key: str, league_ids: list[int], days: int = 14, progress=None) -> tuple[pd.DataFrame, list[str]]:
    """API-Football: her lig için from/to aralığıyla yaklaşan maçlar (lig başına 1 istek)."""
    t0 = pd.Timestamp.today().normalize()
    rows, log = [], []
    for i, lid in enumerate(league_ids):
        code = APIF_LEAGUES[lid][0]
        season = t0.year if code in _CAL_YEAR_LEAGUES else current_season_year(t0)
        r, msg = _apif_get(key, {"league": lid, "season": season, "from": f"{t0:%Y-%m-%d}",
                                 "to": f"{t0 + pd.Timedelta(days=days):%Y-%m-%d}", "timezone": "Europe/Istanbul"})
        rows += r; log.append(f"{APIF_LEAGUES[lid][1]} (önümüzdeki {days} gün): {msg}")
        if progress:
            progress((i + 1) / len(league_ids), log[-1])
        if i < len(league_ids) - 1:
            time.sleep(6.5)
    df = pd.DataFrame(rows)
    if len(df):
        merge_into_cache(df)
    return df, log


def fetch_fd_upcoming(token: str, comps: list[str], days: int = 14, progress=None) -> tuple[pd.DataFrame, list[str]]:
    """football-data.org: dateFrom/dateTo ile yaklaşan maçlar."""
    t0 = pd.Timestamp.today().normalize()
    rows, log = [], []
    for i, c in enumerate(comps):
        try:
            resp = requests.get(f"{API}/competitions/{c}/matches", headers={"X-Auth-Token": token}, timeout=20,
                                params={"dateFrom": f"{t0:%Y-%m-%d}", "dateTo": f"{t0 + pd.Timedelta(days=days):%Y-%m-%d}"})
            if resp.status_code == 200:
                r = parse_matches(resp.json(), c); rows += r; log.append(f"{c}: {len(r)} maç")
            else:
                log.append(f"{c}: HTTP {resp.status_code}")
        except requests.RequestException as e:
            log.append(f"{c}: bağlantı hatası ({type(e).__name__})")
        if progress:
            progress((i + 1) / len(comps), log[-1])
        if i < len(comps) - 1:
            time.sleep(6.5)
    df = pd.DataFrame(rows)
    if len(df):
        merge_into_cache(df)
    return df, log


# ======================================================================
# 6) V3: SENELİK & AYLIK ÖRÜNTÜLER
# ======================================================================
"""
seasonal.py — Matrix V3: takvim / sezon örüntüleri
  1. Aylık karakter      : takımın belirli ayda (ve ayın ilk/son maçında) yıllar içinde tekrarlayan 6+ / 1/2 / 2/1 alışkanlığı
  2. Sezon kapanışı      : sondan kaçıncı maç (N-1, N-2, N-3 ...) simetrisi
  3. Domino etkisi       : "X, Y ile profil P oynadı -> g hafta sonra Y, W ile sürpriz" kalıbının rol değiştirmiş tekrarı
  4. Gelecek fikstür     : bu örüntüleri yaklaşan maçlarla eşleştirir (scan_upcoming)
Tüm hesaplar yalnızca hedef tarihten ÖNCEKİ verilerle yapılır. Her örüntü için tesadüf olasılığı verilir.
"""




MONTHS = ["Ocak", "Şubat", "Mart", "Nisan", "Mayıs", "Haziran", "Temmuz", "Ağustos", "Eylül", "Ekim", "Kasım", "Aralık"]
EV_TXT = {"6+": "+6 Gol", "REV": "1/2 veya 2/1", "1/2": "1/2", "2/1": "2/1"}
MIN_SEASONS = 3          # alışkanlık için asgari sezon sayısı
LOOKBACK = 4             # en çok son 4 sezon
S_MIN_RATE = 0.75          # sezonların en az %75'inde tekrar (3/3, 3/4, 4/4)
MAX_CHANCE = 0.12        # tesadüf olasılığı bu değerin üstündeyse örüntü gösterilmez


class Seasonal:
    """Bir lig için takvim tablosunu hazırlar; aylık, sezon sonu ve domino örüntülerini arar."""

    def __init__(self, d: pd.DataFrame):
        d = d.sort_values("date").reset_index(drop=True).copy()
        d["mid"] = d.index
        tot = d.FTHG + d.FTAG
        htk = d.HTHG.notna()
        r12 = htk & (d.HTHG > d.HTAG) & (d.FTHG < d.FTAG)
        r21 = htk & (d.HTHG < d.HTAG) & (d.FTHG > d.FTAG)
        d["ev"] = np.where(d.played & (tot >= 6), "6+", np.where(r12, "1/2", np.where(r21, "2/1", "")))
        d["ht00"] = np.where(htk, ((d.HTHG + d.HTAG) == 0).astype(float), np.nan)
        self.d = d

        def side(team, opp, venue, gf, ga):
            return pd.DataFrame({"team": d[team], "opp": d[opp], "venue": venue, "date": d.date, "season": d.season,
                                 "month": d.date.dt.month, "played": d.played, "mid": d.mid, "gf": d[gf], "ga": d[ga],
                                 "ev": d.ev, "ht00": d.ht00})
        T = pd.concat([side("home", "away", "H", "FTHG", "FTAG"), side("away", "home", "A", "FTAG", "FTHG")])
        T = T.sort_values(["team", "date", "mid"]).reset_index(drop=True)
        g = T.groupby(["team", "season"])
        T["k"] = g.cumcount() + 1
        T["n"] = g.mid.transform("size")
        # sezon uzunluğu: son sezonun fikstürü eksikse önceki sezonların medyanını kullan
        sl = T.groupby("season").n.median()
        order = sorted(sl.index, key=lambda s: T.loc[T.season == s, "date"].min())
        prev_med = float(np.median([sl[s] for s in order[:-1]])) if len(order) > 1 else float(sl[order[-1]])
        T["n_use"] = np.where((T.season == order[-1]) & (T.n < 0.9 * prev_med), prev_med, T.n)
        T["rev"] = (T.n_use - T.k + 1).astype(int)           # sondan kaçıncı maç (1 = son maç)
        gm = T.groupby(["team", "season", "month"]).date
        T["m_first"] = T.date == gm.transform("min")
        T["m_last"] = T.date == gm.transform("max")
        T["next_date"] = T.groupby("team").date.shift(-1)
        T["tpev"] = np.where(T.played & (T.ev != ""), T.ev, "")
        self.T = T
        self.by_team = {t: x.reset_index(drop=True) for t, x in T.groupby("team")}
        self.latest = order[-1]

        # --- taban oranlar ---
        pl = T[T.played]
        grp = pl.groupby(["team", "season", "month"])
        self.base_month = {
            "6+": float(grp.ev.apply(lambda s: (s == "6+").any()).mean()),
            "REV": float(grp.ev.apply(lambda s: s.isin(["1/2", "2/1"]).any()).mean()),
        }
        self.base_ev = {e: float((pl.ev == e).mean()) for e in ("6+", "1/2", "2/1")}
        end = pl[pl.rev <= 3].groupby(["team", "season"])
        self.base_end = {
            "6+": float(end.ev.apply(lambda s: (s == "6+").any()).mean()),
            "REV": float(end.ev.apply(lambda s: s.isin(["1/2", "2/1"]).any()).mean()),
        }
        self._build_domino()

    # ------------------------------------------------------------------ yardımcılar
    @staticmethod
    def _is_kind(ev: pd.Series, kind: str) -> pd.Series:
        return ev == "6+" if kind == "6+" else ev.isin(["1/2", "2/1"])

    @staticmethod
    def _desc(g: pd.DataFrame, kind: str) -> str:
        """Bir sezon-grubundaki olayları kısa metinle anlatır; yoksa ❌."""
        hit = g[Seasonal._is_kind(g.ev, kind)]
        if hit.empty:
            return "olay yok ❌"
        return "; ".join(f"{r.date:%d.%m} {r.opp} {r.gf:.0f}-{r.ga:.0f} ({r.ev})" for r in hit.itertuples()) + " ✅"

    def _past(self, team: str, date, season: str) -> pd.DataFrame:
        X = self.by_team.get(team)
        if X is None:
            return pd.DataFrame()
        return X[X.played & (X.season != season) & (X.date < date)]

    # ------------------------------------------------------------------ 1) aylık karakter
    def month_habits(self, team: str, date, season: str) -> list[dict]:
        """Aynı takvim ayında son 3-4 sezonda tekrarlanan 6+ / 1/2-2/1 alışkanlığı."""
        out = []
        past = self._past(team, date, season)
        if past.empty:
            return out
        pm = past[past.date.dt.month == date.month]
        seasons = sorted(pm.season.unique(), key=lambda s: pm[pm.season == s].date.min())[-LOOKBACK:]
        if len(seasons) >= MIN_SEASONS:
            for kind in ("6+", "REV"):
                flags = [bool(self._is_kind(pm[pm.season == s].ev, kind).any()) for s in seasons]
                k, S = sum(flags), len(seasons)
                ch = binom_sf(k, S, self.base_month[kind])
                if k >= MIN_SEASONS and k / S >= S_MIN_RATE and ch <= MAX_CHANCE:
                    out.append({
                        "team": team, "scope": "ay", "kind": kind, "pick": EV_TXT[kind], "k": k, "n": S, "chance": ch,
                        "title": f"{team}, {MONTHS[date.month-1]} ayını seviyor!",
                        "text": f"Son {S} sezonun {MONTHS[date.month-1]} ayı maçlarının {k}'inde en az bir "
                                f"{EV_TXT[kind]} sonucu çıktı (tüm takım-sezon ayları için taban %{100*self.base_month[kind]:.0f}).",
                        "history": [f"{s}: {self._desc(pm[pm.season == s], kind)}" for s in seasons]})
        # ayın ilk / son maçı: aynı olay art arda ≥3 sezon
        X = self.by_team[team]
        cur = X[(X.season == season) & (X.date.dt.month == date.month)]
        row = cur[cur.date == date]
        if row.empty:
            return out
        r0 = row.iloc[0]
        for scope, col, label in (("ilk", "m_first", "ilk"), ("son", "m_last", "son")):
            if not bool(r0[col]):
                continue
            if scope == "son" and pd.isna(r0.next_date) and r0.rev != 1:
                continue                                        # ayın sonu belirsiz (fikstür eksik)
            pos = pm[pm[col]].sort_values("date")
            ev_by_season = [(s, pos[pos.season == s]) for s in seasons]
            ev_by_season = [(s, g.iloc[-1]) for s, g in ev_by_season if len(g)]
            for ev in ("6+", "1/2", "2/1"):
                k = 0
                for s, r in reversed(ev_by_season):               # en yeniden geriye ardışık
                    if r.ev == ev: k += 1
                    else: break
                if k >= MIN_SEASONS:
                    ch = self.base_ev[ev] ** k
                    if ch <= MAX_CHANCE:
                        out.append({
                            "team": team, "scope": scope, "kind": ev, "pick": EV_TXT[ev], "k": k, "n": k, "chance": ch,
                            "title": f"{team}, {MONTHS[date.month-1]} ayının {label} maçında hep {EV_TXT[ev]} yapıyor!",
                            "text": f"{team}, son {k} sezondur {MONTHS[date.month-1]} ayının {label} maçında {EV_TXT[ev]} sonucu verdi.",
                            "history": [f"{s}: {r.date:%d.%m.%Y} {r.opp} {r.gf:.0f}-{r.ga:.0f} ({r.ev or 'normal'})"
                                        f"{' ✅' if r.ev == ev else ' ❌'}" for s, r in ev_by_season[-max(k, 3):]]})
        return out

    # ------------------------------------------------------------------ 2) sezon kapanışı simetrisi
    def endseason_habits(self, team: str, date, season: str) -> list[dict]:
        """Sondan N. maç (N-1, N-2, N-3) ve son 3 hafta penceresi alışkanlıkları."""
        out = []
        X = self.by_team.get(team)
        if X is None:
            return out
        row = X[(X.season == season) & (X.date == date)]
        if row.empty:
            return out
        rv = int(row.iloc[0].rev)
        if rv > 3:
            return out
        P = X[X.played & (X.season != season) & (X.date < date)]
        seasons = sorted(P.season.unique(), key=lambda s: P[P.season == s].date.min())[-LOOKBACK:]
        if len(seasons) < MIN_SEASONS:
            return out
        win = P[P.rev <= 3]
        for kind in ("6+", "REV"):
            flags = [bool(self._is_kind(win[win.season == s].ev, kind).any()) for s in seasons]
            k, S = sum(flags), len(seasons)
            ch = binom_sf(k, S, self.base_end[kind])
            if k >= MIN_SEASONS and k / S >= S_MIN_RATE and ch <= MAX_CHANCE:
                out.append({
                    "team": team, "scope": "kapanış", "kind": kind, "pick": EV_TXT[kind], "k": k, "n": S, "chance": ch,
                    "title": f"{team}, sezon kapanışında (son 3 hafta) hep sürpriz yapıyor!",
                    "text": f"Bu maç sezonun sondan {rv}. maçı (N-{rv-1}). {team}, son {S} sezonun son 3 haftasının "
                            f"{k}'inde {EV_TXT[kind]} çıkardı (taban %{100*self.base_end[kind]:.0f}).",
                    "history": [f"{s}: {self._desc(win[win.season == s], kind)}" for s in seasons]})
        # aynı sondan sıra, aynı olay
        at = P[P.rev == rv]
        by_s = [(s, at[at.season == s]) for s in seasons]
        by_s = [(s, g.iloc[-1]) for s, g in by_s if len(g)]
        for ev in ("6+", "1/2", "2/1"):
            k = 0
            for s, r in reversed(by_s):
                if r.ev == ev: k += 1
                else: break
            if k >= MIN_SEASONS and self.base_ev[ev] ** k <= MAX_CHANCE:
                out.append({
                    "team": team, "scope": "kapanış", "kind": ev, "pick": EV_TXT[ev], "k": k, "n": k,
                    "chance": self.base_ev[ev] ** k,
                    "title": f"{team}, sezonun sondan {rv}. maçında hep {EV_TXT[ev]} yapıyor!",
                    "text": f"Son {k} sezondur sondan {rv}. maçında {EV_TXT[ev]} sonucu çıktı.",
                    "history": [f"{s}: {r.date:%d.%m.%Y} {r.opp} {r.gf:.0f}-{r.ga:.0f} ({r.ev or 'normal'})"
                                f"{' ✅' if r.ev == ev else ' ❌'}" for s, r in by_s[-max(k, 3):]]})
        return out

    # ------------------------------------------------------------------ 3) domino etkisi
    def _build_domino(self):
        """
        (Y, W) çifti -> [X, profil, g, E2...]: X, Y ile profil P (olay + İY 0-0 bilgisi) oynadıktan sonra Y'nin
        g. sonraki maçı (g=1..3) W ile bir sürprizle bitmiş.
        """
        pos = {}
        for t, x in self.by_team.items():
            for i, m in enumerate(x.mid.values):
                pos[(t, int(m))] = i
        self.dom_by_pair: dict = defaultdict(list)
        for X, x in self.by_team.items():
            sub = x[x.played & (x.ev != "")]
            for r in sub.itertuples():
                Y = r.opp
                y = self.by_team.get(Y)
                if y is None:
                    continue
                p = pos.get((Y, int(r.mid)))
                if p is None:
                    continue
                for g in (1, 2, 3):
                    if p + g < len(y) and y.played.iloc[p + g] and y.ev.iloc[p + g] != "":
                        n2 = y.iloc[p + g]
                        self.dom_by_pair[(Y, n2.opp)].append({
                            "X": X, "P": (r.ev, r.ht00), "g": g, "date": r.date, "season": r.season,
                            "m1": int(r.mid), "m2": int(n2.mid), "Y": Y, "W": n2.opp, "ev2": n2.ev,
                            "Yvenue2": n2.venue, "ht00_1": r.ht00})

    def domino(self, home: str, away: str, date, season: str) -> list[dict]:
        """Bu fikstür (home-away) bir domino kalıbının rol değiştirmiş tekrarı mı?"""
        out, seen = [], set()
        for W, Y, W_is_home in ((home, away, True), (away, home, False)):
            for e in self.dom_by_pair.get((Y, W), []):
                X = e["X"]
                if X in (W, Y):
                    continue
                xr = self.by_team[X]
                last = xr[xr.played & (xr.date < date)]
                if last.empty:
                    continue
                lr = last.iloc[-1]
                if (date - lr.date).days > 35 or lr.opp != W or e["date"] >= lr.date:
                    continue
                if (lr.ev, lr.ht00) != e["P"]:
                    continue
                # W'nin lr.date ile bu maç arasındaki maç sayısı => bu maç W'nin kaçıncı sonraki maçı?
                wr = self.by_team[W]
                between = int(((wr.date > lr.date) & (wr.date < date)).sum())
                if between + 1 != e["g"]:
                    continue
                if (X, e["g"]) in seen:
                    continue
                seen.add((X, e["g"]))
                m2 = self.d.loc[e["m2"]]
                # Y'nin geçmişteki sonucunu bugünkü maça Y'nin perspektifinden taşı
                ev2 = e["ev2"]
                if ev2 in ("1/2", "2/1"):
                    y_home_then = e["Yvenue2"] == "H"
                    y_home_now = not W_is_home
                    pick = ev2 if y_home_then == y_home_now else ("2/1" if ev2 == "1/2" else "1/2")
                else:
                    pick = "+6 Gol"
                ht_txt = {1.0: "İY 0-0", 0.0: "İY gollü", np.nan: ""}.get(e["P"][1], "")
                prof = f"{ht_txt + ', ' if ht_txt else ''}{EV_TXT.get(e['P'][0], e['P'][0])}"
                out.append({
                    "team": X, "scope": "domino", "kind": ev2, "pick": pick, "k": 1, "n": 1, "chance": float("nan"),
                    "title": f"DOMİNO ETKİSİ: {X} → {W} → {Y}",
                    "text": f"Geçmişte {X}, {Y} ile ({prof}) oynadıktan {e['g']} hafta sonra {Y}–{e['W']} maçı {ev2} oldu "
                            f"({m2.date:%d.%m.%Y}, {m2.home} {m2.FTHG:.0f}-{m2.FTAG:.0f} {m2.away}). "
                            f"ŞİMDİ {X}, {W} ile ({prof}) oynadı ({lr.date:%d.%m.%Y}); {e['g']} hafta sonra "
                            f"{home}–{away} maçında {pick} aranır.",
                    "history": [f"{e['season']}: {X} – {Y} ({prof}) ➔ {e['g']} hafta sonra {Y} – {e['W']}: {ev2}"]})
        return out

    # ------------------------------------------------------------------ 4) gelecek fikstür taraması
    def scan_fixture(self, home: str, away: str, date, season: str) -> list[dict]:
        findings = []
        for T in (home, away):
            if T not in self.by_team:
                continue
            findings += self.month_habits(T, date, season)
            findings += self.endseason_habits(T, date, season)
        findings += self.domino(home, away, date, season)
        return sorted(findings, key=lambda f: 0.5 if math.isnan(f["chance"]) else f["chance"])

    def scan_upcoming(self, today: pd.Timestamp, days: int = 14) -> list[dict]:
        """Önümüzdeki `days` gündeki tüm fikstürleri tarar; yalnızca bulgusu olan maçları döndürür."""
        d = self.d
        up = d[(~d.played) & (d.date >= today) & (d.date <= today + pd.Timedelta(days=days))]
        res = []
        for r in up.sort_values("date").itertuples():
            f = self.scan_fixture(r.home, r.away, r.date, r.season)
            if f:
                res.append({"home": r.home, "away": r.away, "date": r.date, "season": r.season, "findings": f})
        return res


# ======================================================================
# 7) STREAMLIT ARAYÜZÜ
# ======================================================================
"""
app.py  —  Matrix V2 Streamlit arayüzü
Çalıştırma (app.py'nin bulunduğu klasörde):  python -m streamlit run app.py
Veri: ./data içindeki ZIP'ler + (isteğe bağlı) football-data.org canlı verisi (data/live_cache.csv).
"""




# API anahtarı doğrudan koda gömülü (dosyayı başkasıyla paylaşırsan sil!)
EMBEDDED_KEYS = {"API_FOOTBALL_KEY": "5f436794b85a4215bde8a66bd402ae62", "FOOTBALL_DATA_TOKEN": ""}

st.set_page_config(page_title="Maç Analiz Paneli · Matrix V3", page_icon="⚽", layout="centered")

st.markdown(
    """
    <style>
      .block-container {padding-top: 2rem; max-width: 900px;}
      div[data-testid="stMetricValue"] {font-size: 1.9rem;}
      .fav {font-size: 1.5rem; font-weight: 700; margin: .2rem 0 .6rem 0;}
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# Veri yükleme (önbellekli). `stamp` canlı önbellek dosyası değişince cache'i yeniler.
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Veriler okunuyor ve birleştiriliyor…")
def get_data(sources: tuple[str, ...], stamp: float) -> pd.DataFrame:
    """ZIP kaynakları + canlı API önbelleğini tek DataFrame'e birleştirir."""
    return load_matches(list(sources), extra=load_cache())


@st.cache_resource(show_spinner="Matrix tablosu hazırlanıyor…")
def get_matrix(sources: tuple[str, ...], stamp: float, league: str) -> Matrix:
    """Seçili lig için Matrix Analisti'nin ön hesaplamaları (rakip zincirleri, indeksler)."""
    return Matrix(get_data(sources, stamp).query("league == @league"))


@st.cache_resource(show_spinner="Takvim örüntüleri hazırlanıyor…")
def get_seasonal(sources: tuple[str, ...], stamp: float, league: str) -> Seasonal:
    """Seçili lig için aylık / sezon sonu / domino örüntü motorunu bir kez hazırlar."""
    return Seasonal(get_data(sources, stamp).query("league == @league"))


@st.cache_data(show_spinner=False)
def get_burst_hist(sources: tuple[str, ...], stamp: float) -> dict:
    """Patlama koşulunun tüm liglerdeki geçmiş gerçekleşme oranı."""
    return burst_history(get_data(sources, stamp))


def pct(x: float) -> str:
    """0.1234 -> '%12.3'"""
    return f"%{100 * x:.1f}"


# --------------------------------------------------------------------------- #
# Sol menü: veri kaynağı + API güncelleme
# --------------------------------------------------------------------------- #
default_src = default_sources()
today_ts = pd.Timestamp.today().normalize()
with st.sidebar:
    st.header("Veri kaynağı")
    src_text = st.text_area("ZIP / klasör yolları (satır başına bir tane)", value="\n".join(default_src))

    st.divider()
    st.header("🔄 Veri Güncelleme (API)")

    def _secret(name: str) -> str:
        """Anahtarı önce .streamlit/secrets.toml, sonra ortam değişkeninden okur (arayüzden girmek gerekmez)."""
        try:
            v = str(st.secrets.get(name, "")) or os.environ.get(name, "")
        except Exception:
            v = os.environ.get(name, "")
        return v or EMBEDDED_KEYS.get(name, "")

    provider = st.radio("Sağlayıcı", ["API-Football", "football-data.org"], horizontal=True)
    if provider == "API-Football":
        api_key = st.text_input("API Key", type="password", value=_secret("API_FOOTBALL_KEY"),
                                help="Varsayılan olarak .streamlit/secrets.toml içindeki anahtar kullanılır.")
        lg_labels = {f"{v[1]} ({k})": k for k, v in APIF_LEAGUES.items()}
        chosen = st.multiselect("Ligler (sezon güncellemesi için)", list(lg_labels),
                                default=[l for l in lg_labels if "(2)" in l or "(203)" in l or "(39)" in l])
        c_a, c_b = st.columns(2)
        do_today = c_a.button("Günün Maçları", use_container_width=True,
                              help="Bugün + yarın tüm maçlar (gün başına 1 istek).")
        do_season = c_b.button("Sezonu Güncelle", use_container_width=True,
                               help="Seçili liglerin bu sezonki tüm maçları (lig başına 1 istek).")
        st.caption("Ücretsiz API-Football planı günde 100 istekle sınırlıdır ve bazı planlarda güncel sezon kapalıdır; "
                   "API'nin döndürdüğü hata mesajı aşağıda görünür.")
        do_up = st.button("📆 Gelecek 2 Haftalık Fikstürü Çek", use_container_width=True,
                          help="Seçili liglerin önümüzdeki 14 günlük maçlarını çeker (lig başına 1 istek).")
        run = None
        if do_up:
            run = lambda bar: fetch_apif_upcoming(api_key, [lg_labels[c] for c in chosen], 14,
                                                     progress=lambda f, m: bar.progress(f, text=m))
        elif do_today:
            run = lambda bar: fetch_apif_dates(api_key, [today_ts, today_ts + pd.Timedelta(days=1)],
                                                  progress=lambda f, m: bar.progress(f, text=m))
        elif do_season:
            run = lambda bar: fetch_apif(api_key, [lg_labels[c] for c in chosen], current_season_year(),
                                            progress=lambda f, m: bar.progress(f, text=m))
    else:
        api_key = st.text_input("API Key", type="password", value=_secret("FOOTBALL_DATA_TOKEN"))
        comp_labels = {f"{v[1]} ({k})": k for k, v in COMPETITIONS.items()}
        chosen = st.multiselect("Yarışmalar", list(comp_labels),
                                default=[l for l in comp_labels if "(CL)" in l or "(PL)" in l])
        prev_season = st.checkbox("Önceki sezonu da çek", value=False)
        run = None
        if st.button("📆 Gelecek 2 Haftalık Fikstürü Çek", use_container_width=True):
            run = lambda bar: fetch_fd_upcoming(api_key, [comp_labels[c] for c in chosen], 14,
                                                   progress=lambda f, m: bar.progress(f, text=m))
        if st.button("Günün Maçlarını / Güncel Verileri Çek", use_container_width=True):
            y = current_season_year()
            run = lambda bar: fetch_all(api_key, [comp_labels[c] for c in chosen], [y, y - 1] if prev_season else [y],
                                           progress=lambda f, m: bar.progress(f, text=m))
    if run is not None:
        if not api_key:
            st.error("API anahtarı bulunamadı.")
        else:
            bar = st.progress(0.0, text="Çekiliyor…")
            new, log = run(bar)
            for line in log:
                st.caption("• " + line)
            if len(new):
                st.success(f"{len(new)} maç alındı ve kaydedildi.")
                st.cache_data.clear(); st.cache_resource.clear()
                st.rerun()
            else:
                st.warning("Hiç maç alınamadı (yukarıdaki mesajlara bakın).")
    st.caption(f"Canlı önbellek: {'var' if CACHE.exists() else 'yok'} · "
               "Veri setinde kart/şut yoktur; analizler gol ve skor temellidir.")

sources = tuple(s.strip() for s in src_text.splitlines() if s.strip() and Path(s.strip()).exists())
stamp = cache_stamp()
if not sources and stamp == 0:
    st.error("Veri bulunamadı. `data/` klasörüne ZIP dosyalarını koyun ya da soldan yol girin.")
    st.stop()

df = get_data(sources, stamp)
if df.empty:
    st.error("Okunabilir maç verisi bulunamadı.")
    st.stop()

# --------------------------------------------------------------------------- #
# Üst bölüm: Günün maçları + Lig / Ev Sahibi / Deplasman
# --------------------------------------------------------------------------- #
st.title("⚽ Maç Analiz Paneli · Matrix V3")

counts = df[df.played].groupby(["league", "league_name"]).size().reset_index(name="n")
counts = counts[counts.n >= 200].sort_values("n", ascending=False)
league_options = dict(zip(counts.league_name, counts.league))
names = list(league_options)
default_name = "Türkiye – Süper Lig" if "Türkiye – Süper Lig" in names else names[0]
st.session_state.setdefault("sel_league", default_name)
if st.session_state["sel_league"] not in names:
    st.session_state["sel_league"] = default_name

# Günün (ve önümüzdeki 2 günün) fikstürü
today = pd.Timestamp.today().normalize()
upcoming = df[(~df.played) & (df.date >= today) & (df.date <= today + pd.Timedelta(days=2))
              & df.league_name.isin(names)].sort_values(["date", "league_name"])
fx_map = {f"{r.date:%d.%m} · {r.league_name} · {r.home} – {r.away}": (r.league_name, r.home, r.away)
          for r in upcoming.itertuples()}


def apply_fixture():
    """Seçilen günün maçını lig/takım kutularına uygular."""
    sel = st.session_state.get("pick_fx")
    if sel in fx_map:
        lg, h, a = fx_map[sel]
        st.session_state["sel_league"], st.session_state["sel_home"], st.session_state["sel_away"] = lg, h, a


with st.expander(f"📅 Günün / yakın maçlar ({len(fx_map)})", expanded=False):
    if fx_map:
        st.selectbox("Bir maç seçin (otomatik analiz edilir)", ["—"] + list(fx_map), key="pick_fx", on_change=apply_fixture)
    else:
        st.caption("Önümüzdeki 2 gün için fikstür bulunamadı. Soldan API ile güncelleyebilirsiniz.")

c0, c1, c2 = st.columns([1.2, 1, 1])
league_name = c0.selectbox("Lig", names, key="sel_league")
league_code = league_options[league_name]
d = df[df.league == league_code]
teams = teams_in_league(d)
if st.session_state.get("sel_home") not in teams:
    st.session_state["sel_home"] = teams[0]
if st.session_state.get("sel_away") not in teams or st.session_state.get("sel_away") == st.session_state["sel_home"]:
    st.session_state["sel_away"] = next(t for t in teams if t != st.session_state["sel_home"])
home = c1.selectbox("Ev Sahibi Takım Seç", teams, key="sel_home")
away = c2.selectbox("Deplasman Takım Seç", teams, key="sel_away")

if home == away:
    st.warning("Lütfen iki farklı takım seçin.")
    st.stop()

with st.expander("Maç tarihi / hafta / aşama (isteğe bağlı)"):
    st.caption("Boş bırakılırsa fikstür varsa oradan, yoksa bugün/bir sonraki hafta alınır.")
    use_custom = st.checkbox("Elle gir")
    cd1, cd2, cd3 = st.columns(3)
    custom_date = cd1.date_input("Maç tarihi", value=pd.Timestamp.today().date(), disabled=not use_custom)
    custom_week = cd2.number_input("Lig haftası", min_value=1, max_value=60, value=10, disabled=not use_custom)
    stage_sel = cd3.selectbox("ŞL aşaması", ["Otomatik", "Grup", "Eleme"], disabled=league_code != "ucl")

# --------------------------------------------------------------------------- #
# Analiz
# --------------------------------------------------------------------------- #
r = analyze(
    d, home, away,
    match_date=pd.Timestamp(custom_date) if use_custom else None,
    week=float(custom_week) if use_custom else None,
)
p, xg = r["probs"], r["xg"]

fx_row = find_next_fixture(d, home, away, today)
mine = d[d.played & ((d.home == home) | (d.away == home))].sort_values("date")
season_now = fx_row["season"] if fx_row is not None else (mine["season"].iloc[-1] if len(mine) else "")
stage_now = None
if league_code == "ucl":
    if stage_sel != "Otomatik":
        stage_now = stage_sel
    elif fx_row is not None and fx_row.get("stage"):
        stage_now = fx_row["stage"]

st.caption(
    f"Analiz tarihi: {r['match_date']:%d.%m.%Y} · hafta: {'—' if pd.isna(r['week']) else int(r['week'])}"
    f"{' · aşama: ' + stage_now if stage_now else ''} · veri son güncelleme: {r['ref_date']:%d.%m.%Y} · "
    f"H2H maç sayısı: {r['n_matches']['h2h']}"
)
if league_code == "ucl":
    st.info("Şampiyonlar Ligi: yerel liglere göre az maç var, güç tahminleri lig ortalamasına daha çok büzülür. "
            "Dahili ZIP verisinde ŞL 2025-26 sonrası yok; güncel sezon için soldan API ile çekin.")

tab_main, tab_seasonal = st.tabs(["🎯 Maç Analizi", "📅 Senelik & Aylık Fikstür Örüntüleri"])

with tab_main:
    # ============================= 1) FAVORİ ve GÜÇ DENGESİ ===================== #
    with st.container(border=True):
        st.subheader("Favori ve Güç Dengesi")
        fav = r["favorite"]
        st.markdown(f"<div class='fav'>{'⚖️' if fav == 'Beraberliğe Yakın' else '🏆'} {fav} "
                    f"<span style='font-weight:400;font-size:1rem;opacity:.7'>· {r['favorite_note']}</span></div>",
                    unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3)
        m1.metric(f"{home} (Ev)", pct(p["home"]))
        m2.metric("Beraberlik", pct(p["draw"]))
        m3.metric(f"{away} (Dep)", pct(p["away"]))
        ht = r["ht"]
        lab = {"1": f"1 · {home}", "X": "X · Beraberlik", "2": f"2 · {away}"}
        t1, t2 = st.columns(2)
        t1.metric("İlk Yarı Tahmini (İY)", lab[ht["pick"]])
        t1.caption(f"İY olasılıkları: 1 %{100*ht['p1']:.0f} · X %{100*ht['px']:.0f} · 2 %{100*ht['p2']:.0f}")
        t2.metric("Maç Sonu Tahmini (MS)", lab[r["ft_pick"]])
        t2.caption(f"MS olasılıkları: 1 %{100*p['home']:.0f} · X %{100*p['draw']:.0f} · 2 %{100*p['away']:.0f}")

        def mom_txt(team, h):
            m = h["momentum"]
            if m > 0.08:
                return f"{team}: ikinci yarılarda yükseliyor (+{m:.2f} gol farkı/maç)"
            if m < -0.08:
                return f"{team}: ikinci yarılarda düşüş gösteriyor ({m:.2f} gol farkı/maç)"
            return f"{team}: iki yarı dengeli"
        st.caption("Momentum — " + " · ".join([mom_txt(home, ht["home"]), mom_txt(away, ht["away"])]))
        st.progress(float(p["home"]), text=f"Ev sahibi {pct(p['home'])}")
        st.progress(float(p["draw"]), text=f"Beraberlik {pct(p['draw'])}")
        st.progress(float(p["away"]), text=f"Deplasman {pct(p['away'])}")
        f = r["form"]
        st.caption(
            f"Son 5 maç (G/B/M): {home} **{f['home'] or '—'}** · {away} **{f['away'] or '—'}**  |  "
            f"Maç başı puan (son 5): {f['ppg5'][0]:.2f} – {f['ppg5'][1]:.2f}  |  "
            f"Saha içi/dışı form (son 8): {f['ppg_venue'][0]:.2f} – {f['ppg_venue'][1]:.2f}  |  "
            f"H2H: {r['h2h_summary']['home_w']}G-{r['h2h_summary']['draw']}B-{r['h2h_summary']['away_w']}M"
        )

    # ============================= 2) KRİTİK ÖRÜNTÜLER ========================== #
    with st.container(border=True):
        st.subheader("Kritik Örüntüler")
        if r["patterns"]:
            for pt in r["patterns"][:8]:
                icon = {"H2H": "🔁", "Hafta": "📅", "Ay": "🗓️", "Seri": "📈"}.get(pt["kind"], "•")
                st.markdown(f"{icon} **{pt['text']}**")
                st.caption(f"Tesadüfen oluşma olasılığı ≈ {pct(pt['chance'])} · tür: {pt['kind']}")
        else:
            st.info("İstatistiksel olarak dikkat çeken tekrar eden bir örüntü bulunamadı.")
        st.caption("Uyarı: çok sayıda koşul taranır; düşük 'tesadüf olasılığı' bile tek başına kesinlik demek değildir.")

    # ============================= 3) MATRIX SÜRPRİZ RADARI ===================== #
    with st.container(border=True):
        st.subheader("🧬 Matrix Sürpriz Radarı")
        M = get_matrix(sources, stamp, league_code)
        msig_all = M.scan(home, away, r["match_date"], season_now, r["week"])
        msig = [x for x in msig_all if x["chance"] <= 0.10]
        ref = r["ref_date"]
        lv = _long_view(d, ref)
        ch = character(lv, home, away)
        bu = burst(lv, home, away, xg["total"], float(lv[lv.venue == "H"].tg.mean()))
        bh = get_burst_hist(sources, stamp)
        cross = ucl_cross(df, home, away, r["match_date"], stage_now) if league_code == "ucl" else \
            {"signals": [], "notes": [], "ucl_base": np.nan}
        radar = build_radar(r, home, away, msig, ch, bu, bh, cross["signals"])

        # --- Zaman Çizelgesi Kopyalama Alarmı (kural 8): 4 aşamalı döngü, gelecek fikstürle birlikte ---
        for x in [s_ for s_ in msig_all if s_["rule"] == 8][:3]:
            st.error("🚨 **ZAMAN ÇİZELGESİ KOPYALAMA ALARMI!**")
            st.markdown(f"- {x['flow_orig']}")
            st.markdown(f"- {x['flow_now']}")
            st.markdown(f"🎯 **Vuruş Tahmini:** Geçmişteki döngünün patlama sırası bu maça denk geliyor "
                        f"({x['orig_team']}: {x['orig_score']}). Şimdiki maç ({home} – {away}) için beklenti: "
                        f"**{x['pick_text']}** · kopya skor: {x['score_pred']}")
            st.caption(f"{x['extra']} · tesadüfen böyle bir kopya bulma olasılığı ≈ {pct(x['chance'])}")
        if any(s_["rule"] == 8 for s_ in msig_all):
            st.caption(TIMELINE_NOTE)
        for n_ in M._notes[:2]:
            st.caption("ℹ️ " + n_)

        if not radar:
            st.info("Bu maç için belirgin bir 1/2, 2/1 veya +6 gol sürpriz potansiyeli tespit edilmedi.")
        for it in radar:
            tier = TIER_LABEL[it["tier"]]
            title = "+6 Gol" if it["dir"] == "6+" else it["dir"]
            box = st.error if it["tier"] == 2 else st.warning if it["tier"] == 1 else st.info
            box(f"**{tier}: {title}**")
            k1, k2, k3 = st.columns(3)
            k1.metric("Model olasılığı", pct(it["pot"]))
            k2.metric("Kanıt gücü", f"%{100*it['evid']:.0f}" if it["evid"] > 0 else "—")
            ts = TIER_STATS.get((it["dir"], it["tier"]))
            k3.metric("Geçmişte bu seviye", f"%{100*ts[0]/ts[1]:.0f} ({ts[0]}/{ts[1]})" if ts and ts[1] else "—")
            for rs in it["reasons"]:
                st.markdown(f"- {rs}")
            for x in it["signals"]:
                with st.expander(f"Örüntü detayı: {x['title']}"):
                    st.markdown(f"**Geçmiş Döngü:** {x['past']}")
                    st.markdown(f"**Güncel Durum:** {x['now']}")
                    st.markdown(f"**Ek Detaylar:** {x['extra']}")
                    st.markdown(f"👉 **Vuruş Tahmini:** {x['pick']}")
                    st.caption(f"Kural {x['rule']} · örnek: {x['n']} · sürpriz: {x['k']} · tesadüf olasılığı ≈ {pct(x['chance'])}")

        # Şampiyonlar Ligi: çapraz fikstür notları (lig maçı <-> ŞL maçı)
        if league_code == "ucl":
            st.markdown("**🔀 Çapraz fikstür (hafta sonu lig ↔ hafta içi ŞL)**")
            info = [s for s in cross["signals"] if s.get("kind") == "cross_info"]
            for s in info:
                st.markdown(f"- {s['past']} {s['now']}")
            if not info:
                st.caption("Çapraz fikstürde tekrar eden kayda değer bir örüntü bulunamadı.")
            for n_ in cross["notes"]:
                st.caption("• " + n_)

        with st.expander("Seviye mantığı ve geçerlilik"):
            st.markdown(
                "- **🔥🔥 Yüksek potansiyel:** 1/2 ve 2/1 için takım karakteri (ev sahibinin ilk yarıyı önde kapatıp puan "
                "kaybetme oranı × deplasmanın geriden dönme oranı, lig tabanının ≥%25 üstünde) **ve** aynı yönde en az bir Matrix "
                "örüntüsü. +6 için sıkışma/patlama koşulu **ve** aynı yönde örüntü (koşul tarihsel olarak desteklenmiyorsa seviye yükselmez).\n"
                "- **Model olasılığı:** istatistiksel modelin gerçek olay olasılığı. **Kanıt gücü:** 1 − Π(tesadüf olasılıkları); "
                "çok sayıda kural tarandığı için iyimserdir. **Geçmişte bu seviye:** aynı seviyedeki uyarıların geçmişte gerçekleşme oranı.\n"
                f"- Sıkışma/patlama koşulu geçmişte {bh['n']} maçta oluştu, {bh['k']}'inde 6+ gol çıktı "
                f"({'—' if not bh['n'] else pct(bh['rate'])}; taban {pct(bh['base'])})."
            )
            if BACKTEST_NOTE:
                st.caption(BACKTEST_NOTE)

    # ============================= 4) GOL BEKLENTİSİ ============================ #
    with st.container(border=True):
        st.subheader("Gol Beklentisi (Over / Under)")
        g1, g2, g3 = st.columns(3)
        g1.metric(f"{home} xG", f"{xg['home']:.2f}")
        g2.metric(f"{away} xG", f"{xg['away']:.2f}")
        g3.metric("Beklenen toplam gol", f"{xg['total']:.2f}")
        st.markdown("**En olası 3 skor**")
        st.caption(f"{r['score_scenario']} — skorlar, maçın gol potansiyeline göre seçilir; yüzdeler "
                   f"maçın gerçek (koşulsuz) skor olasılıklarıdır.")
        sc = st.columns(3)
        for col, (a, b, pr) in zip(sc, r["top_scores"]):
            col.metric(f"{a} – {b}", pct(pr))
        ou = pd.DataFrame({
            "Çizgi": [f"{L}" for L in r["over"]],
            "Üst": [pct(v) for v in r["over"].values()],
            "Alt": [pct(1 - v) for v in r["over"].values()],
        })
        ou2 = ou[ou["Çizgi"].isin(["1.5", "2.5", "3.5", "4.5"])]
        ocols = st.columns(len(ou2))
        for col, (_, row) in zip(ocols, ou2.iterrows()):
            col.metric(f"{row['Çizgi']} Üst / Alt", f"{row['Üst']}", delta=f"Alt {row['Alt']}", delta_color="off")
        st.caption(f"Karşılıklı gol (KG Var): {pct(r['btts'])} · "
                   f"Hücum/savunma gücü (1.00 = lig ort.): {home} {r['strength']['home'].attack:.2f}/"
                   f"{r['strength']['home'].defense:.2f} · {away} {r['strength']['away'].attack:.2f}/"
                   f"{r['strength']['away'].defense:.2f}")

    with st.expander("Yöntem"):
        st.markdown(
            "- **Güç modeli:** zaman ağırlıklı hücum/savunma oranları, az maçlı takımlar lig ortalamasına büzülür. "
            "xG burada şut bazlı değil, **gol bazlı beklenen gol**dür.\n"
            "- **Skor dağılımı:** bağımsız Poisson; skorlar gol senaryosuna (2.5/3.5 Üst olasılığı) göre seçilir.\n"
            "- **İY / MS:** yarı bazlı güç modeli + momentum.\n"
            "- **Matrix kuralları:** temas takımı, sandviç, hafta kopyası, takvim, tetikleyici skor, H2H döngüsü, N-1, skor döngüsü; "
            "ŞL'de ayrıca çapraz fikstür (hafta sonu lig maçı ↔ ŞL).\n"
            "- **Veri:** ZIP + isteğe bağlı football-data.org. Kart/şut/bahis oranı verisi yoktur."
        )


# ============================= 📅 SENELİK & AYLIK FİKSTÜR ÖRÜNTÜLERİ ======== #
with tab_seasonal:
    st.subheader("📅 Senelik & Aylık Fikstür Örüntüleri")
    st.caption("Takımların aylık ve sezon sonu alışkanlıkları ile domino etkileri; yaklaşan (gelecek) maçlarla eşleştirilir. "
               "Gelecek fikstür için soldan 'Gelecek 2 Haftalık Fikstürü Çek' butonunu kullanın.")
    sc1, sc2 = st.columns(2)
    scope_sel = sc1.radio("Kapsam", ["Seçili lig", "Tüm ligler"], horizontal=True)
    look = sc2.slider("Kaç gün ileriye bakılsın", 3, 21, 14)
    codes = [league_code] if scope_sel == "Seçili lig" else list(league_options.values())
    name_by_code = {v: k for k, v in league_options.items()}
    found, n_scanned = [], 0
    for code in codes:
        S_ = get_seasonal(sources, stamp, code)
        dd = S_.d
        n_scanned += int(((~dd.played) & (dd.date >= today) & (dd.date <= today + pd.Timedelta(days=look))).sum())
        for m_ in S_.scan_upcoming(today, look):
            m_["league"] = name_by_code.get(code, code)
            found.append(m_)
    found.sort(key=lambda m_: (m_["date"], min(0.5 if np.isnan(f_["chance"]) else f_["chance"] for f_ in m_["findings"])))
    st.caption(f"Taranan yaklaşan maç: {n_scanned} · örüntü bulunan maç: {len(found)}")
    if n_scanned == 0:
        st.info("Bu aralıkta fikstür verisi yok. Soldan API ile 'Gelecek 2 Haftalık Fikstürü Çek' butonuna basın.")
    elif not found:
        st.info("Yaklaşan maçlarda tarihsel aylık / sezon sonu / domino örüntüsü bulunamadı.")
    for m_ in found:
        best = min(0.5 if np.isnan(f_["chance"]) else f_["chance"] for f_ in m_["findings"])
        box = st.error if best <= 0.05 else st.warning
        box(f"🚨 **Maç:** {m_['home']} – {m_['away']} (Tarih: {m_['date']:%d} {MONTHS[m_['date'].month-1]}) · {m_['league']}")
        for f_ in m_["findings"][:3]:
            st.markdown(f"**Matrix Tespiti:** {f_['title']}")
            st.markdown(f_["text"])
            if f_["history"]:
                st.markdown("**Geçmiş Yıllar:**\n" + "\n".join(f"- {h_}" for h_ in f_["history"]))
            ch_ = "—" if np.isnan(f_["chance"]) else pct(f_["chance"])
            st.markdown(f"👉 **Vuruş Tahmini:** Bu maçta potansiyel **{f_['pick']}**.")
            st.caption(f"Tür: {f_['scope']} · tesadüf olasılığı ≈ {ch_}")
    with st.expander("Kurallar ve geçerlilik"):
        st.markdown(
            "- **Aylık karakter:** aynı takvim ayında son 3-4 sezonun ≥%75'inde 6+ gol veya 1/2-2/1; ayrıca ayın ilk/son maçında "
            "art arda ≥3 sezon aynı sonuç.\n"
            "- **Sezon kapanışı:** sondan 1-3. maçlarda (N-1, N-2, N-3) ve son 3 hafta penceresinde tekrar eden sürprizler.\n"
            "- **Domino:** X, Y ile (olay + İY profili) oynadıktan g hafta sonra Y, W ile sürpriz yapmışsa; şimdi X, W ile aynı "
            "profili oynadıysa ve W'nin g. sonraki maçı Y ise aynı sürpriz aranır.\n"
            "- Her örüntüde 'tesadüf olasılığı' vardır; çok sayıda takım/ay taranır, düşük olasılık bile kesinlik değildir."
        )
        if SEASONAL_NOTE:
            st.caption(SEASONAL_NOTE)
