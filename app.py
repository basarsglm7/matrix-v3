"""
app.py  —  Matrix V2 Streamlit arayüzü
Çalıştırma (app.py'nin bulunduğu klasörde):  python -m streamlit run app.py
Veri: ./data içindeki ZIP'ler + (isteğe bağlı) football-data.org canlı verisi (data/live_cache.csv).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

import analysis as an
import data_loader as dl
import live_data as ld
import matrix as mx
import seasonal as sn
import v2

st.set_page_config(page_title="Maç Analiz Paneli · Matrix V2", page_icon="⚽", layout="centered")

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
    return dl.load_matches(list(sources), extra=ld.load_cache())


@st.cache_resource(show_spinner="Matrix tablosu hazırlanıyor…")
def get_matrix(sources: tuple[str, ...], stamp: float, league: str) -> mx.Matrix:
    """Seçili lig için Matrix Analisti'nin ön hesaplamaları (rakip zincirleri, indeksler)."""
    return mx.Matrix(get_data(sources, stamp).query("league == @league"))


@st.cache_resource(show_spinner="Takvim örüntüleri hazırlanıyor…")
def get_seasonal(sources: tuple[str, ...], stamp: float, league: str) -> sn.Seasonal:
    """Seçili lig için aylık / sezon sonu / domino örüntü motorunu bir kez hazırlar."""
    return sn.Seasonal(get_data(sources, stamp).query("league == @league"))


@st.cache_data(show_spinner=False)
def get_burst_hist(sources: tuple[str, ...], stamp: float) -> dict:
    """Patlama koşulunun tüm liglerdeki geçmiş gerçekleşme oranı."""
    return v2.burst_history(get_data(sources, stamp))


def pct(x: float) -> str:
    """0.1234 -> '%12.3'"""
    return f"%{100 * x:.1f}"


# --------------------------------------------------------------------------- #
# Sol menü: veri kaynağı + API güncelleme
# --------------------------------------------------------------------------- #
default_src = dl.default_sources()
today_ts = pd.Timestamp.today().normalize()
with st.sidebar:
    st.header("Veri kaynağı")
    src_text = st.text_area("ZIP / klasör yolları (satır başına bir tane)", value="\n".join(default_src))

    st.divider()
    st.header("🔄 Veri Güncelleme (API)")

    def _secret(name: str) -> str:
        """Anahtarı önce .streamlit/secrets.toml, sonra ortam değişkeninden okur (arayüzden girmek gerekmez)."""
        try:
            return str(st.secrets.get(name, "")) or os.environ.get(name, "")
        except Exception:
            return os.environ.get(name, "")

    provider = st.radio("Sağlayıcı", ["API-Football", "football-data.org"], horizontal=True)
    if provider == "API-Football":
        api_key = st.text_input("API Key", type="password", value=_secret("API_FOOTBALL_KEY"),
                                help="Varsayılan olarak .streamlit/secrets.toml içindeki anahtar kullanılır.")
        lg_labels = {f"{v[1]} ({k})": k for k, v in ld.APIF_LEAGUES.items()}
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
            run = lambda bar: ld.fetch_apif_upcoming(api_key, [lg_labels[c] for c in chosen], 14,
                                                     progress=lambda f, m: bar.progress(f, text=m))
        elif do_today:
            run = lambda bar: ld.fetch_apif_dates(api_key, [today_ts, today_ts + pd.Timedelta(days=1)],
                                                  progress=lambda f, m: bar.progress(f, text=m))
        elif do_season:
            run = lambda bar: ld.fetch_apif(api_key, [lg_labels[c] for c in chosen], ld.current_season_year(),
                                            progress=lambda f, m: bar.progress(f, text=m))
    else:
        api_key = st.text_input("API Key", type="password", value=_secret("FOOTBALL_DATA_TOKEN"))
        comp_labels = {f"{v[1]} ({k})": k for k, v in ld.COMPETITIONS.items()}
        chosen = st.multiselect("Yarışmalar", list(comp_labels),
                                default=[l for l in comp_labels if "(CL)" in l or "(PL)" in l])
        prev_season = st.checkbox("Önceki sezonu da çek", value=False)
        run = None
        if st.button("📆 Gelecek 2 Haftalık Fikstürü Çek", use_container_width=True):
            run = lambda bar: ld.fetch_fd_upcoming(api_key, [comp_labels[c] for c in chosen], 14,
                                                   progress=lambda f, m: bar.progress(f, text=m))
        if st.button("Günün Maçlarını / Güncel Verileri Çek", use_container_width=True):
            y = ld.current_season_year()
            run = lambda bar: ld.fetch_all(api_key, [comp_labels[c] for c in chosen], [y, y - 1] if prev_season else [y],
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
    st.caption(f"Canlı önbellek: {'var' if ld.CACHE.exists() else 'yok'} · "
               "Veri setinde kart/şut yoktur; analizler gol ve skor temellidir.")

sources = tuple(s.strip() for s in src_text.splitlines() if s.strip() and Path(s.strip()).exists())
stamp = ld.cache_stamp()
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
st.title("⚽ Maç Analiz Paneli · Matrix V2")

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
teams = an.teams_in_league(d)
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
r = an.analyze(
    d, home, away,
    match_date=pd.Timestamp(custom_date) if use_custom else None,
    week=float(custom_week) if use_custom else None,
)
p, xg = r["probs"], r["xg"]

fx_row = an.find_next_fixture(d, home, away, today)
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
        lv = an._long_view(d, ref)
        ch = v2.character(lv, home, away)
        bu = v2.burst(lv, home, away, xg["total"], float(lv[lv.venue == "H"].tg.mean()))
        bh = get_burst_hist(sources, stamp)
        cross = v2.ucl_cross(df, home, away, r["match_date"], stage_now) if league_code == "ucl" else \
            {"signals": [], "notes": [], "ucl_base": np.nan}
        radar = v2.build_radar(r, home, away, msig, ch, bu, bh, cross["signals"])

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
            st.caption(v2.TIMELINE_NOTE)
        for n_ in M._notes[:2]:
            st.caption("ℹ️ " + n_)

        if not radar:
            st.info("Bu maç için belirgin bir 1/2, 2/1 veya +6 gol sürpriz potansiyeli tespit edilmedi.")
        for it in radar:
            tier = v2.TIER_LABEL[it["tier"]]
            title = "+6 Gol" if it["dir"] == "6+" else it["dir"]
            box = st.error if it["tier"] == 2 else st.warning if it["tier"] == 1 else st.info
            box(f"**{tier}: {title}**")
            k1, k2, k3 = st.columns(3)
            k1.metric("Model olasılığı", pct(it["pot"]))
            k2.metric("Kanıt gücü", f"%{100*it['evid']:.0f}" if it["evid"] > 0 else "—")
            ts = v2.TIER_STATS.get((it["dir"], it["tier"]))
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
            if v2.BACKTEST_NOTE:
                st.caption(v2.BACKTEST_NOTE)

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
        box(f"🚨 **Maç:** {m_['home']} – {m_['away']} (Tarih: {m_['date']:%d} {sn.MONTHS[m_['date'].month-1]}) · {m_['league']}")
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
        if v2.SEASONAL_NOTE:
            st.caption(v2.SEASONAL_NOTE)
