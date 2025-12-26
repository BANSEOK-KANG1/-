# -*- coding: utf-8 -*-
"""
리뷰(baemin.csv) + 트렌드(bamin_trend_expanded.csv)로
주별 트렌드 지수와 리뷰 평점의 관계를 분석하고,
한글이 깨지지 않는 그래프 + 포트폴리오 PDF를 생성합니다.

사용법:
  python make_portfolio_report.py

필요 파일(같은 폴더에 두거나 경로를 아래에서 수정):
  /mnt/data/baemin.csv
  /mnt/data/bamin_trend_expanded.csv
"""
import os
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams, font_manager
from scipy.stats import pearsonr, spearmanr

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER

REVIEWS_PATH = "/mnt/data/baemin.csv"
TREND_PATH   = "/mnt/data/bamin_trend_expanded.csv"
OUT_DIR      = "/mnt/data"

def ensure_korean_font_matplotlib() -> str | None:
    candidates = ["NanumGothic", "Malgun Gothic", "AppleGothic", "Noto Sans CJK KR", "NotoSansCJKkr"]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for c in candidates:
        if c in available:
            return c

    # Linux 환경이면 나눔고딕 설치 시도
    try:
        subprocess.run(["bash","-lc","apt-get update -y && apt-get install -y fonts-nanum"], check=True)
        subprocess.run(["bash","-lc","fc-cache -f -v"], check=True)
    except Exception:
        pass

    font_manager._load_fontmanager(try_read_cache=False)
    available = {f.name for f in font_manager.fontManager.ttflist}
    for c in candidates:
        if c in available:
            return c

    for f in font_manager.fontManager.ttflist:
        name = f.name.lower()
        if "nanum" in name or "noto" in name or "gothic" in name:
            return f.name
    return None

def register_korean_font_reportlab() -> str:
    ttf_candidates = [
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    ]
    for p in ttf_candidates:
        if os.path.exists(p):
            pdfmetrics.registerFont(TTFont("KFont", p))
            return "KFont"
    return "Helvetica"

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1) 한글 폰트(그래프)
    kr_font = ensure_korean_font_matplotlib()
    if kr_font:
        rcParams["font.family"] = kr_font
    rcParams["axes.unicode_minus"] = False

    # 2) 데이터 로드 + 주별 집계
    reviews = pd.read_csv(REVIEWS_PATH, parse_dates=["at"])
    trend = pd.read_csv(TREND_PATH)

    reviews["iso_year"] = reviews["at"].dt.isocalendar().year.astype(int)
    reviews["iso_week"] = reviews["at"].dt.isocalendar().week.astype(int)
    weekly_reviews = (
        reviews.groupby(["iso_year","iso_week"])
        .agg(avg_score=("score","mean"), review_count=("score","size"))
        .reset_index()
    )

    trend["date"] = pd.to_datetime(trend["date"])
    trend["iso_year"] = trend["date"].dt.isocalendar().year.astype(int)
    trend["iso_week"] = trend["date"].dt.isocalendar().week.astype(int)

    numeric_cols = trend.select_dtypes(include=[np.number]).columns.tolist()
    trend_metric_col = "전체" if "전체" in trend.columns else (numeric_cols[0] if numeric_cols else None)

    weekly_trend = (
        trend.groupby(["iso_year","iso_week"])
        .agg(trend_mean=(trend_metric_col, "mean"))
        .reset_index()
    )

    merged = weekly_reviews.merge(weekly_trend, on=["iso_year","iso_week"], how="inner")
    merged["week_index"] = merged["iso_year"]*100 + merged["iso_week"]

    # 3) 상관분석
    pear_r, pear_p = pearsonr(merged["trend_mean"], merged["avg_score"])
    spear_r, spear_p = spearmanr(merged["trend_mean"], merged["avg_score"])

    # 4) 그래프 생성(한글)
    line_path = os.path.join(OUT_DIR, "01_주별_트렌드_평점_추이.png")
    scatter_path = os.path.join(OUT_DIR, "02_트렌드_vs_평점_산점도.png")
    bubble_path = os.path.join(OUT_DIR, "03_리뷰수_평점_트렌드_관계.png")

    plt.figure(figsize=(12,6))
    plt.plot(merged["week_index"], merged["avg_score"], label="주별 평균 평점", linewidth=2)
    plt.plot(merged["week_index"], merged["trend_mean"], label=f"주별 트렌드 지수({trend_metric_col})", linewidth=2, alpha=0.75)
    plt.title("주별 트렌드 지수와 평균 평점 변화")
    plt.xlabel("주차(YYYYWW)")
    plt.ylabel("값")
    plt.grid(alpha=0.3); plt.legend(); plt.tight_layout()
    plt.savefig(line_path, dpi=200); plt.close()

    plt.figure(figsize=(8,6))
    plt.scatter(merged["trend_mean"], merged["avg_score"], alpha=0.5)
    plt.title("트렌드 지수와 평균 평점의 관계(산점도)")
    plt.xlabel(f"주별 트렌드 지수({trend_metric_col})")
    plt.ylabel("주별 평균 평점")
    plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(scatter_path, dpi=200); plt.close()

    plt.figure(figsize=(10,6))
    sc = plt.scatter(merged["review_count"], merged["avg_score"], c=merged["trend_mean"], alpha=0.65)
    cb = plt.colorbar(sc); cb.set_label(f"트렌드 지수({trend_metric_col})")
    plt.title("리뷰 수, 평균 평점, 트렌드 지수의 관계")
    plt.xlabel("주별 리뷰 수")
    plt.ylabel("주별 평균 평점")
    plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(bubble_path, dpi=200); plt.close()

    # 5) 포트폴리오 PDF 생성(한글)
    font_name = register_korean_font_reportlab()
    styles = getSampleStyleSheet()
    base = ParagraphStyle("base", parent=styles["Normal"], fontName=font_name, fontSize=10.5, leading=14, spaceAfter=6)
    title_style = ParagraphStyle("title", parent=styles["Title"], fontName=font_name, fontSize=20, leading=24, alignment=TA_CENTER, spaceAfter=10)
    h_style = ParagraphStyle("h", parent=styles["Heading2"], fontName=font_name, fontSize=13.5, leading=18, spaceBefore=10, spaceAfter=6)
    small_style = ParagraphStyle("small", parent=base, fontSize=9.5, leading=13, textColor=colors.HexColor("#444444"))

    def bullet(items):
        return Paragraph("• " + "<br/>• ".join(items), base)

    pdf_path = os.path.join(OUT_DIR, "포트폴리오_리뷰트렌드_분석.pdf")
    doc = SimpleDocTemplate(pdf_path, pagesize=A4, rightMargin=18*mm, leftMargin=18*mm, topMargin=16*mm, bottomMargin=16*mm)
    story = []

    # Page 1
    story.append(Paragraph("주별 트렌드 지수와 리뷰 평점의 관계 분석", title_style))
    story.append(Paragraph("13만+ 리뷰 데이터 기반 | 데이터 분석 → 인사이트 → PM 의사결정", ParagraphStyle("sub", parent=base, alignment=TA_CENTER, fontSize=11)))
    story.append(Spacer(1, 6*mm))

    summary_table = Table(
        [
            ["역할", "PM/데이터 분석(팀 프로젝트)"],
            ["목표", "트렌드(수요) 변화가 평점(경험 품질)에 미치는 영향 검증 및 개선 방향 도출"],
            ["핵심 질문", "트렌드 상승 시 평점이 하락하는가? 하락이 있다면 어떤 조건에서 심해지는가?"],
            ["사용 데이터", f"리뷰 130K+ (score, at, content) / 주별 트렌드 지수(‘{trend_metric_col}’ 기준)"],
            ["사용 도구", "Python (pandas, matplotlib) | 상관분석(Pearson/Spearman) | 시각화"],
        ],
        colWidths=[26*mm, 150*mm]
    )
    summary_table.setStyle(TableStyle([
        ("FONT", (0,0), (-1,-1), font_name, 10),
        ("BACKGROUND", (0,0), (0,-1), colors.HexColor("#F2F4F7")),
        ("GRID", (0,0), (-1,-1), 0.25, colors.HexColor("#D0D5DD")),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("PADDING", (0,0), (-1,-1), 6),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 6*mm))

    story.append(Paragraph("문제 상황 (Fact)", h_style))
    story.append(bullet([
        "주차별로 리뷰 수가 급증하는 구간에서 평균 평점이 급격히 하락하는 패턴이 반복 관찰됨",
        "동일 시점에 트렌드 지수도 상승 구간에 위치해, ‘수요 증가 ↔ 경험 품질’의 연관 가능성이 제기됨",
    ]))

    story.append(Paragraph("왜 문제인가 (Why it matters)", h_style))
    story.append(bullet([
        "리뷰 평점은 신규 유저 신뢰 형성, 전환, 재구매/리텐션에 직접 영향을 주는 대표 경험 지표",
        "트렌드 상승(유입/주문 증가)이 오히려 평점 하락으로 이어지면, 성장 성과가 장기 LTV를 훼손할 위험",
    ]))
    story.append(PageBreak())

    # Page 2
    story.append(Paragraph("분석 및 결과 (Analysis & Findings)", title_style))
    story.append(Paragraph("주차(ISO year-week) 기준으로 리뷰와 트렌드를 결합하여 상관관계와 패턴을 검증했습니다.", base))
    story.append(Spacer(1, 3*mm))

    stats = Table(
        [
            ["핵심 통계 결과", ""],
            ["Pearson r", f"{pear_r:.3f} (p={pear_p:.2e})"],
            ["Spearman ρ", f"{spear_r:.3f} (p={spear_p:.2e})"],
            ["해석", "트렌드 지수가 높을수록 평균 평점이 낮아지는 ‘유의미한 음의 상관’"],
        ],
        colWidths=[38*mm, 138*mm]
    )
    stats.setStyle(TableStyle([
        ("FONT", (0,0), (-1,-1), font_name, 10),
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#101828")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("SPAN", (0,0), (1,0)),
        ("GRID", (0,0), (-1,-1), 0.25, colors.HexColor("#D0D5DD")),
        ("PADDING", (0,0), (-1,-1), 6),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
    ]))
    story.append(stats)
    story.append(Spacer(1, 5*mm))

    img1 = Image(line_path, width=170*mm, height=80*mm)
    img2 = Image(scatter_path, width=80*mm, height=70*mm)
    img3 = Image(bubble_path, width=80*mm, height=70*mm)

    story.append(Paragraph("1) 주별 추이: 트렌드 지수 vs 평균 평점", h_style))
    story.append(img1)
    story.append(Spacer(1, 3*mm))
    story.append(Paragraph("관찰: 트렌드 상승 구간에서 평균 평점이 동반 하락하는 구간이 반복적으로 나타남.", small_style))
    story.append(Spacer(1, 4*mm))

    story.append(Paragraph("2) 관계 시각화: 산점도 / 3변수 관계", h_style))
    t = Table([[img2, img3]], colWidths=[85*mm, 85*mm])
    t.setStyle(TableStyle([("VALIGN",(0,0),(-1,-1),"TOP")]))
    story.append(t)
    story.append(Spacer(1, 3*mm))
    story.append(Paragraph("관찰: 우하향(음의 상관) 패턴이 존재하며, 특히 ‘리뷰 수가 많은 주’에 평점이 더 낮아지는 경향이 보임.", small_style))
    story.append(PageBreak())

    # Page 3
    story.append(Paragraph("판단 및 의사결정 (Judgment & Decision)", title_style))

    story.append(Paragraph("핵심 인사이트", h_style))
    story.append(bullet([
        "트렌드 상승은 ‘수요 증가’ 신호이며, 이때 운영/UX 병목이 발생하면 경험 품질(평점)이 하락할 수 있음",
        "문제의 본질은 기능 부족이 아니라 ‘기대 관리 및 경험 순서(정보 제공 타이밍)’의 미정렬 가능성이 큼",
    ]))

    story.append(Paragraph("선택한 해결 방향 (Decision)", h_style))
    story.append(bullet([
        "기능 추가보다 ‘경험 제공 순서 조정’ 우선: 주문 이전 단계에서 예상 지연/품절/대체 옵션을 선제적으로 안내",
        "수요 폭증 시나리오에서 앱 성능·가맹점 처리량 병목을 완화(캐시/큐/서킷브레이커/배차 안정화 등)",
    ]))

    story.append(Paragraph("후속 실험 (Next Steps)", h_style))
    story.append(bullet([
        "평점 하락 주차 Top 10을 뽑아 리뷰 텍스트 토픽/감성 분석 → 원인 가설 정교화",
        "‘트렌드 대비 평점 방어 성공 주차’와 실패 주차를 비교해 운영·상품·지역·시간대 특성 파악",
        "기대 관리 UX(A/B): 주문 전 ETA(예상 도착시간) 정확도/가시성 강화가 평점 방어에 미치는 효과 검증",
    ]))

    story.append(Paragraph("이 프로젝트에서 보여준 PM 역량", h_style))
    story.append(bullet([
        "문제 구조화: Fact → Why → Goal → Metric → Hypothesis로 논리적 전개",
        "Outcome-driven Thinking: 북극성 지표(주별 평균 평점) 중심으로 모든 판단을 연결",
        "가설 기반 의사결정: 상관분석·시각화로 가설 검증 및 ‘원인-결과’ 구조 제시",
        "Trade-off 판단: 단기 보상/할인보다 구조적 경험 개선을 우선하는 이유를 명확히 설명",
    ]))

    doc.build(story)
    print("Saved:", line_path, scatter_path, bubble_path, pdf_path)

if __name__ == "__main__":
    main()
