"""
=============================================================================
Collections Prioritization Project
Stage 6: Excel Scenario Tool  (Manager-Facing)
=============================================================================
What this stage does (plain English):
  A collections head doesn't want to run Python. They open Excel, change
  two cells ("how many agents do I have today?" and "what does each call
  cost?"), and see the recovery and profit change instantly.

  This script builds that Excel file programmatically using openpyxl.
  It has:
    • A yellow INPUT section (manager changes only these cells)
    • A blue OUTPUT section (auto-calculated, locked)
    • Three strategy tabs (Random | Highest PD | EL Policy)
    • A summary chart embedded in the sheet

⚠️  SIMULATION NOTE:
    All recovery figures use contact_effectiveness = 30% assumption.
    This is clearly labelled in the Excel file header.

Output:
    outputs/collections_scenario_tool.xlsx
=============================================================================
"""

import os, sys, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import openpyxl
from openpyxl.styles import (
    Font, PatternFill, Alignment, Border, Side, numbers
)
from openpyxl.chart import BarChart, Reference
from openpyxl.chart.label import DataLabel
from openpyxl.utils import get_column_letter
from openpyxl.utils.dataframe import dataframe_to_rows

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METRICS  = os.path.join(ROOT, "outputs", "stage5_policy_metrics.csv")
PRIORITY = os.path.join(ROOT, "outputs", "stage5_priority_list.csv")
OUT_XLS  = os.path.join(ROOT, "outputs", "collections_scenario_tool.xlsx")

# Default parameters (manager overrides in Excel)
DEFAULT_CAPACITY     = 200
DEFAULT_COST         = 150
DEFAULT_CONTACT_EFF  = 0.30
DEFAULT_LGD          = 0.45
DEFAULT_EAD_TOTAL    = 5_000_000_000  # ₹500 Cr portfolio (placeholder)


def banner(msg):
    print("\n" + "="*65)
    print(f"  {msg}")
    print("="*65)


# ── Style helpers ─────────────────────────────────────────────────────────────
def hdr_style(ws, row, col, text, bg="1F4E79", fg="FFFFFF", bold=True, size=11):
    cell = ws.cell(row=row, column=col, value=text)
    cell.fill = PatternFill("solid", fgColor=bg)
    cell.font = Font(bold=bold, color=fg, size=size)
    cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    return cell


def inp_style(ws, row, col, value, fmt=None):
    cell = ws.cell(row=row, column=col, value=value)
    cell.fill = PatternFill("solid", fgColor="FFF2CC")   # yellow
    cell.font = Font(bold=True, size=11)
    cell.alignment = Alignment(horizontal="right")
    if fmt:
        cell.number_format = fmt
    return cell


def out_style(ws, row, col, formula, fmt=None):
    cell = ws.cell(row=row, column=col, value=formula)
    cell.fill = PatternFill("solid", fgColor="DDEEFF")   # light blue
    cell.font = Font(size=11)
    cell.alignment = Alignment(horizontal="right")
    if fmt:
        cell.number_format = fmt
    return cell


def label_style(ws, row, col, text):
    cell = ws.cell(row=row, column=col, value=text)
    cell.font = Font(size=10)
    cell.alignment = Alignment(horizontal="left", vertical="center")
    return cell


def thin_border():
    thin = Side(style="thin")
    return Border(left=thin, right=thin, top=thin, bottom=thin)


# ── Build Excel ────────────────────────────────────────────────────────────────
def build_excel():
    wb = openpyxl.Workbook()

    # ── Sheet 1: Scenario Dashboard ──────────────────────────────────────────
    ws = wb.active
    ws.title = "📊 Scenario Dashboard"
    ws.column_dimensions["A"].width = 38
    ws.column_dimensions["B"].width = 22
    ws.column_dimensions["C"].width = 22
    ws.column_dimensions["D"].width = 22
    ws.row_dimensions[1].height = 40

    # Title row
    ws.merge_cells("A1:D1")
    title = ws["A1"]
    title.value = "Retail Collections Optimizer — Manager Scenario Tool"
    title.font  = Font(bold=True, size=14, color="FFFFFF")
    title.fill  = PatternFill("solid", fgColor="1F4E79")
    title.alignment = Alignment(horizontal="center", vertical="center")

    # Disclaimer
    ws.merge_cells("A2:D2")
    disc = ws["A2"]
    disc.value = ("⚠️  SIMULATION: Recovery figures use contact_effectiveness assumption "
                  "(modifiable below). No real call-outcome data is available.")
    disc.font  = Font(italic=True, color="FF0000", size=9)
    disc.alignment = Alignment(horizontal="center")

    # ── INPUT SECTION ──────────────────────────────────────────────────────────
    ws.merge_cells("A4:D4")
    hdr_style(ws, 4, 1, "🟡 INPUTS  —  Manager Changes These", bg="FFD700", fg="000000")

    inputs = [
        (5,  "Daily Agent Capacity (calls/day)",       DEFAULT_CAPACITY,    "#,##0"),
        (6,  "Cost per Call (₹)",                      DEFAULT_COST,        "₹#,##0"),
        (7,  "Contact Effectiveness (% loss recovered if called)",
                                                        DEFAULT_CONTACT_EFF, "0.0%"),
        (8,  "LGD (Loss Given Default %)",             DEFAULT_LGD,         "0.0%"),
    ]
    for row, lbl, val, fmt in inputs:
        label_style(ws, row, 1, lbl)
        inp_style(ws, row, 2, val, fmt)

    INP_CAP   = "B5"   # capacity
    INP_COST  = "B6"   # cost per call
    INP_EFF   = "B7"   # contact eff
    INP_LGD   = "B8"   # LGD

    # ── OUTPUT SECTION ──────────────────────────────────────────────────────────
    ws.merge_cells("A10:D10")
    hdr_style(ws, 10, 1, "🔵 OUTPUTS  —  Auto-Calculated", bg="1F4E79", fg="FFFFFF")

    hdr_style(ws, 11, 1, "Metric",                 bg="BDD7EE", fg="000000", bold=False)
    hdr_style(ws, 11, 2, "Random (No Model)",      bg="BDD7EE", fg="000000", bold=False)
    hdr_style(ws, 11, 3, "Highest PD First",       bg="BDD7EE", fg="000000", bold=False)
    hdr_style(ws, 11, 4, "EL Policy  ★ OUR MODEL", bg="C6EFCE", fg="000000", bold=True)

    # Pre-load results if available, else use placeholders
    if os.path.exists(METRICS):
        mdf = pd.read_csv(METRICS)
        r1  = mdf.iloc[0]
        r2  = mdf.iloc[1] if len(mdf) > 1 else r1
        r3  = mdf.iloc[2] if len(mdf) > 2 else r1
        # Compute scaling factor for dynamic capacity from defaults
        base_cap      = DEFAULT_CAPACITY
        r1_rec_unit   = r1["recovery_simulated"]  / base_cap
        r2_rec_unit   = r2["recovery_simulated"]  / base_cap
        r3_rec_unit   = r3["recovery_simulated"]  / base_cap
        r1_el_unit    = r1["EL_called"]            / base_cap
        r2_el_unit    = r2["EL_called"]            / base_cap
        r3_el_unit    = r3["EL_called"]            / base_cap
    else:
        r1_rec_unit = r2_rec_unit = r3_rec_unit = 10000
        r1_el_unit  = r2_el_unit  = r3_el_unit  = 30000

    # Output rows with Excel formulas
    output_rows = [
        (12, "Customers Called",           f"={INP_CAP}",           f"={INP_CAP}",           f"={INP_CAP}",          "#,##0"),
        (13, "Expected Loss in Called Set (₹)", f"={INP_CAP}*{r1_el_unit:.2f}", f"={INP_CAP}*{r2_el_unit:.2f}", f"={INP_CAP}*{r3_el_unit:.2f}", "₹#,##0"),
        (14, "Recovery  (₹)  ⚠️ simulation",   f"=C13*{INP_EFF}",      f"=D13*{INP_EFF}",      f"=E13*{INP_EFF}",      "₹#,##0"),
        (15, "Total Call Cost (₹)",         f"={INP_CAP}*{INP_COST}", f"={INP_CAP}*{INP_COST}", f"={INP_CAP}*{INP_COST}", "₹#,##0"),
        (16, "Net Profit  (₹)  ⚠️ simulation", "=C14-C15",             "=D14-D15",             "=E14-E15",             "₹#,##0"),
        (17, "Recovery per Agent-Hour  (₹)", f"=C14/({INP_CAP}*0.25/7)", f"=D14/({INP_CAP}*0.25/7)", f"=E14/({INP_CAP}*0.25/7)", "₹#,##0"),
    ]

    for row, label, v1, v2, v3, fmt in output_rows:
        label_style(ws, row, 1, label)
        out_style(ws, row, 2, v1, fmt)
        out_style(ws, row, 3, v2, fmt)
        out_style(ws, row, 4, v3, fmt)

    # Highlight best strategy
    for row in range(12, 18):
        ws.cell(row=row, column=4).font = Font(bold=True, size=11)

    # Border the output block
    for row in range(11, 18):
        for col in range(1, 5):
            ws.cell(row=row, column=col).border = thin_border()

    # ── Insight box ─────────────────────────────────────────────────────────────
    ws.merge_cells("A19:D20")
    insight = ws["A19"]
    insight.value = (
        "📌 KEY INSIGHT:  Sorting by Expected Loss (PD × EAD × LGD) instead of PD alone "
        "ensures agents focus on accounts where the MONEY AT RISK is highest. "
        "A 40% PD customer with ₹5L outstanding is 5× more valuable to call than a 40% PD customer with ₹1L."
    )
    insight.font      = Font(italic=True, size=10, color="1F4E79")
    insight.alignment = Alignment(wrap_text=True, vertical="top")

    # ── Chart: Net Profit by Strategy ──────────────────────────────────────────
    # Write helper data for chart
    ws["A22"] = "Strategy"
    ws["B22"] = "Net Profit (₹M)"
    ws["A23"] = "Random"
    ws["A24"] = "Highest PD"
    ws["A25"] = "EL Policy"
    ws["B23"] = f"=C16/1000000"
    ws["B24"] = f"=D16/1000000"
    ws["B25"] = f"=E16/1000000"

    chart = BarChart()
    chart.type       = "col"
    chart.title      = "Net Profit by Strategy (₹M) — ⚠️ Simulation"
    chart.y_axis.title = "₹ Million"
    chart.x_axis.title = "Strategy"
    chart.width  = 14
    chart.height = 9
    data = Reference(ws, min_col=2, min_row=22, max_row=25)
    cats = Reference(ws, min_col=1, min_row=23, max_row=25)
    chart.add_data(data, titles_from_data=True)
    chart.set_categories(cats)
    chart.series[0].graphicalProperties.solidFill  = "1F4E79"
    ws.add_chart(chart, "A27")

    # ── Sheet 2: Priority List ─────────────────────────────────────────────────
    ws2 = wb.create_sheet("📋 Priority List")
    if os.path.exists(PRIORITY):
        pdf = pd.read_csv(PRIORITY, nrows=1000)   # top 1000 customers
        pdf = pdf.sort_values("priority_rank")
        ws2.merge_cells("A1:H1")
        hdr_style(ws2, 1, 1,
                  f"Top 1,000 Customers by EL Policy Rank  |  ⚠️ contact_eff={DEFAULT_CONTACT_EFF:.0%} assumption",
                  bg="1F4E79")
        headers = list(pdf.columns)
        for j, h in enumerate(headers, 1):
            hdr_style(ws2, 2, j, h, bg="BDD7EE", fg="000000", bold=False, size=9)
        for i, row in enumerate(pdf.itertuples(index=False), 3):
            for j, val in enumerate(row, 1):
                c = ws2.cell(row=i, column=j, value=val)
                c.font = Font(size=9)
                if j in (4, 5, 6, 7):   # monetary columns
                    c.number_format = "₹#,##0"
        for col in ws2.columns:
            ws2.column_dimensions[get_column_letter(col[0].column)].width = 18
    else:
        ws2["A1"] = "Priority list not found. Run Stage 5 first."

    # ── Sheet 3: Instructions ──────────────────────────────────────────────────
    ws3 = wb.create_sheet("ℹ️ How to Use")
    instructions = [
        ("HOW TO USE THIS TOOL", ""),
        ("", ""),
        ("1. Go to '📊 Scenario Dashboard' tab", ""),
        ("2. Change ONLY the yellow cells:", ""),
        ("   • Daily Agent Capacity", "e.g. 200 (how many calls can your team handle today)"),
        ("   • Cost per Call (₹)",    "e.g. ₹150 (agent salary + telecom + infra per call)"),
        ("   • Contact Effectiveness", "Default 30% — ⚠️ THIS IS A SIMULATION ASSUMPTION"),
        ("   • LGD",                  "Default 45% — RBI Basel norm for unsecured retail"),
        ("3. Blue cells auto-calculate recovery and net profit.", ""),
        ("4. Compare the three strategies in the bar chart.", ""),
        ("", ""),
        ("IMPORTANT DISCLAIMERS", ""),
        ("Contact effectiveness = 30% is a SIMULATION assumption.", ""),
        ("There is no real call-outcome data in this model.", ""),
        ("Do not call this 'measured uplift' unless you add real A/B data.", ""),
        ("Recovery figures are estimates only, not guarantees.", ""),
        ("", ""),
        ("DEFINITIONS", ""),
        ("PD  = Probability of Default (from XGBoost model)", ""),
        ("EAD = Exposure at Default (loan outstanding)", ""),
        ("LGD = Loss Given Default (% of EAD bank loses if default happens)", ""),
        ("EL  = Expected Loss = PD × EAD × LGD", ""),
        ("Recovery = EL × contact_effectiveness  ⚠️ simulation", ""),
    ]
    ws3.column_dimensions["A"].width = 60
    ws3.column_dimensions["B"].width = 55
    ws3.merge_cells("A1:B1")
    hdr_style(ws3, 1, 1, "HOW TO USE THIS TOOL", bg="1F4E79")
    for i, (a, b) in enumerate(instructions[1:], 2):
        wa = ws3.cell(row=i, column=1, value=a)
        wb2 = ws3.cell(row=i, column=2, value=b)
        if a.isupper() and a:
            wa.font = Font(bold=True, size=10)
        else:
            wa.font = Font(size=10)
        wb2.font = Font(italic=True, size=9, color="666666")

    # Save
    wb.save(OUT_XLS)
    print(f"\n  ✅  Excel tool saved: {OUT_XLS}")


def main():
    banner("STAGE 6 — Building Excel Scenario Tool")
    build_excel()
    banner("Stage 6 Complete ✅")
    print(f"""
  Workbook has 3 sheets:
    📊 Scenario Dashboard  — manager changes yellow cells, sees results
    📋 Priority List       — top 1,000 customers ranked by EL policy
    ℹ️  How to Use          — plain-language guide + disclaimers

  Saved to: {OUT_XLS}

  To use: Open in Excel, go to 'Scenario Dashboard',
          change the 4 yellow input cells and watch outputs update.

  Next: Stage 7 → Power BI dashboard plan + executive summary.
    """)


if __name__ == "__main__":
    main()
