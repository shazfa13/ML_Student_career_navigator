"""Professional PDF report generation for the student success workspace."""

from datetime import date
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    BaseDocTemplate,
    Flowable,
    Frame,
    KeepTogether,
    PageTemplate,
    PageBreak,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

NAVY = colors.HexColor("#17233f")
INK = colors.HexColor("#26324d")
MUTED = colors.HexColor("#596780")
PURPLE = colors.HexColor("#6552c7")
LINE = colors.HexColor("#d9e1ee")
PALE = colors.HexColor("#f3f1ff")
GREEN = colors.HexColor("#187342")
AMBER = colors.HexColor("#87630b")
RED = colors.HexColor("#a42f2f")


def _text(value):
    return escape(str(value if value is not None else ""))


def _p(value, style):
    return Paragraph(_text(value).replace("\n", "<br/>"), style)


class CareerFlowChart(Flowable):
    """Draw the career-guidance sequence without requiring Graphviz."""

    def __init__(self, steps, width=170 * mm):
        super().__init__()
        self.steps = steps
        self.width = width
        self.height = max(42 * mm, len(steps) * 13 * mm)

    def draw(self):
        canvas = self.canv
        box_width = self.width * 0.72
        box_height = 8 * mm
        x = (self.width - box_width) / 2
        y = self.height - box_height
        for index, step in enumerate(self.steps):
            canvas.setFillColor(PALE if index % 2 == 0 else colors.HexColor("#eaf5ff"))
            canvas.setStrokeColor(PURPLE)
            canvas.roundRect(x, y, box_width, box_height, 2 * mm, fill=1, stroke=1)
            canvas.setFillColor(NAVY)
            canvas.setFont("Helvetica-Bold", 9)
            canvas.drawCentredString(x + box_width / 2, y + 3 * mm, str(step))
            if index < len(self.steps) - 1:
                canvas.setStrokeColor(PURPLE)
                canvas.setLineWidth(1.2)
                arrow_x = x + box_width / 2
                canvas.line(arrow_x, y - 1 * mm, arrow_x, y - 5 * mm)
                canvas.line(arrow_x, y - 5 * mm, arrow_x - 1.5 * mm, y - 3.2 * mm)
                canvas.line(arrow_x, y - 5 * mm, arrow_x + 1.5 * mm, y - 3.2 * mm)
            y -= 13 * mm


def _table(data, widths, styles, header=True):
    table_data = []
    for row_index, row in enumerate(data):
        table_data.append([
            _p(value, styles["table_header"] if header and row_index == 0 else styles["table_cell"])
            for value in row
        ])
    table = Table(table_data, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
    commands = [
        ("GRID", (0, 0), (-1, -1), 0.45, LINE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]
    if header:
        commands.extend([
            ("BACKGROUND", (0, 0), (-1, 0), NAVY),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ])
        for row_index in range(1, len(table_data)):
            if row_index % 2 == 0:
                commands.append(("BACKGROUND", (0, row_index), (-1, row_index), colors.HexColor("#f7f8fc")))
    table.setStyle(TableStyle(commands))
    return table


def _section(title, styles):
    return [Spacer(1, 5 * mm), Paragraph(_text(title), styles["section"]), Spacer(1, 2 * mm)]


def _risk_color(label):
    return {"Low Risk": GREEN, "Medium Risk": AMBER, "High Risk": RED}.get(label, INK)


class ReportDocument(BaseDocTemplate):
    def __init__(self, filename, styles):
        super().__init__(filename, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=18 * mm, bottomMargin=17 * mm)
        frame = Frame(self.leftMargin, self.bottomMargin, self.width, self.height, id="normal")
        self.addPageTemplates([PageTemplate(id="report", frames=frame, onPage=self._header_footer)])
        self.report_styles = styles

    def _header_footer(self, canvas, document):
        canvas.saveState()
        if document.page > 1:
            canvas.setStrokeColor(LINE)
            canvas.line(self.leftMargin, A4[1] - 13 * mm, A4[0] - self.rightMargin, A4[1] - 13 * mm)
            canvas.setFont("Helvetica-Bold", 8)
            canvas.setFillColor(PURPLE)
            canvas.drawString(self.leftMargin, A4[1] - 10 * mm, "ML-Based Student Success & Career Navigator")
        canvas.setStrokeColor(LINE)
        canvas.line(self.leftMargin, 12 * mm, A4[0] - self.rightMargin, 12 * mm)
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(MUTED)
        canvas.drawString(self.leftMargin, 8 * mm, "ML-Based Student Success & Career Navigator")
        canvas.drawRightString(A4[0] - self.rightMargin, 8 * mm, f"Page {document.page}")
        canvas.restoreState()


def generate_student_report(filename, report_data):
    """Build a multi-page report from data already generated by the app."""
    base = getSampleStyleSheet()
    styles = {
        "cover_title": ParagraphStyle("cover_title", parent=base["Title"], fontName="Helvetica-Bold", fontSize=29, leading=34, textColor=NAVY, alignment=TA_CENTER, spaceAfter=8 * mm),
        "cover_subtitle": ParagraphStyle("cover_subtitle", parent=base["Normal"], fontName="Helvetica", fontSize=13, leading=18, textColor=MUTED, alignment=TA_CENTER),
        "cover_meta": ParagraphStyle("cover_meta", parent=base["Normal"], fontName="Helvetica-Bold", fontSize=11, leading=17, textColor=INK, alignment=TA_CENTER),
        "section": ParagraphStyle("section", parent=base["Heading2"], fontName="Helvetica-Bold", fontSize=16, leading=20, textColor=NAVY, spaceBefore=4 * mm, keepWithNext=True),
        "body": ParagraphStyle("body", parent=base["BodyText"], fontName="Helvetica", fontSize=9.5, leading=14, textColor=INK),
        "small": ParagraphStyle("small", parent=base["BodyText"], fontName="Helvetica", fontSize=8.5, leading=12, textColor=MUTED),
        "table_header": ParagraphStyle("table_header", parent=base["BodyText"], fontName="Helvetica-Bold", fontSize=8.5, leading=11, textColor=colors.white),
        "table_cell": ParagraphStyle("table_cell", parent=base["BodyText"], fontName="Helvetica", fontSize=8.5, leading=11, textColor=INK),
        "callout": ParagraphStyle("callout", parent=base["BodyText"], fontName="Helvetica-Bold", fontSize=11, leading=15, textColor=NAVY, alignment=TA_CENTER),
    }
    document = ReportDocument(filename, styles)
    story = []
    name = report_data.get("student_name", "Student")
    target = report_data.get("target_career", "Not selected")
    risk = report_data.get("current_risk", "Not predicted")
    compatibility = report_data.get("compatibility_score", 0)
    story.extend([Spacer(1, 25 * mm), Paragraph("ML-Based Student Success & Career Navigator", styles["cover_title"]), Spacer(1, 25 * mm)])
    story.append(Table([[Paragraph(_text(name), styles["cover_meta"])], [Paragraph(_text(report_data.get("student_year", "")), styles["cover_meta"])], [Paragraph(_text(target), styles["cover_meta"])], [Paragraph(date.today().strftime("%d %B %Y"), styles["cover_meta"])]], colWidths=[110 * mm], hAlign="CENTER", style=TableStyle([("BACKGROUND", (0, 0), (-1, -1), PALE), ("BOX", (0, 0), (-1, -1), 0.8, LINE), ("TOPPADDING", (0, 0), (-1, -1), 8), ("BOTTOMPADDING", (0, 0), (-1, -1), 8)])))
    story.extend([Spacer(1, 28 * mm), Paragraph("Personalized Academic Risk, Career Guidance and Skill Development Report", styles["cover_subtitle"]), PageBreak()])

    story.extend(_section("1. Student Profile", styles))
    story.append(_table([["Field", "Value"], ["Student Name", name], ["Year", report_data.get("student_year", "")], ["CGPA", f'{report_data.get("student_cgpa", 0):.1f}'], ["Target Career", target], ["Existing Technical Skills", ", ".join(report_data.get("student_skills", [])) or "None selected"]], [55 * mm, 112 * mm], styles))

    story.extend(_section("2. Academic Performance", styles))
    academic = report_data.get("academic", {})
    academic_rows = [["Indicator", "Current Value"], ["Attendance", f'{academic.get("Attendance", 0):g}%'], ["Assignment Completion", f'{academic.get("Assignment_Completion", 0):g}%'], ["Quiz Average", f'{academic.get("Quiz_Average", 0):g}%'], ["Previous Marks", f'{academic.get("Previous_Marks", 0):g}%'], ["Study Hours", f'{academic.get("Study_Hours", 0):g} per week']]
    story.append(_table(academic_rows, [85 * mm, 82 * mm], styles))
    story.append(Spacer(1, 3 * mm))
    for label, value in academic.items():
        normalized = float(value) / (40 if label == "Study_Hours" else 100)
        story.append(Paragraph(_text(label.replace("_", " ")), styles["small"]))
        story.append(Table([[""]], colWidths=[167 * mm * max(0, min(normalized, 1))], style=TableStyle([("BACKGROUND", (0, 0), (-1, -1), PURPLE), ("LINEBELOW", (0, 0), (-1, -1), 5, PALE)])))

    story.extend(_section("3. Academic Risk Prediction", styles))
    risk_para = Paragraph(f"Predicted Risk: <font color='{_risk_color(risk).hexval()}'><b>{_text(risk)}</b></font>", styles["callout"])
    story.append(Table([[risk_para]], colWidths=[167 * mm], style=TableStyle([("BACKGROUND", (0, 0), (-1, -1), PALE), ("BOX", (0, 0), (-1, -1), 0.8, LINE), ("TOPPADDING", (0, 0), (-1, -1), 10), ("BOTTOMPADDING", (0, 0), (-1, -1), 10)])))
    story.append(Spacer(1, 3 * mm))
    story.append(Paragraph("This risk result is an estimate for learning support and is not a guarantee about future academic outcomes.", styles["body"]))

    story.extend(_section("4. Career Guidance", styles))
    story.append(_table([["Measure", "Result"], ["Target Career", target], ["Recommended Career Match", report_data.get("recommended_career", target)], ["Career Compatibility", f"{compatibility:.1f}%"], ["Existing Skills", ", ".join(report_data.get("student_skills", [])) or "None"], ["Required Skills", ", ".join(report_data.get("required_skills", [])) or "None"], ["Skill Gap", ", ".join(report_data.get("missing_skills", [])) or "No missing skills identified"]], [55 * mm, 112 * mm], styles))
    story.extend(_section("Career Guidance Flow", styles))
    story.append(CareerFlowChart(["Student Profile", "Academic & Skill Data", "Career Selection", "Career-Skill Matching", "Compatibility Analysis", "Skill Gap Identification", "Project Recommendations", "Learning Learning Path", "Career Readiness"]))

    story.extend(_section("5. Skill Gap Analysis", styles))
    gap_rows = [["Skill", "Importance", "Status", "Priority"]]
    for row in report_data.get("skill_gaps", []):
        gap_rows.append([row.get("Required Skill", ""), f'{float(row.get("Importance", 0)):.1f}', "✅ Existing" if "Yes" in row.get("Have It?", "") else "⚠️ Missing", row.get("Priority if Missing", "-")])
    story.append(_table(gap_rows, [54 * mm, 28 * mm, 40 * mm, 45 * mm], styles))

    story.extend(_section("6. Recommended Projects", styles))
    for project in report_data.get("projects", []):
        existing = [skill for skill in project.get("required_skills", []) if skill in report_data.get("student_skills", [])]
        missing = [skill for skill in project.get("required_skills", []) if skill not in existing]
        project_rows = [
            ["Project", project.get("project_name", "")],
            ["Difficulty / Relevance", f'{project.get("difficulty", "")} / {project.get("relevance", 0)}%'],
            ["Why Recommended", project.get("why", "")],
            ["Required Skills", ", ".join(project.get("required_skills", []))],
            ["Existing / Missing Skills", f'Existing: {", ".join(existing) or "None"}; Missing: {", ".join(missing) or "None"}'],
            ["Skills Gained", ", ".join(project.get("skills_gained", []))],
        ]
        story.append(KeepTogether([_table(project_rows, [45 * mm, 122 * mm], styles, header=False), Spacer(1, 3 * mm)]))

    story.extend(_section("7. Learning Learning Path", styles))
    roadmap = report_data.get("roadmap", [])
    roadmap_steps = ["Current Skills"] + [item.get("skill", "") for item in roadmap[:3]] + ["Recommended Projects", "Career Preparation"]
    story.append(CareerFlowChart(roadmap_steps))
    story.append(Spacer(1, 3 * mm))
    story.append(Paragraph("Priority learning sequence: " + (" → ".join(item.get("skill", "") for item in roadmap) if roadmap else "No additional skills are currently identified."), styles["body"]))

    story.extend(_section("8. Final Student Summary", styles))
    summary = report_data.get("missing_skills", [])
    story.append(Paragraph(f"The analysis indicates a current academic risk level of <b>{_text(risk)}</b> and a {compatibility:.1f}% compatibility score for <b>{_text(target)}</b>. The recommended development areas are { _text(', '.join(summary[:5]) if summary else 'continued practice across the selected career skills') }. Based on the current profile, the recommended project direction is to build practical evidence in the target career while following the priority learning roadmap. These comparisons support planning and do not predict employment or placement outcomes.", styles["body"]))
    document.build(story)
