from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    ListFlowable,
    ListItem,
    Preformatted,
)

source = Path("docs/poisonshield_technical_documentation.md")
output = Path("docs/PoisonShield_Technical_Documentation.pdf")

styles = getSampleStyleSheet()
normal = styles["BodyText"]
normal.fontName = "Helvetica"
normal.fontSize = 10.5
normal.leading = 14
normal.spaceAfter = 6

h1 = ParagraphStyle("H1", parent=styles["Heading1"], fontSize=17, spaceBefore=16, spaceAfter=8)
h2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=13.5, spaceBefore=12, spaceAfter=6)
h3 = ParagraphStyle("H3", parent=styles["Heading3"], fontSize=11.5, spaceBefore=10, spaceAfter=4)
code_style = ParagraphStyle("Code", parent=normal, fontName="Courier", fontSize=8.5, leading=10.5, spaceAfter=8)

doc = SimpleDocTemplate(
    str(output),
    pagesize=A4,
    leftMargin=2 * cm,
    rightMargin=2 * cm,
    topMargin=2 * cm,
    bottomMargin=2 * cm,
)

story = []
in_code = False
code_lines = []
bullets = []

def flush_bullets():
    global bullets
    if bullets:
        story.append(
            ListFlowable(
                [ListItem(Paragraph(escape(item), normal), bulletColor="#0e7490") for item in bullets],
                bulletType="bullet",
                start="circle",
                leftIndent=18,
            )
        )
        story.append(Spacer(1, 4))
        bullets = []

def flush_code():
    global code_lines
    if code_lines:
        story.append(Preformatted("\n".join(code_lines), code_style))
        story.append(Spacer(1, 6))
        code_lines = []

for raw_line in source.read_text(encoding="utf-8").splitlines():
    line = raw_line.rstrip()
    if line.startswith("```"):
        if in_code:
            flush_code()
            in_code = False
        else:
            flush_bullets()
            in_code = True
        continue
    if in_code:
        code_lines.append(line)
        continue
    if not line.strip():
        flush_bullets()
        story.append(Spacer(1, 6))
        continue
    if line.startswith("# "):
        flush_bullets()
        story.append(Paragraph(escape(line[2:]), h1))
        continue
    if line.startswith("## "):
        flush_bullets()
        story.append(Paragraph(escape(line[3:]), h2))
        continue
    if line.startswith("### "):
        flush_bullets()
        story.append(Paragraph(escape(line[4:]), h3))
        continue
    if line.startswith("- ") or line.startswith("* "):
        bullets.append(line[2:])
        continue
    flush_bullets()
    story.append(Paragraph(escape(line), normal))

flush_bullets()
flush_code()

doc.build(story)
print(f"Created {output} ({output.stat().st_size} bytes)")