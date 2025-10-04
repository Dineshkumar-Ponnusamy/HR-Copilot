"""Sample policy PDF generator for HR Policy Copilot.

This script generates sample PDF documents containing HR policies for testing
and demonstration purposes. It uses ReportLab to create PDF files with
multi-page content that can be indexed by the build_index.py script.

The generated PDFs include sample policies for Code of Conduct, Leave Policy,
and Travel & Expense Policy to simulate real HR documentation.
"""

import os
from reportlab.lib.pagesizes import LETTER
from reportlab.pdfgen import canvas

# Directory where sample policy PDFs will be generated
DATA_DIR = "data_policies"
os.makedirs(DATA_DIR, exist_ok=True)

# Dictionary defining sample PDF content: filename -> list of page texts
docs = {
    "Code_of_Conduct.pdf": [
        "Code of Conduct\n\nAll employees must follow ethical guidelines.\nHarassment is strictly prohibited.\nSee page 2 for reporting procedures.",
        "Reporting Procedures\n\nReport incidents to HR within 5 business days.\nAnonymous hotline is available."
    ],
    "Leave_Policy.pdf": [
        "Leave Policy Overview\n\nEmployees receive 15 days of paid annual leave.\nSick leave: 10 days per calendar year.\nCarryover limit: 5 days to next year.",
        "Parental Leave\n\n12 weeks paid parental leave for eligible employees.\nContractors are not eligible for parental leave."
    ],
    "Travel_Expense_Policy.pdf": [
        "Travel & Expenses\n\nUber/Lyft allowed for business travel.\nMeals reimbursed up to $60/day.\nReceipts required for expenses over $25.",
        "Approval & Submission\n\nPre-approval required for airfare.\nSubmit expenses within 14 days of travel end."
    ]
}

# Generate PDFs using ReportLab
for filename, pages in docs.items():
    path = os.path.join(DATA_DIR, filename)
    c = canvas.Canvas(path, pagesize=LETTER)
    width, height = LETTER

    # Create each page of the PDF
    for text in pages:
        # Add title in bold at top
        c.setFont("Helvetica-Bold", 14)
        c.drawString(72, height-72, filename.replace(".pdf", ""))

        # Add content in normal font
        c.setFont("Helvetica", 11)
        y = height-110
        for line in text.split("\n"):
            c.drawString(72, y, line)
            y -= 16  # Line spacing
        c.showPage()  # New page for next content
    c.save()  # Save the PDF

print(f"Generated {len(docs)} PDFs in {DATA_DIR}")
