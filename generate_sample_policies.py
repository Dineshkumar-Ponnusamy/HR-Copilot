import os
from reportlab.lib.pagesizes import LETTER
from reportlab.pdfgen import canvas

DATA_DIR = "data_policies"
os.makedirs(DATA_DIR, exist_ok=True)

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

for filename, pages in docs.items():
    path = os.path.join(DATA_DIR, filename)
    c = canvas.Canvas(path, pagesize=LETTER)
    width, height = LETTER
    for text in pages:
        c.setFont("Helvetica-Bold", 14)
        c.drawString(72, height-72, filename.replace(".pdf",""))
        c.setFont("Helvetica", 11)
        y = height-110
        for line in text.split("\n"):
            c.drawString(72, y, line)
            y -= 16
        c.showPage()
    c.save()

print(f"Generated {len(docs)} PDFs in {DATA_DIR}")
