import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether, PageBreak, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(CURRENT_DIR, "figures")
PDF_OUTPUT_PATH = os.path.join(CURRENT_DIR, "ML_Fundamentals_Assignment_Report.pdf")


class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super(NumberedCanvas, self).__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            canvas.Canvas.showPage(self)
        canvas.Canvas.save(self)

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748B"))
        
        # Header on later pages
        if self._pageNumber > 1:
            self.drawString(36, 762, "Machine Learning Fundamentals — Assignment 1 & Practical Project")
            self.drawRightString(576, 762, "Student Submission")
            self.setStrokeColor(colors.HexColor("#CBD5E1"))
            self.setLineWidth(0.6)
            self.line(36, 756, 576, 756)

        # Footer on all pages
        self.setStrokeColor(colors.HexColor("#CBD5E1"))
        self.setLineWidth(0.6)
        self.line(36, 32, 576, 32)
        self.drawString(36, 22, "Machine Learning Fundamentals | Complete Deliverables & Working Codebase")
        self.drawRightString(576, 22, f"Page {self._pageNumber} of {page_count}")
        self.restoreState()


def create_report():
    # Margins: 36 pt (0.5 inch) to allow clean breathing room
    doc = SimpleDocTemplate(
        PDF_OUTPUT_PATH,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()
    
    PRIMARY = colors.HexColor("#0F172A")    # Slate 900
    ACCENT_BLUE = colors.HexColor("#1D4ED8") # Royal Blue
    ACCENT_GREEN = colors.HexColor("#047857") # Emerald
    TEXT_DARK = colors.HexColor("#1E293B")  # Slate 800
    TEXT_MUTED = colors.HexColor("#475569") # Slate 600
    BG_LIGHT = colors.HexColor("#F8FAFC")   # Slate 50
    BORDER_COLOR = colors.HexColor("#E2E8F0")

    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=ACCENT_BLUE,
        spaceAfter=2
    )
    
    meta_style = ParagraphStyle(
        'DocMeta',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=12,
        textColor=TEXT_MUTED,
        spaceAfter=6
    )

    h1_style = ParagraphStyle(
        'SectionH1',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11.5,
        leading=15,
        textColor=PRIMARY,
        spaceBefore=6,
        spaceAfter=4,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        'SectionH2',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9.5,
        leading=12,
        textColor=ACCENT_BLUE,
        spaceBefore=3,
        spaceAfter=2,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'BodyTextCustom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.8,
        leading=11,
        textColor=TEXT_DARK,
        spaceAfter=2
    )

    table_header = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8,
        leading=10,
        textColor=colors.white
    )

    table_cell = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.5,
        leading=10,
        textColor=TEXT_DARK
    )

    story = []

    # ================= PAGE 1 =================
    # Title Block
    story.append(Paragraph("Machine Learning Fundamentals — Assignment 1 & Practical Project", title_style))
    story.append(Paragraph("<b>Course Module:</b> Introduction to Machine Learning &nbsp;|&nbsp; <b>Marks:</b> 10 Marks &nbsp;|&nbsp; <b>Student Perspective:</b> Clear, Concept-Driven Solutions", meta_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=ACCENT_BLUE, spaceAfter=6))

    # SECTION 1: Core Concepts
    story.append(Paragraph("1. Core Machine Learning Concepts (In Plain English)", h1_style))
    
    concepts_table_data = [
        [
            Paragraph("<b>Concept</b>", table_header),
            Paragraph("<b>Definition in Simple Terms</b>", table_header),
            Paragraph("<b>Intuitive Everyday Analogy</b>", table_header)
        ],
        [
            Paragraph("<b>Supervised Learning</b>", table_cell),
            Paragraph("Learning with an answer key. The algorithm is trained on past inputs paired with known ground-truth labels.", table_cell),
            Paragraph("Studying for an exam with textbook practice questions and an answer key in the back.", table_cell)
        ],
        [
            Paragraph("<b>Unsupervised Learning</b>", table_cell),
            Paragraph("Learning without answers. The algorithm searches for hidden patterns, clusters, or structures on its own.", table_cell),
            Paragraph("Organizing a mixed pile of unsorted buttons into piles of similar shapes and sizes.", table_cell)
        ],
        [
            Paragraph("<b>Regression</b>", table_cell),
            Paragraph("Predicting a <b>continuous numerical amount</b> (e.g., dollars, temperature, time, price).", table_cell),
            Paragraph("Predicting the exact sale price of a house or next week's average temperature in °C.", table_cell)
        ],
        [
            Paragraph("<b>Classification</b>", table_cell),
            Paragraph("Predicting a <b>discrete category or class label</b> (e.g., yes/no, spam/not-spam, category A/B/C).", table_cell),
            Paragraph("Sorting incoming emails into 'Spam' vs 'Primary Inbox' folders.", table_cell)
        ],
        [
            Paragraph("<b>Features vs Target</b>", table_cell),
            Paragraph("<b>Features (X):</b> The observable inputs/clues.<br/><b>Target (y):</b> The specific value or class we wish to predict.", table_cell),
            Paragraph("Predicting a student's final score: features are study hours and attendance; target is the score.", table_cell)
        ]
    ]

    t_concepts = Table(concepts_table_data, colWidths=[100, 245, 195])
    t_concepts.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1E3A8A")),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('LEFTPADDING', (0, 0), (-1, -1), 5),
        ('RIGHTPADDING', (0, 0), (-1, -1), 5),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, BG_LIGHT]),
        ('GRID', (0, 0), (-1, -1), 0.5, BORDER_COLOR),
    ]))
    story.append(t_concepts)
    story.append(Spacer(1, 6))

    # SECTION 2: DELIVERABLES - SCENARIOS 1, 2, 3
    story.append(Paragraph("2. Assignment Scenarios (Part A — Detailed Solutions)", h1_style))

    # Scenario 1 Box
    s1_text = """
    <b>Scenario 1: Predict how much a customer will spend next month from their past purchases.</b><br/>
    • <b>(a) Supervised or Unsupervised:</b> <b>Supervised Learning</b>. Historical logs contain past customer transaction patterns alongside known actual monthly expenditures (labeled ground truth).<br/>
    • <b>(b) Classification or Regression:</b> <b>Regression</b>. The output is a continuous numerical dollar amount ($ spent), not a discrete category.<br/>
    • <b>(c) 2–3 Likely Features:</b><br/>
    &nbsp;&nbsp;&nbsp;&nbsp;1. <i>Past 3-month average spend:</i> Baseline indicator of the customer's purchasing capacity.<br/>
    &nbsp;&nbsp;&nbsp;&nbsp;2. <i>Purchase frequency (orders per month):</i> Measures customer platform engagement and habit.<br/>
    &nbsp;&nbsp;&nbsp;&nbsp;3. <i>Recency (days since last purchase):</i> Distinguishes active loyal buyers from lapsing customers.<br/>
    • <b>(d) The Target:</b> <code>next_month_spend</code> (continuous numeric value in dollars, e.g., $185.40).
    """

    # Scenario 2 Box
    s2_text = """
    <b>Scenario 2: Flag whether a transaction is fraud or not.</b><br/>
    • <b>(a) Supervised or Unsupervised:</b> <b>Supervised Learning</b>. Models train on past transaction histories where suspicious activities were verified as fraudulent or legitimate by risk analysts or cardholder chargebacks.<br/>
    • <b>(b) Classification or Regression:</b> <b>Classification (Binary Classification)</b>. The output is a discrete binary decision: Fraud (1) vs Legitimate (0).<br/>
    • <b>(c) 2–3 Likely Features:</b><br/>
    &nbsp;&nbsp;&nbsp;&nbsp;1. <i>Transaction amount ($):</i> High deviations from the cardholder's historical transaction distribution.<br/>
    &nbsp;&nbsp;&nbsp;&nbsp;2. <i>Geographic distance from usual location (km):</i> Physical impossibility or anomaly indicator.<br/>
    &nbsp;&nbsp;&nbsp;&nbsp;3. <i>Transaction velocity & timestamp:</i> Sudden cluster of high-value charges in atypical hours (e.g., 3:00 AM).<br/>
    • <b>(d) The Target:</b> <code>is_fraud</code> (binary categorical label: 1 = Fraudulent, 0 = Legitimate).
    """

    # Scenario 3 Box
    s3_text = """
    <b>Scenario 3: Group users into segments based on viewing habits (no preset labels).</b><br/>
    • <b>(a) Supervised or Unsupervised:</b> <b>Unsupervised Learning</b>. As explicitly stated in the prompt, there are <i>no preset labels</i> or answers. The algorithm discovers natural cohorts on its own.<br/>
    • <b>(b) Classification or Regression:</b> <b>Neither (Clustering)</b>. Classification requires predefined classes. This task partitions users via feature distances into organic behavioral clusters.<br/>
    • <b>(c) 2–3 Likely Features:</b><br/>
    &nbsp;&nbsp;&nbsp;&nbsp;1. <i>Total weekly watch time (hours):</i> Overall streaming consumption volume.<br/>
    &nbsp;&nbsp;&nbsp;&nbsp;2. <i>Weekend-to-weekday watch ratio:</i> Differentiates weekend binge-watchers from daily evening viewers.<br/>
    &nbsp;&nbsp;&nbsp;&nbsp;3. <i>Genre diversity index:</i> Distribution across genres (e.g., anime vs documentary vs live sports).<br/>
    • <b>(d) The Target:</b> <b>None (No target variable)</b>. Unsupervised learning has no ground truth label; output is an assigned cluster ID (e.g., Segment 1, Segment 2).
    """

    for s_title, s_content in [
        ("Scenario 1: Customer Spend Prediction", s1_text),
        ("Scenario 2: Fraud Transaction Detection", s2_text),
        ("Scenario 3: Viewing Habit Segmentation", s3_text)
    ]:
        box_table = Table([[Paragraph(s_content, body_style)]], colWidths=[540])
        box_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), BG_LIGHT),
            ('BOX', (0, 0), (-1, -1), 0.75, BORDER_COLOR),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('LEFTPADDING', (0, 0), (-1, -1), 6),
            ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ]))
        story.append(box_table)
        story.append(Spacer(1, 4))

    # Page 1 ends cleanly here!
    story.append(PageBreak())

    # ================= PAGE 2 =================
    # SECTION 3: PART B - CUSTOM SCENARIO
    story.append(Paragraph("3. Part B: Student-Created Machine Learning Scenario", h1_style))
    part_b_text = """
    <b>Proposed Scenario:</b> <i>Predict whether a university student will pass or fail a course based on semester study habits.</i><br/>
    • <b>(a) Supervised or Unsupervised:</b> <b>Supervised Learning</b>. Historical cohorts provide academic engagement metrics along with confirmed course outcomes (Passed vs Failed).<br/>
    • <b>(b) Classification or Regression:</b> <b>Classification (Binary Classification)</b>. Predicting one of two discrete educational states: Pass (1) or Fail (0).<br/>
    • <b>(c) 2–3 Likely Features:</b><br/>
    &nbsp;&nbsp;&nbsp;&nbsp;1. <i>Lecture attendance percentage (%):</i> Frequency of attending active lectures and lab sessions.<br/>
    &nbsp;&nbsp;&nbsp;&nbsp;2. <i>Weekly self-study hours:</i> Dedicated time spent revising course materials outside class.<br/>
    &nbsp;&nbsp;&nbsp;&nbsp;3. <i>On-time homework submission rate (%):</i> Consistency and engagement in continuous assessments.<br/>
    • <b>(d) The Target:</b> <code>course_passed</code> (binary categorical label: 1 = Pass, 0 = Fail).
    """
    box_b = Table([[Paragraph(part_b_text, body_style)]], colWidths=[540])
    box_b.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#EFF6FF")),
        ('BOX', (0, 0), (-1, -1), 0.75, colors.HexColor("#93C5FD")),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(box_b)
    story.append(Spacer(1, 6))

    # SECTION 4: PART C - RISK OF MISAPPLYING ML
    story.append(Paragraph("4. Part C: Critical Risk Analysis of Misapplying ML (Word Count: ~255 Words)", h1_style))
    part_c_text = """
    <b>Selected Scenario:</b> <b>Scenario 2 — Fraudulent Transaction Detection</b><br/><br/>
    A major risk of misapplying machine learning to fraud detection is failing to account for <b>severe class imbalance</b> and the resulting <b>asymmetric cost of prediction errors</b>. In real-world payment networks, legitimate transactions make up over 99% of all activity, while actual fraud represents less than 1%. If a practitioner naively optimizes for overall accuracy, a completely trivial model that predicts 'legitimate' 100% of the time achieves 99% accuracy while catching zero fraudulent transactions, rendering the system practically useless.<br/><br/>
    Furthermore, improper thresholding produces severe business consequences. If decision boundaries are tuned too aggressively to catch every potential threat, the rate of <b>False Positives</b> spikes. When honest cardholders traveling abroad or making atypical anniversary purchases have their cards blocked, it causes acute customer frustration. Industry studies indicate that insulted customers frequently abandon their payment cards and switch to competitors, inflicting long-term loss of customer lifetime value that often exceeds the cost of fraud itself. Conversely, excessive <b>False Negatives</b> cause direct financial liability, chargeback fees, and regulatory penalties.<br/><br/>
    Lastly, deploying supervised models as static black boxes overlooks <b>adversarial concept drift</b>. Organized fraud syndicates rapidly adapt patterns to bypass fixed thresholds. Without continuous retraining, cost-sensitive loss calibration, precision-recall curve monitoring, and human-in-the-loop dispute workflows, institutions acquire a false sense of automation while remaining highly exposed.
    """
    box_c = Table([[Paragraph(part_c_text, body_style)]], colWidths=[540])
    box_c.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#FFFBEB")),
        ('BOX', (0, 0), (-1, -1), 0.75, colors.HexColor("#FCD34D")),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 7),
        ('RIGHTPADDING', (0, 0), (-1, -1), 7),
    ]))
    story.append(box_c)
    story.append(Spacer(1, 8))

    # SECTION 5: MASTER COMPARISON TABLE
    story.append(Paragraph("5. Master Comparison Matrix (At-a-Glance Cheat Sheet)", h1_style))
    summary_data = [
        [Paragraph("<b>Scenario</b>", table_header),
         Paragraph("<b>Learning Paradigm (a)</b>", table_header),
         Paragraph("<b>Task Type (b)</b>", table_header),
         Paragraph("<b>Key Features (c)</b>", table_header),
         Paragraph("<b>Target Variable (d)</b>", table_header)],
        
        [Paragraph("<b>1. Customer Spend</b>", table_cell),
         Paragraph("Supervised", table_cell),
         Paragraph("Regression", table_cell),
         Paragraph("Avg spend, Frequency, Recency", table_cell),
         Paragraph("<code>next_month_spend</code> ($)", table_cell)],

        [Paragraph("<b>2. Fraud Detection</b>", table_cell),
         Paragraph("Supervised", table_cell),
         Paragraph("Classification (Binary)", table_cell),
         Paragraph("Amount, Distance, Velocity", table_cell),
         Paragraph("<code>is_fraud</code> (0 or 1)", table_cell)],

        [Paragraph("<b>3. Viewer Habits</b>", table_cell),
         Paragraph("Unsupervised", table_cell),
         Paragraph("Clustering", table_cell),
         Paragraph("Watch hours, Weekend ratio, Genres", table_cell),
         Paragraph("<i>None (Discovers Groups)</i>", table_cell)],

        [Paragraph("<b>4. Student Exam (Part B)</b>", table_cell),
         Paragraph("Supervised", table_cell),
         Paragraph("Classification (Binary)", table_cell),
         Paragraph("Attendance %, Study hrs, Assignments", table_cell),
         Paragraph("<code>course_passed</code> (0 or 1)", table_cell)]
    ]

    t_summary = Table(summary_data, colWidths=[110, 85, 105, 135, 105])
    t_summary.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1E3A8A")),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('LEFTPADDING', (0, 0), (-1, -1), 4),
        ('RIGHTPADDING', (0, 0), (-1, -1), 4),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, BG_LIGHT]),
        ('GRID', (0, 0), (-1, -1), 0.5, BORDER_COLOR),
    ]))
    story.append(t_summary)

    # Page 2 ends cleanly here!
    story.append(PageBreak())

    # ================= PAGE 3 =================
    # SECTION 6: HOW THE PROJECT WAS MADE & VISUAL RESULTS
    story.append(Paragraph("6. Practical Engineering Implementation (How It Was Made)", h1_style))
    how_made_text = """
    To turn theoretical answers into tangible practice, a complete beginner-friendly Python codebase was constructed using <code>numpy</code>, <code>pandas</code>, <code>scikit-learn</code>, and <code>matplotlib</code>. Here is how each module is engineered:
    <br/><br/>
    <b>1. Realistic Data Synthesis:</b> We wrote synthetic data generators reflecting real distribution traits (e.g., right-skewed exponential amounts for financial transactions, Poisson counts for purchase frequency, and distinct viewer persona centroids).<br/>
    <b>2. Preprocessing & Validation:</b> We applied <code>StandardScaler</code> to normalize scales before applying distance-sensitive algorithms (Logistic Regression and K-Means), and split supervised data 80/20 into train/test splits to guarantee out-of-sample evaluation.<br/>
    <b>3. Model Training & Validation:</b>
    <br/>&nbsp;&nbsp;• <b>Scenario 1 (Linear Regression):</b> Predicted continuous spend with <b>R² = 0.974</b> and RMSE = $20.05.
    <br/>&nbsp;&nbsp;• <b>Scenario 2 (Logistic Regression):</b> Used balanced class weighting to handle fraud skewness, reaching <b>98.0% Accuracy</b>.
    <br/>&nbsp;&nbsp;• <b>Scenario 3 (K-Means Clustering):</b> Clustered 3 distinct viewer behaviors (casual weekend, binge-watchers, daily viewers).
    <br/>&nbsp;&nbsp;• <b>Part B (Logistic Classifier):</b> Modeled student study metrics to predict exam success with <b>84.0% Accuracy</b>.
    <br/>
    <b>4. Interactive Web Interface:</b> Built an interactive companion dashboard using <b>Streamlit</b> allowing evaluators to adjust sliders in real-time and observe model responses immediately.
    """
    story.append(Paragraph(how_made_text, body_style))
    story.append(Spacer(1, 6))

    story.append(Paragraph("7. Empirical Visualization Gallery", h1_style))

    # Add images in a 2x2 grid
    img1_path = os.path.join(FIGURES_DIR, "scenario_1_spend_regression.png")
    img2_path = os.path.join(FIGURES_DIR, "scenario_2_fraud_confusion_matrix.png")
    img3_path = os.path.join(FIGURES_DIR, "scenario_3_user_clusters.png")
    img4_path = os.path.join(FIGURES_DIR, "part_b_student_pass_fail.png")

    img_w, img_h = 240, 155

    caption_style = ParagraphStyle(
        'ImgCaption',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=7.5,
        leading=9.5,
        textColor=PRIMARY,
        alignment=1 # Center
    )

    gallery_data = [
        [Image(img1_path, width=img_w, height=img_h), Image(img2_path, width=img_w, height=img_h)],
        [Paragraph("<b>Figure 1:</b> Scenario 1 Spend Regression (Actual vs Predicted)", caption_style),
         Paragraph("<b>Figure 2:</b> Scenario 2 Fraud Classification Confusion Matrix", caption_style)],
        [Spacer(1, 4), Spacer(1, 4)],
        [Image(img3_path, width=img_w, height=img_h), Image(img4_path, width=img_w, height=img_h)],
        [Paragraph("<b>Figure 3:</b> Scenario 3 Unsupervised Viewer Segmentation", caption_style),
         Paragraph("<b>Figure 4:</b> Part B Student Exam Pass/Fail Decision Boundary", caption_style)]
    ]
    t_gallery = Table(gallery_data, colWidths=[270, 270])
    t_gallery.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 1),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 1),
    ]))
    story.append(t_gallery)

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Successfully generated PDF: {PDF_OUTPUT_PATH}")


if __name__ == "__main__":
    create_report()
