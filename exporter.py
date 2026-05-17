from fpdf import FPDF
import datetime
import os

class ChatExporter(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(30, 30, 60)
        self.cell(0, 10, "NeuralDoc — RAG Chat Export", align="C", new_x="LMARGIN", new_y="NEXT")
        self.set_font("Helvetica", "", 8)
        self.set_text_color(120, 120, 140)
        self.cell(0, 6, f"Exported: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}", align="C", new_x="LMARGIN", new_y="NEXT")
        self.ln(4)
        self.set_draw_color(200, 200, 220)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(160, 160, 180)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")


def export_chat_to_pdf(messages: list, filepath: str = "chat_export.pdf"):
    pdf = ChatExporter()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    for msg in messages:
        role = msg["role"]
        content = msg.get("content", "")
        verdict = msg.get("verdict", "")
        sources = msg.get("sources", [])

        if role == "user":
            # User bubble
            pdf.set_fill_color(235, 235, 255)
            pdf.set_text_color(40, 40, 100)
            pdf.set_font("Helvetica", "B", 9)
            pdf.cell(0, 7, "YOU", new_x="LMARGIN", new_y="NEXT", fill=True)
            pdf.set_font("Helvetica", "", 10)
            pdf.set_text_color(30, 30, 60)
            pdf.multi_cell(0, 6, content, new_x="LMARGIN", new_y="NEXT")
            pdf.ln(3)

        else:
            # AI bubble
            pdf.set_fill_color(230, 245, 255)
            pdf.set_text_color(10, 60, 100)
            pdf.set_font("Helvetica", "B", 9)
            pdf.cell(0, 7, "NEURALDOC AI", new_x="LMARGIN", new_y="NEXT", fill=True)
            pdf.set_font("Helvetica", "", 10)
            pdf.set_text_color(20, 40, 70)
            pdf.multi_cell(0, 6, content, new_x="LMARGIN", new_y="NEXT")

            # Verdict
            if verdict:
                is_grounded = "Not" not in verdict and "Grounded" in verdict
                pdf.set_font("Helvetica", "I", 8)
                pdf.set_text_color(0, 140, 80) if is_grounded else pdf.set_text_color(180, 100, 0)
                pdf.cell(0, 5, f"{'✓' if is_grounded else '⚠'} {verdict}", new_x="LMARGIN", new_y="NEXT")

            # Sources
            if sources:
                pdf.ln(2)
                pdf.set_font("Helvetica", "B", 8)
                pdf.set_text_color(80, 80, 120)
                pdf.cell(0, 5, "Sources:", new_x="LMARGIN", new_y="NEXT")
                for i, src in enumerate(sources):
                    fname = os.path.basename(src.get("source", "Unknown"))
                    page  = src.get("page", "?")
                    score = src.get("score", "?")
                    snippet = src.get("content", "")[:120].replace("\n", " ")
                    pdf.set_font("Helvetica", "", 7.5)
                    pdf.set_text_color(100, 100, 140)
                    pdf.multi_cell(0, 5,
                        f"[{i+1}] {fname} · pg {page} · score {score}\n{snippet}…",
                        new_x="LMARGIN", new_y="NEXT"
                    )

            pdf.ln(4)

        # Divider between messages
        pdf.set_draw_color(220, 220, 235)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(4)

    pdf.output(filepath)
    return filepath
