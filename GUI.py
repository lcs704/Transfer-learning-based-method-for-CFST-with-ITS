import sys
import joblib
from TrHGBT import optProposedAlg
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QGroupBox, QGridLayout, QMessageBox
)
from PyQt5.QtGui import QFont, QPainter, QPen, QBrush, QColor
from PyQt5.QtCore import Qt,QPoint

# ================================================
# Load Model
# ================================================
pkg = joblib.load("model_package.pkl")
model = pkg["model"]
scaler = pkg["scaler"]


# ================================================
# Compact & Perfectly Aligned 3D Column Sketch
# ================================================
class ColumnSketch(QWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumSize(360, 440)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        w = self.width()
        h = self.height()

        steel_color     = QColor("#2d3748")
        concrete_color  = QColor("#e2e8f0")
        accent_color    = QColor("#4fd1c5")
        text_color      = QColor("#1a202c")

        cx = w * 0.5
        cy = h * 0.54
        diameter = min(w * 0.78, h * 0.68)-20
        height_cyl = h * 0.56
        ellipse_h = diameter * 0.18

        p.setPen(QPen(steel_color, 4))
        p.setBrush(steel_color)
        p.drawEllipse(int(cx - diameter/2), int(cy - height_cyl/2 - ellipse_h/2), int(diameter), int(ellipse_h))
        p.drawRect(int(cx - diameter/2), int(cy - height_cyl/2), int(diameter), int(height_cyl))
        p.drawEllipse(int(cx - diameter/2), int(cy + height_cyl/2 - ellipse_h/2), int(diameter), int(ellipse_h))

        inner_d = diameter * 0.83
        p.setBrush(concrete_color)
        p.setPen(QPen(accent_color, 3))
        p.drawRect(int(cx - inner_d/2), int(cy - height_cyl/2), int(inner_d), int(height_cyl))
        p.drawEllipse(int(cx - inner_d/2), int(cy - height_cyl/2 - ellipse_h/2), int(inner_d), int(ellipse_h))
        p.drawEllipse(int(cx - inner_d/2), int(cy + height_cyl/2 - ellipse_h/2), int(inner_d), int(ellipse_h))

        p.setPen(QPen(text_color, 3))
        p.setFont(QFont("Segoe UI", 12, QFont.Bold))

        # D
        y_d = cy + height_cyl/2 + 40
        p.drawLine(int(cx - diameter/2), int(y_d), int(cx + diameter/2), int(y_d))
        p.drawLine(int(cx - diameter/2), int(y_d - 8), int(cx - diameter/2), int(y_d + 8))
        p.drawLine(int(cx + diameter/2), int(y_d - 8), int(cx + diameter/2), int(y_d + 8))
        p.drawText(int(cx - 18), int(y_d + 22), "D")

        # t
        p.setPen(QPen(accent_color, 4))
        p.drawLine(int(cx - diameter/2), int(cy), int(cx - inner_d/2), int(cy))
        p.drawText(int(cx - diameter/2 + 8), int(cy - 12), "t")

        # L
        p.setPen(QPen(text_color, 4))
        p.drawLine(int(cx + diameter/2 + 25), int(cy - height_cyl/2), int(cx + diameter/2 + 25), int(cy + height_cyl/2))
        p.drawLine(int(cx + diameter/2 + 20), int(cy - height_cyl/2), int(cx + diameter/2 + 30), int(cy - height_cyl/2))
        p.drawLine(int(cx + diameter/2 +20), int(cy + height_cyl/2), int(cx + diameter/2 + 30), int(cy + height_cyl/2))
        p.drawText(int(cx + diameter/2 + 32), int(cy), "L")

        p.setPen(QPen(QColor("#1a202c"), 2))
        p.setFont(QFont("Segoe UI", 12, QFont.Bold))

        text = "Cross-section type:\n 1 = Circular, 0 = Square"

        text_y = cy - height_cyl
        p.drawText(
            int(cx - 120),
            int(text_y),
            250,
            100,
            Qt.AlignCenter,
            text
        )
        p.setFont(QFont("Segoe UI", 9))
        text = "(The schematic below takes type=1 as an example)"

        text_y = cy - height_cyl + 80
        p.drawText(
            0,
            int(text_y),
            w,
            50,
            Qt.AlignCenter,
            text
        )

        # ============================================
        # Eccentric Load
        # ============================================
        p.setPen(QPen(QColor("#e53e3e"), 4))
        p.setBrush(QBrush(QColor("#e53e3e")))

        e_offset = diameter * 0.18

        arrow_top_y = cy - height_cyl / 2 - 60
        arrow_bottom_y = cy - height_cyl / 2 - 5

        p.drawLine(
            int(cx + e_offset),
            int(arrow_top_y),
            int(cx + e_offset),
            int(arrow_bottom_y)
        )

        arrow_head = [
            QPoint(int(cx + e_offset - 10), int(arrow_bottom_y - 10)),
            QPoint(int(cx + e_offset + 10), int(arrow_bottom_y - 10)),
            QPoint(int(cx + e_offset), int(arrow_bottom_y))
        ]
        p.drawPolygon(*arrow_head)

        # e
        p.setFont(QFont("Segoe UI", 12, QFont.Bold))
        p.drawText(
            int(cx + e_offset + 12),
            int(arrow_top_y + 5),
            "Nᵤ"
        )

        p.setPen(QPen(QColor("#e53e3e"), 3))

        top_center_x = cx
        top_center_y = cy - height_cyl / 2

        arrow_x = cx + e_offset
        e_line_y = top_center_y

        p.drawLine(
            int(top_center_x),
            int(e_line_y),
            int(arrow_x),
            int(e_line_y)
        )

        p.drawLine(
            int(top_center_x),
            int(e_line_y - 6),
            int(top_center_x),
            int(e_line_y + 6)
        )

        p.drawLine(
            int(arrow_x),
            int(e_line_y - 6),
            int(arrow_x),
            int(e_line_y + 6)
        )

        p.setFont(QFont("Segoe UI", 12, QFont.Bold))
        p.drawText(
            int((top_center_x + arrow_x) / 2 - 8),
            int(e_line_y - 8),
            "e"
        )

        p.setPen(QPen(text_color, 4))
        legend_y = int(y_d + 55)
        p.setBrush(steel_color)
        p.drawRect(int(cx - 70), legend_y, 22, 16)
        p.setBrush(concrete_color)
        p.setPen(accent_color)
        p.drawRect(int(cx + 5), legend_y, 22, 16)
        p.setPen(text_color)
        p.setFont(QFont("Segoe UI", 12, QFont.Bold))
        p.drawText(int(cx - 45), legend_y + 12, "Steel")
        p.drawText(int(cx + 32), legend_y + 12, "ITS Concrete")

class MainUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ITS CFST Column Compressive Capacity Predictor")
        self.resize(880, 660)

        self.setStyleSheet("""
            QWidget {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #f0f4f8, stop:1 #e2e8f0);
                font-family: "Segoe UI", "Microsoft YaHei";
            }
            QGroupBox {
                background: white;
                border-radius: 14px;
                border: none;
                padding-top: 16px;
                margin-top: 18px;
                font-size: 16px;
                font-weight: 600;
                color: #2d3748;
                box-shadow: 0 8px 20px rgba(0,0,0,0.08);
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center;
                left: 0px;
                padding: 6px 20px;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #667eea, stop:1 #764ba2);
                color: white;
                border-radius: 10px;
            }
            QLineEdit {
                height: 40px;
                padding: 0 14px;
                border: 2px solid #e2e8f0;
                border-radius: 10px;
                background: white;
                font-size: 16px;
            }
            QLineEdit:focus {
                border-color: #667eea;
            }
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #667eea, stop:1 #764ba2);
                color: white;
                height: 52px;
                border-radius: 14px;
                font-size: 17px;
                font-weight: 600;
                border: none;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #764ba2, stop:1 #764ba2);
                box-shadow: 0 8px 16px rgba(102,126,234,0.3);
            }
            #resultLabel {
                font-size: 26px;
                font-weight: 700;
                color: #2d3748;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #ffecd2, stop:1 #fcb69f);
                padding: 18px;
                border-radius: 16px;
                min-height: 60px;
            }
        """)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(16)

        # 标题
        title = QLabel("Compressive Capacity Prediction of\nCFST Columns with ITS")
        title.setAlignment(Qt.AlignCenter)
        title.setFont(QFont("Segoe UI", 22, QFont.Bold))
        title.setStyleSheet("color: #2d3748; padding: 16px; background: rgba(255,255,255,0.8); border-radius: 16px;")
        main_layout.addWidget(title)

        content = QHBoxLayout()
        content.setSpacing(18)
        main_layout.addLayout(content)

        left = QVBoxLayout()
        left.setSpacing(18)
        left.addWidget(self.create_input_card())
        left.addWidget(self.create_result_card())
        content.addLayout(left)

        content.addWidget(self.create_sketch_card())

    def create_input_card(self):
        g = QGroupBox("Input Parameters")
        grid = QGridLayout()
        grid.setSpacing(14)
        grid.setContentsMargins(24, 36, 24, 24)

        labels = [
            "Steel tube outer diameter D (mm)",
            "Steel tube thickness t (mm)",
            "Specimen length L (mm)",
            "ITS replacement radio r (%)",
            "Cube compressive strength f<sub>cu</sub> (MPa)",
            "Steel yield strength f<sub>y</sub> (MPa)",
            "eccentricity e (mm)",
            "Cross-section type (1/0)"
        ]

        self.edits = []
        for i, text in enumerate(labels):
            lab = QLabel(text)
            lab.setFont(QFont("Segoe UI", 13))
            edit = QLineEdit()
            edit.setPlaceholderText("Enter value...")
            grid.addWidget(lab, i, 0)
            grid.addWidget(edit, i, 1)
            self.edits.append(edit)

        g.setLayout(grid)
        return g

    def create_sketch_card(self):
        g = QGroupBox("Column Schematic")
        v = QVBoxLayout()
        # 已修复：完整正确的 margins，顶部留足空间给标题，其余收紧
        v.setContentsMargins(15, 38, 15, 15)   # left, top, right, bottom
        self.sketch = ColumnSketch()
        v.addWidget(self.sketch)
        g.setLayout(v)
        return g

    def create_result_card(self):
        g = QGroupBox("Prediction Result")
        v = QVBoxLayout()
        v.setSpacing(20)
        v.setContentsMargins(24, 36, 24, 24)

        self.result_label = QLabel("Compressive capacity Nᵤ = 0 kN")
        self.result_label.setObjectName("resultLabel")
        self.result_label.setAlignment(Qt.AlignCenter)

        btn = QPushButton("🚀 Start Prediction")
        btn.clicked.connect(self.predict)

        v.addWidget(self.result_label)
        v.addStretch()
        v.addWidget(btn, alignment=Qt.AlignCenter)

        g.setLayout(v)
        return g

    def predict(self):
        try:
            vals = [float(e.text()) for e in self.edits if e.text().strip()]
            if len(vals) != 8:
                raise ValueError
        except:
            QMessageBox.warning(self, "Input Error", "Please enter valid numeric values for all fields!")
            return

        x = scaler.transform([vals])
        pred = model.predict(x)[0]
        self.result_label.setText(f"Compressive capacity Nᵤ = <span style='color:#e53e3e;font-size:30px'>{pred:.1f}</span> kN")


app = QApplication(sys.argv)
window = MainUI()
window.show()
sys.exit(app.exec_())