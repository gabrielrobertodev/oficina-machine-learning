import tkinter as tk
from tkinter import simpledialog, messagebox
import time
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

class PaintGame:
    def __init__(self, root):
        self.root = root
        self.root.title("ML Paint")
        self.root.configure(bg="black")

        self.canvas = tk.Canvas(root, bg="black", width=400, height=300,
                                highlightbackground="cyan", highlightthickness=2)
        self.canvas.pack(pady=10)
        self.text_id = self.canvas.create_text(200, 150, text="ÁREA DE DESENHO",
                                               fill="white", font=("Arial", 14))

        btn_frame = tk.Frame(root, bg="black")
        btn_frame.pack(pady=10)

        self.train_btn = tk.Button(btn_frame, text="Ensinar IA", bg="cyan", command=self.train_mode)
        self.train_btn.grid(row=0, column=0, padx=10)

        self.send_train_btn = tk.Button(btn_frame, text="Enviar teste", bg="cyan", command=self.send_train)
        self.send_train_btn.grid(row=0, column=1, padx=10)

        self.test_btn = tk.Button(btn_frame, text="Testar", bg="cyan", command=self.test_mode)
        self.test_btn.grid(row=0, column=2, padx=10)

        self.send_draw_btn = tk.Button(btn_frame, text="Enviar desenho", bg="cyan", command=self.send_draw)
        self.send_draw_btn.grid(row=0, column=3, padx=10)

        self.clear_btn = tk.Button(root, text="Limpar", bg="orange", command=self.clear_canvas)
        self.clear_btn.pack(pady=5)

        # Estado
        self.current_label = None
        self.drawing_points = []
        self.test_points = []
        self.mode = None
        self.showing_message = False
        self.first_draw = True

        # Dados para ML
        self.X = []  # características
        self.y = []  # rótulos
        self.model = None

        # Controle de limite
        self.example_count = {}  # quantos exemplos por forma
        self.max_examples = 5    # limite máximo

        # Label para mostrar contador
        self.counter_label = tk.Label(root, text="Nenhuma forma ensinada ainda.",
                                      bg="black", fg="white", font=("Arial", 12))
        self.counter_label.pack(pady=5)

        self.canvas.bind("<B1-Motion>", self.paint)

    def extract_features(self, points):
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        width = max(xs) - min(xs)
        height = max(ys) - min(ys)
        num_points = len(points)
        ratio = width / height if height != 0 else 0
        return [width, height, ratio, num_points]

    def train_mode(self):
        self.mode = "train"
        label = simpledialog.askstring("Ensinar IA", "Qual forma você vai desenhar?")
        if label:
            self.current_label = label
            self.drawing_points.clear()
            print(f"Modo Treinar: {label}")

    def send_train(self):
        if self.current_label and self.drawing_points:
            count = self.example_count.get(self.current_label, 0)
            if count >= self.max_examples:
                messagebox.showinfo("Limite",
                                    f"A forma '{self.current_label}' já tem {self.max_examples} exemplos.")
                self.drawing_points.clear()
                return

            features = self.extract_features(self.drawing_points)
            self.X.append(features)
            self.y.append(self.current_label)

            # Atualiza contador
            self.example_count[self.current_label] = count + 1

            # Treina modelo
            self.model = KNeighborsClassifier(n_neighbors=3)
            self.model.fit(self.X, self.y)

            messagebox.showinfo("Treino",
                                f"Exemplo de '{self.current_label}' adicionado! "
                                f"({self.example_count[self.current_label]}/{self.max_examples})")
            self.drawing_points.clear()

            # Atualiza contador na tela
            self.update_counter_label()

    def update_counter_label(self):
        texto = "Exemplos por forma:\n"
        for forma, qtd in self.example_count.items():
            texto += f"- {forma}: {qtd}/{self.max_examples}\n"
        self.counter_label.config(text=texto)

    def test_mode(self):
        self.mode = "test"
        self.test_points.clear()
        for i in range(3, 0, -1):
            self.canvas.delete("all")
            self.canvas.create_text(200, 150, text=f"{i}", fill="white", font=("Arial", 40))
            self.root.update()
            time.sleep(1)
        self.canvas.delete("all")
        self.text_id = self.canvas.create_text(200, 150, text="Pode desenhar!", fill="white", font=("Arial", 20))
        self.showing_message = True
        self.root.update()

    def send_draw(self):
        if self.test_points:
            if not self.model:
                messagebox.showinfo("Resultado", "Que isso que você fez mds?")
                return
            features = self.extract_features(self.test_points)
            prediction = self.model.predict([features])[0]
            messagebox.showinfo("Resultado", f"A IA acha que é: {prediction}")
        else:
            messagebox.showinfo("Resultado", "Nenhum desenho foi feito.")

    def clear_canvas(self):
        self.canvas.delete("all")
        self.text_id = self.canvas.create_text(200, 150, text="ÁREA DE DESENHO",
                                               fill="white", font=("Arial", 14))
        self.first_draw = True
        self.drawing_points.clear()
        self.test_points.clear()
        self.showing_message = False

    def paint(self, event):
        x, y = event.x, event.y
        self.canvas.create_oval(x-2, y-2, x+2, y+2, fill="white", outline="white")
        if self.first_draw and self.text_id:
            self.canvas.delete(self.text_id)
            self.text_id = None
            self.first_draw = False
        if self.showing_message and self.text_id:
            self.canvas.delete(self.text_id)
            self.text_id = None
            self.showing_message = False
        if self.mode == "train":
            self.drawing_points.append((x, y))
        elif self.mode == "test":
            self.test_points.append((x, y))

# Executar
root = tk.Tk()
game = PaintGame(root)
root.mainloop()
