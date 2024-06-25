import tkinter as tk
from tkinter import messagebox
import model_prediction as mp
from lime.lime_text import LimeTextExplainer
import matplotlib.pyplot as plt
from io import BytesIO
from bs4 import BeautifulSoup
from PIL import Image, ImageTk

class HelloApplication(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        self.grid(row=0, column=0, sticky="nsew")
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        # Centering Frame
        frame = tk.Frame(self)
        frame.grid(row=0, column=0)
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_rowconfigure([0, 1, 2, 3, 4, 5], weight=1)

        # Email Input Label
        labelemail = tk.Label(frame, text="Email Input:", font=("Arial", 24))
        labelemail.grid(row=0, column=0, pady=10, padx=20, sticky="ew")
        
        # Email Text Area
        self.textemail = tk.Text(frame, height=15, font=("Arial", 16), wrap="word")
        self.textemail.grid(row=1, column=0, pady=10, padx=20, sticky="ew")
        self.textemail.insert(tk.END, "Write your email here..")

        # Bind the click event to the clear_text method
        self.textemail.bind("<FocusIn>", self.clear_text)
        
        # Check Input Email Button
        checkbuton = tk.Button(frame, text="Check Input Email", bg="#6b6b6b", fg="white", font=("Arial", 16), width=20, height=2, command=self.on_check_input_email)
        checkbuton.grid(row=2, column=0, pady=10, padx=20, sticky="ew")
        
        # Separator
        blackspace = tk.Frame(frame, height=2, bd=1, relief=tk.SUNKEN)
        blackspace.grid(row=3, column=0, sticky="ew", padx=5, pady=20)

        # Explanation Frame
        self.explanation_frame = tk.Frame(self)
        self.explanation_frame.grid(row=4, column=0, pady=10, padx=20, sticky="ew")

        # Highlighted Explanation Text
        self.highlight_text = tk.Text(self, height=15, font=("Arial", 12), wrap="word")
        self.highlight_text.grid(row=5, column=0, pady=10, padx=20, sticky="ew")
        self.highlight_text.tag_configure('highlight', background='yellow')

    def clear_text(self, event):
        if self.textemail.get("1.0", tk.END).strip() == "Write your email here..":
            self.textemail.delete("1.0", tk.END)
        
    def on_check_input_email(self):
        contentofemail = self.textemail.get("1.0", tk.END).strip()
        method = self.controller.method
        classf = self.controller.classifier

        # Select the appropriate model based on user input
        if method == "Bag of Words":
            if classf == "Logistic Regression":
                predic = mp.predict_lrbow(contentofemail)
            elif classf == "Naive Bayes":
                predic = mp.predict_cnbbow(contentofemail)
        elif method == "TF-IDF":
            if classf == "Logistic Regression":
                predic = mp.predict_lrtfidf(contentofemail)
            elif classf == "Naive Bayes":
                predic = mp.predict_cnbtfidf(contentofemail)
            elif classf == "Support Vector Machine":
                predic = mp.predict_svm(contentofemail)
                proba = mp.predict_proba([contentofemail])
                pred_class = "Spam" if proba[0][1] > 0.5 else "Not Spam"
                self.show_lime_explanation(contentofemail, pred_class)
        
        result = "Spam" if predic == [1] else "Not Spam"
        
        messagebox.showinfo("Prediction Result", result)


    def show_lime_explanation(self, text, pred_class):
        # Generate LIME explanation
        explainer = LimeTextExplainer(class_names=['Not Spam', 'Spam'])
        exp = explainer.explain_instance(text, mp.predict_proba, num_features=6, labels=[1 if pred_class == "Spam" else 0])
        
        # Plot explanation
        fig = exp.as_pyplot_figure(label=1 if pred_class == "Spam" else 0)
        buf = BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        img = Image.open(buf)
        img = ImageTk.PhotoImage(img)
        
        # Clear previous explanation
        for widget in self.explanation_frame.winfo_children():
            widget.destroy()
        
        # Display explanation plot
        label = tk.Label(self.explanation_frame, image=img)
        label.image = img  # Keep a reference to avoid garbage collection
        label.grid(row=0, column=0, padx=10, pady=10)

        # Clear previous tags and content in highlight_text
        self.highlight_text.delete("1.0", tk.END)
        
        # Insert original email text
        self.highlight_text.insert(tk.END, text + "\n\n")

        # Display explanation text and highlight words in email text
        explanation_text = exp.as_list(label=1 if pred_class == "Spam" else 0)
        for word, weight in explanation_text:
            # Highlight words in the email text
            self.highlight_word_in_text(self.highlight_text, word)
            self.highlight_text.insert(tk.END, f"{word}: {weight:.4f}\n", 'highlight')

    def highlight_word_in_text(self, text_widget, word):
        start = '1.0'
        while True:
            start = text_widget.search(word, start, stopindex=tk.END, nocase=True)
            if not start:
                break
            end = f"{start}+{len(word)}c"
            text_widget.tag_add('highlight', start, end)
            start = end