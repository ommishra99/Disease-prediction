from tkinter import *
import numpy as np
import pandas as pd

# ================= LOAD DATA =================
df = pd.read_csv("Training.csv")
tr = pd.read_csv("Testing.csv")

# ---------------- SYMPTOMS (AUTO FROM CSV) ----------------
l1 = df.columns[:-1].tolist()   # all symptoms
OPTIONS = sorted(l1)

# ---------------- ENCODING ----------------
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

X = df[l1]
y = le.fit_transform(df['prognosis'])

X_test = tr[l1]
y_test = le.transform(tr['prognosis'])

# ================= MODELS =================
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

dt = DecisionTreeClassifier(random_state=0)
rf = RandomForestClassifier(n_estimators=150, random_state=0)
nb = GaussianNB()

dt.fit(X, y)
rf.fit(X, y)
nb.fit(X, y)

# ================= PREDICTION FUNCTION =================
def predict_disease(model, output_box):
    l2 = [0] * len(l1)

    symptoms = [Symptom1.get(), Symptom2.get(),
                Symptom3.get(), Symptom4.get(), Symptom5.get()]

    if "" in symptoms:
        output_box.delete("1.0", END)
        output_box.insert(END, "Please select all symptoms")
        return

    symptoms = list(set(symptoms))  # remove duplicates

    for i in range(len(l1)):
        if l1[i] in symptoms:
            l2[i] = 1

    l2 = np.array(l2).reshape(1, -1)

    prediction = model.predict(l2)[0]
    disease = le.inverse_transform([prediction])[0]

    output_box.delete("1.0", END)
    output_box.insert(END, disease)

# ================= GUI =================
root = Tk()
root.title("Disease Prediction System")
root.geometry("1000x450")
root.configure(bg="blue")

Label(root, text="Disease Prediction System",
      font=("Elephant", 30),
      fg="white", bg="blue").grid(row=0, column=0, columnspan=5, pady=20)

Symptom1 = StringVar()
Symptom2 = StringVar()
Symptom3 = StringVar()
Symptom4 = StringVar()
Symptom5 = StringVar()

# ---------------- LABELS ----------------
labels = ["Symptom 1", "Symptom 2", "Symptom 3", "Symptom 4", "Symptom 5"]
for i, lbl in enumerate(labels):
    Label(root, text=lbl, fg="yellow", bg="black",
          font=("Arial", 12)).grid(row=i+1, column=0, sticky=W, pady=5)

# ---------------- DROPDOWNS ----------------
OptionMenu(root, Symptom1, *OPTIONS).grid(row=1, column=1)
OptionMenu(root, Symptom2, *OPTIONS).grid(row=2, column=1)
OptionMenu(root, Symptom3, *OPTIONS).grid(row=3, column=1)
OptionMenu(root, Symptom4, *OPTIONS).grid(row=4, column=1)
OptionMenu(root, Symptom5, *OPTIONS).grid(row=5, column=1)

# ---------------- OUTPUT BOXES ----------------
t1 = Text(root, height=1, width=35, bg="orange")
t2 = Text(root, height=1, width=35, bg="orange")
t3 = Text(root, height=1, width=35, bg="orange")

# ---------------- BUTTONS ----------------
Button(root, text="Decision Tree",
       command=lambda: predict_disease(dt, t1),
       bg="green", fg="yellow", width=15).grid(row=1, column=3)
t1.grid(row=1, column=4)

Button(root, text="Random Forest",
       command=lambda: predict_disease(rf, t2),
       bg="green", fg="yellow", width=15).grid(row=2, column=3)
t2.grid(row=2, column=4)

Button(root, text="Naive Bayes",
       command=lambda: predict_disease(nb, t3),
       bg="green", fg="yellow", width=15).grid(row=3, column=3)
t3.grid(row=3, column=4)

root.mainloop()
