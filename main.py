import os
import time
import tkinter as tk
from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
from scipy.fftpack import dctn, idctn
from ttkthemes import ThemedTk

def load_image():
    # Apre una finestra di dialogo per caricare un'immagine
    file_path = filedialog.askopenfilename(filetypes=[('BMP files', '*.bmp')])

    # Verifica se un file è stato selezionato
    if file_path:
        # Carica l'immagine selezionata
        img = Image.open(file_path)

        m = min(img.height, img.width)
        F_slider.configure(from_=1, to=m)
        d_slider.configure(from_=1, to=m - 2)
        F_slider.set(int(m / 2))
        d_slider.set(int(m / 2) - 2)

        # peso dell'immagine caricata
        original_size = os.path.getsize(file_path)
        original_size_label.configure(text=f"Original Size: {format_size(original_size)}")
        # reset dei tempi in caso di caricamento non iniziale
        compressed_time_label.configure(text="Compressed Time: ")
        compressed_size_label.configure(text="Compressed Size:")

        # Mostra l'immagine nella GUI
        img.thumbnail((800, 800))  # Ridimensiona l'immagine per adattarla alla GUI
        img_tk = ImageTk.PhotoImage(img)
        img_label.configure(image=img_tk)
        img_label.image = img_tk


        # abilitazione bottone per la compressione
        compress_button.configure(state=tk.NORMAL)  # Abilita il pulsante di compressione
        compress_button.config(command=lambda: compress_button_clicked(file_path))


def compress_button_clicked(image_path):
    # Ottieni i valori di F e d dai due slider
    F = int(F_slider.get())
    d = int(d_slider.get())
    if d < 0 | d > F - 2:
        d = F - 2

    # chiamata alla funzione di compressione
    tic = time.perf_counter()
    compressed_img = compress_image(image_path, F, d)
    toc = time.perf_counter()
    time_compr = round(float(toc - tic), 4)
    compressed_time_label.configure(text="Compressed Time: " + str(time_compr) + "s")

    # per salvare l'immagine compressa e prenderne la dimensione
    #get name
    filename = os.path.splitext(os.path.basename(image_path))[0]

    # Salva nella cartella "Compressed"
    save_folder = "Compressed"
    if not os.path.exists(save_folder):
        # se non esiste creala
        os.makedirs(save_folder)

    # salva in formato .bmp
    save_path = os.path.join(save_folder, f"{filename}_compressed.bmp")
    compressed_img.save(save_path)
    print(f"Compressed image saved: {save_path}")

    # Peso dell'immagine compressa
    compressed_size = os.path.getsize(save_path)
    compressed_size_label.configure(text=f"Compressed Size: {format_size(compressed_size)}")

    # Mostra l'immagine compressa nella GUI
    compressed_img.thumbnail((800, 800))  # Ridimensiona l'immagine per adattarla alla GUI
    compressed_img_tk = ImageTk.PhotoImage(compressed_img)
    compressed_img_label.configure(image=compressed_img_tk)
    compressed_img_label.image = compressed_img_tk

def compress_image(image_path, F, d):
    # Carica l'immagine
    with open(image_path, 'rb') as f:
        # Carica l'immagine
        img = Image.open(f).convert('L')

    # Ottieni le dimensioni dell'immagine
    width, height = img.size

    max_dim = min(width, height)

    # Verifica dei parametri e update degli slider
    if F > max_dim:
        F = max_dim
        F_slider.set(max_dim)

    if d > F - 2:
        d = F - 2
        d_slider.set(F - 2)

    # Calcola il numero di blocchi in larghezza e altezza
    num_blocks_w = int(width / F)
    num_blocks_h = int(height / F)

    # Calculate the new dimensions for the compressed image
    new_width = num_blocks_w * F
    new_height = num_blocks_h * F

    # Trasforma l'immagine in un array
    img_arr = np.array(img, dtype=np.float32)

    # nuovo array dove inseriremo l'immagine compressa
    # i bit non compressi vengono ricopiati con questa semplice linea di codice
    compressed_arr = img_arr

    # nel caso non volessimo copiare ma croppare
    #compressed_arr = np.zeros((new_height, new_width), dtype=np.float32)

    # Applica la compressione per ogni blocco
    for i in range(num_blocks_h):
        for j in range(num_blocks_w):
            # Estrai il blocco corrente
            block = img_arr[i * F:(i + 1) * F, j * F:(j + 1) * F]

            dct_block = dctn(block, norm='ortho')

            # Elimina le frequenze
            for h in range(F):
                for k in range(F):
                    if h + k >= d:
                        dct_block[h, k] = 0

            # Applica la DCT2 inversa
            idctn_block = idctn(dct_block, norm='ortho')

            # Aggiungi il blocco compresso all'immagine finale
            compressed_arr[i * F : (i + 1) * F, j * F : (j + 1) * F] = idctn_block

            # Arrotonda i valori e li limita nell'intervallo 0-255
    compressed_arr = np.round(compressed_arr).clip(0, 255)

    # Convert the compressed array back to PIL Image
    compressed_img = Image.fromarray(compressed_arr.astype(np.uint8))

    return compressed_img


def format_size(size):
    # Convert the size in bytes to KB or MB
    kb = size / 1024
    if kb < 1024:
        return f"{kb:.2f} KB"
    else:
        mb = kb / 1024
        return f"{mb:.2f} MB"


# Creazione della GUI
root = ThemedTk("arc")
root.title("DCT2 compression")

# Caricamento dell'immagine
load_button = ttk.Button(root, text="Load Image", command=load_image)
load_button.pack()

# Frame per contenere le immagini
image_frame = ttk.Frame(root)
image_frame.pack()

# Display the original image size
original_size_label = ttk.Label(root, text="Original Size: ")
original_size_label.pack(padx=10, pady=(0, 5))

# Display the compressed image size
compressed_size_label = ttk.Label(root, text="Compressed Size: ")
compressed_size_label.pack(padx=10, pady=(0, 5))

# display time of compression
compressed_time_label = ttk.Label(root, text="Compressed Time: ")
compressed_time_label.pack(padx=10, pady=(0, 5))

# Visualizzazione dell'immagine originale
img_label = ttk.Label(image_frame)
img_label.grid(row=0, column=0, padx=10, pady=10)

# Visualizzazione dell'immagine compressa
compressed_img_label = ttk.Label(image_frame)
compressed_img_label.grid(row=0, column=1, padx=10, pady=10)

# Slider per il parametro F
F_slider = tk.Scale(root, from_=1, to=1000, orient=tk.HORIZONTAL, label="F", length=1500)
F_slider.pack()

# Slider per il parametro d
d_slider = tk.Scale(root, from_=0, to=998, orient=tk.HORIZONTAL, label="d", length=1500)
d_slider.pack()

# Bottone per comprimere l'immagine
compress_button = ttk.Button(root, text="Compress", command=compress_button_clicked, state=tk.DISABLED)
compress_button.pack()

root.mainloop()