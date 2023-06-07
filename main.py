import os
import time
import tkinter as tk
from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
from scipy.fft import dctn, idctn
from ttkthemes import ThemedTk
import matplotlib.pyplot as plt

def load_image():
    # Apre una finestra per caricare un'immagine
    file_path = filedialog.askopenfilename(filetypes=[('BMP files', '*.bmp')])

    # Verifica se il file è stato selezionato
    if file_path:
        # Carica l'immagine selezionata
        img = Image.open(file_path)

        # setta range slider
        m = min(img.height, img.width)
        F_slider.configure(from_=1, to=m)
        d_slider.configure(from_=1, to=m - 2)
        # setta valore slider
        F_slider.set(int(m / 3))
        d_slider.set(int(m / 3) - 2)

        # peso dell'immagine caricata
        original_size = os.path.getsize(file_path)
        original_size_label.configure(text=f"Original Size: {format_size(original_size)}")

        # reset dei tempi in caso di caricamento non iniziale
        compressed_time_label.configure(text="Compressed Time: ")

        # Mostra l'immagine nella GUI
        img.thumbnail((500, 500))  # Ridimensiona l'immagine per adattarla alla GUI
        img_tk = ImageTk.PhotoImage(img)
        img_label.configure(image=img_tk)
        img_label.image = img_tk

        # Bottone compressione abilitato
        compress_button.configure(state=tk.NORMAL)
        # richiama la funzione compress_button_clicked() quando cliccato
        compress_button.config(command=lambda: compress_button_clicked(file_path))


def compress_button_clicked(image_path):

    # Ottieni i valori di F e d dai due slider
    F = int(F_slider.get())
    d = int(d_slider.get())
    # ulteriore check
    if d < 0 | d > F - 2:
        d = F - 2

    # misurazione del tempo
    tic = time.perf_counter()

    # chiamata alla funzione di compressione
    compressed_img = compress_image(image_path, F, d)

    # stop tempo e approssimazione
    toc = time.perf_counter()
    time_compr = round ( float ( toc - tic ) , 4)
    compressed_time_label.configure(text="Compressed Time: " + str(time_compr) + "s")

    # Mostra l'immagine compressa nella GUI
    compressed_img.thumbnail((500, 500))
    compressed_img_tk = ImageTk.PhotoImage(compressed_img)
    compressed_img_label.configure(image=compressed_img_tk)
    compressed_img_label.image = compressed_img_tk

    # Mostra gli istogrammi
    show_original_histogram(image_path, F, d)
    show_compressed_histogram(compressed_img)

    # Aggiorna la visualizzazione della figura
    plt.tight_layout()
    plt.show()

def compress_image(image_path, F, d):
    with open(image_path, 'rb') as f:
        # se RGB, convertila
        img = Image.open(f).convert('L')

    # Ottieni le dimensioni dell'immagine
    width, height = img.size
    min_dim = min(width, height)

    # Ulteriore verifica dei parametri e update eventuale degli slider
    if F > min_dim:
        F = min_dim
        F_slider.set(min_dim)
    if d > F - 2:
        d = F - 2
        d_slider.set(F - 2)

    # Calcola il numero di blocchi in larghezza e altezza - arrotonda i valori = pixel in eccesso scartati
    num_blocks_w = int(width / F)
    num_blocks_h = int(height / F)

    # dimensione dell'immagine compressa
    new_width = num_blocks_w * F
    new_height = num_blocks_h * F

    # Da immagine ad arrampu numpy
    img_arr = np.array(img, dtype=np.float32)

    # Pre-allochiamo un array numpy di 0
    compressed_arr = np.zeros((new_height, new_width), dtype=np.float32)

    # Applica la compressione per ogni blocco
    for i in range(num_blocks_h):
        for j in range(num_blocks_w):
            # Estrai il blocco corrente
            block = img_arr[i * F:(i + 1) * F, j * F:(j + 1) * F]
            # dct2 sul blocco
            dct_block = dctn(block, norm='ortho')
            # tronca le alte frequenze d
            for k in range(F):
                for l in range(F):
                    if k + l >= d:
                        dct_block[k, l] = 0

            # Applica la DCT2 inversa
            idctn_block = idctn(dct_block, norm='ortho')

            # Aggiungi il blocco compresso all'array dell'immagine finale
            compressed_arr[i * F : (i + 1) * F, j * F : (j + 1) * F] = idctn_block

    # Arrotonda i valori nell'intervallo 0-255
    compressed_arr = np.round(compressed_arr).clip(0, 255)

    # Converti l'array ottenuto in immagine
    compressed_img = Image.fromarray(compressed_arr.astype(np.uint8))

    return compressed_img

def show_original_histogram(image_path, F, d):
    # Calcola l'istogramma dell'immagine originale
    with open(image_path, 'rb') as f:
        img = Image.open(f).convert('L')
    img_arr = np.array(img)
    hist = np.histogram(img_arr.flatten(), bins=256, range=[0, 256])

    # Visualizza l'istogramma
    plt.figure()
    plt.title("Original vs Compressed Image Histogram with F =" + str(F) + " d =" + str(d))
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")
    plt.bar(hist[1][:-1], hist[0], width=1, alpha=0.5, label="Original Image")


def show_compressed_histogram(compressed_img):
    # Calcola l'istogramma dell'immagine compressa
    compressed_arr = np.array(compressed_img)
    hist = np.histogram(compressed_arr.flatten(), bins=256, range=[0, 256])

    # Visualizza l'istogramma
    plt.bar(hist[1][:-1], hist[0], width=1, alpha=0.5, label="Compressed Image")
    plt.legend()

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

# Label per il peso dell'immagine originale
original_size_label = ttk.Label(root, text="Original Size: ")
original_size_label.pack(padx=10, pady=(0, 5))

# Label per il tempo di compressione
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